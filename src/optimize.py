from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, cast

import joblib
import mlflow
import mlflow.sklearn as mlflow_sklearn
import optuna
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import (
    build_model_pipeline,
    compute_classification_metrics,
    compute_primary_metric,
    file_md5,
    get_git_commit_hash,
    load_hydra_config,
    load_processed_data,
    resolve_path,
    resolve_tracking_uri,
    set_global_seed,
)


def evaluate_holdout(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    metric: str,
) -> float:
    model.fit(X_train, y_train)
    return compute_primary_metric(model, X_valid, y_valid, metric)


def evaluate_cv(
    model, X: pd.DataFrame, y: pd.Series, metric: str, seed: int, n_splits: int = 5
) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []

    for train_idx, valid_idx in cv.split(X, y):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[valid_idx]
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[valid_idx]
        model_clone = clone(model)
        model_clone.fit(X_tr, y_tr)
        scores.append(compute_primary_metric(model_clone, X_val, y_val, metric))

    return float(sum(scores) / len(scores))


def make_sampler(
    sampler_name: str, seed: int, grid_space: dict[str, Any] | None = None
) -> optuna.samplers.BaseSampler:
    normalized = sampler_name.lower()
    if normalized == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if normalized == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if normalized == "grid":
        if not grid_space:
            raise ValueError("GridSampler requires a non-empty search space.")
        return optuna.samplers.GridSampler(search_space=grid_space)
    raise ValueError("sampler must be one of: tpe, random, grid")


def suggest_params(
    trial: optuna.Trial, model_type: str, cfg: DictConfig
) -> dict[str, Any]:
    if model_type == "random_forest":
        space = cfg.hpo.random_forest
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                int(space.n_estimators.low),
                int(space.n_estimators.high),
            ),
            "max_depth": trial.suggest_int(
                "max_depth", int(space.max_depth.low), int(space.max_depth.high)
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                int(space.min_samples_split.low),
                int(space.min_samples_split.high),
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                int(space.min_samples_leaf.low),
                int(space.min_samples_leaf.high),
            ),
        }

    if model_type == "logistic_regression":
        space = cfg.hpo.logistic_regression
        return {
            "C": trial.suggest_float(
                "C", float(space.C.low), float(space.C.high), log=True
            ),
            "solver": trial.suggest_categorical("solver", list(space.solver)),
            "penalty": trial.suggest_categorical("penalty", list(space.penalty)),
        }

    raise ValueError(f"Unknown model.type='{model_type}'.")


def build_grid_space(cfg: DictConfig) -> dict[str, Any] | None:
    if str(cfg.hpo.sampler).lower() != "grid":
        return None

    if str(cfg.model.type) == "random_forest":
        space = cfg.hpo.grid.random_forest
        return {
            "n_estimators": list(space.n_estimators),
            "max_depth": list(space.max_depth),
            "min_samples_split": list(space.min_samples_split),
            "min_samples_leaf": list(space.min_samples_leaf),
        }

    if str(cfg.model.type) == "logistic_regression":
        space = cfg.hpo.grid.logistic_regression
        return {
            "C": list(space.C),
            "solver": list(space.solver),
            "penalty": list(space.penalty),
        }

    return None


def objective_factory(cfg: DictConfig, X_train: pd.DataFrame, y_train: pd.Series):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, str(cfg.model.type), cfg)
        trial_start = time.perf_counter()

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", str(cfg.model.type))
            mlflow.set_tag("sampler", str(cfg.hpo.sampler))
            mlflow.set_tag("seed", int(cfg.seed))
            mlflow.log_params(params)

            model = build_model_pipeline(
                str(cfg.model.type), params=params, X_sample=X_train, seed=int(cfg.seed)
            )

            if bool(cfg.hpo.use_cv):
                score = evaluate_cv(
                    model,
                    X_train,
                    y_train,
                    metric=str(cfg.hpo.metric),
                    seed=int(cfg.seed),
                    n_splits=int(cfg.hpo.cv_folds),
                )
                mlflow.set_tag("evaluation_strategy", f"cv_{int(cfg.hpo.cv_folds)}fold")
            else:
                X_subtrain, X_valid, y_subtrain, y_valid = train_test_split(
                    X_train,
                    y_train,
                    test_size=float(cfg.hpo.validation_size),
                    random_state=int(cfg.seed),
                    stratify=y_train,
                )
                score = evaluate_holdout(
                    model,
                    X_subtrain,
                    y_subtrain,
                    X_valid,
                    y_valid,
                    metric=str(cfg.hpo.metric),
                )
                mlflow.set_tag("evaluation_strategy", "holdout_validation")

            duration_seconds = time.perf_counter() - trial_start
            mlflow.log_metric(str(cfg.hpo.metric), float(score))
            mlflow.log_metric("trial_duration_seconds", float(duration_seconds))
            trial.set_user_attr("duration_seconds", float(duration_seconds))
            return float(score)

    return objective


def register_model_if_enabled(model_uri: str, model_name: str, stage: str) -> None:
    client = MlflowClient()
    model_version = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(
        name=model_name, version=model_version.version, stage=stage
    )
    client.set_model_version_tag(
        model_name, model_version.version, "registered_by", "lab3"
    )
    client.set_model_version_tag(model_name, model_version.version, "stage", stage)


def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))

    tracking_uri = resolve_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))

    X_train, X_test, y_train, y_test = load_processed_data(str(cfg.data.processed_path))
    grid_space = build_grid_space(cfg)
    sampler = make_sampler(
        str(cfg.hpo.sampler), seed=int(cfg.seed), grid_space=grid_space
    )

    best_model_path = resolve_path(str(cfg.output.best_model_path))
    best_params_path = resolve_path(str(cfg.output.best_params_path))
    summary_path = resolve_path(str(cfg.output.summary_path))
    trial_history_path = resolve_path(str(cfg.output.trial_history_path))
    final_metrics_path = resolve_path(str(cfg.output.final_metrics_path))

    for path in (
        best_model_path,
        best_params_path,
        summary_path,
        trial_history_path,
        final_metrics_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    dataset_md5 = file_md5(str(cfg.data.raw_path))
    git_commit_hash = get_git_commit_hash()

    study_start = time.perf_counter()
    with mlflow.start_run(run_name=str(cfg.mlflow.parent_run_name)) as parent_run:
        resolved_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        mlflow.set_tag("model_type", str(cfg.model.type))
        mlflow.set_tag("sampler", str(cfg.hpo.sampler))
        mlflow.set_tag("seed", int(cfg.seed))
        mlflow.set_tag("git_commit_hash", git_commit_hash)
        mlflow.set_tag("raw_data_md5", dataset_md5)
        mlflow.log_params(
            {
                "n_trials": int(cfg.hpo.n_trials),
                "metric": str(cfg.hpo.metric),
                "direction": str(cfg.hpo.direction),
                "use_cv": bool(cfg.hpo.use_cv),
                "cv_folds": int(cfg.hpo.cv_folds),
                "validation_size": float(cfg.hpo.validation_size),
            }
        )
        mlflow.log_dict(resolved_cfg, "config_resolved.json")

        study = optuna.create_study(
            study_name=str(cfg.study_name),
            direction=str(cfg.hpo.direction),
            sampler=sampler,
        )
        objective = objective_factory(cfg, X_train, y_train)
        study.optimize(objective, n_trials=int(cfg.hpo.n_trials))
        duration_seconds = time.perf_counter() - study_start

        best_trial = study.best_trial
        best_params = dict(best_trial.params)

        best_model = build_model_pipeline(
            str(cfg.model.type), best_params, X_train, int(cfg.seed)
        )
        best_model.fit(X_train, y_train)
        final_metric_value = compute_primary_metric(
            best_model, X_test, y_test, str(cfg.hpo.metric)
        )
        final_metrics = compute_classification_metrics(
            best_model, X_train, y_train, X_test, y_test
        )

        trial_rows = []
        trial_values: list[float] = []
        for trial in study.trials:
            row = {
                "number": int(trial.number),
                "state": str(trial.state),
                "value": None if trial.value is None else float(trial.value),
                "duration_seconds": float(
                    trial.user_attrs.get("duration_seconds", 0.0)
                ),
                "params": json.dumps(trial.params, ensure_ascii=False),
            }
            trial_rows.append(row)
            if trial.value is not None:
                trial_values.append(float(trial.value))

        trial_history = pd.DataFrame(trial_rows)
        trial_history.to_csv(trial_history_path, index=False)

        summary = {
            "study_name": str(cfg.study_name),
            "model_type": str(cfg.model.type),
            "sampler": str(cfg.hpo.sampler),
            "metric": str(cfg.hpo.metric),
            "direction": str(cfg.hpo.direction),
            "seed": int(cfg.seed),
            "git_commit_hash": git_commit_hash,
            "raw_data_md5": dataset_md5,
            "n_trials_requested": int(cfg.hpo.n_trials),
            "n_trials_completed": int(len(study.trials)),
            "best_value": float(best_trial.value),
            "best_params": best_params,
            "final_metric_name": str(cfg.hpo.metric),
            "final_metric_value": float(final_metric_value),
            "final_metrics": final_metrics,
            "mean_trial_value": (
                float(sum(trial_values) / len(trial_values)) if trial_values else 0.0
            ),
            "median_trial_value": (
                float(pd.Series(trial_values).median()) if trial_values else 0.0
            ),
            "std_trial_value": (
                float(pd.Series(trial_values).std(ddof=0)) if trial_values else 0.0
            ),
            "duration_seconds": float(duration_seconds),
            "use_cv": bool(cfg.hpo.use_cv),
            "cv_folds": int(cfg.hpo.cv_folds),
            "validation_size": float(cfg.hpo.validation_size),
            "tracking_uri": tracking_uri,
        }

        best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
        final_metrics_path.write_text(
            json.dumps(final_metrics, indent=2), encoding="utf-8"
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        joblib.dump(best_model, best_model_path)

        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best_trial.value))
        mlflow.log_metric(f"final_{cfg.hpo.metric}", float(final_metric_value))
        mlflow.log_metric("study_duration_seconds", float(duration_seconds))
        for metric_name, metric_value in final_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.log_dict(best_params, best_params_path.name)
        mlflow.log_dict(summary, summary_path.name)
        mlflow.log_artifact(str(trial_history_path))
        mlflow.log_artifact(str(final_metrics_path))
        mlflow.log_artifact(str(best_model_path))

        if bool(cfg.mlflow.log_model):
            mlflow_sklearn.log_model(best_model, artifact_path="model")

        if bool(cfg.mlflow.register_model):
            try:
                model_uri = f"runs:/{parent_run.info.run_id}/model"
                register_model_if_enabled(
                    model_uri, str(cfg.mlflow.model_name), stage=str(cfg.mlflow.stage)
                )
                mlflow.set_tag("model_registry", "registered")
            except Exception as error:
                mlflow.set_tag("model_registry", f"skipped: {error}")

    print("Optimization finished successfully.")
    print(f"Best params: {best_params_path}")
    print(f"Best model: {best_model_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main(load_hydra_config())

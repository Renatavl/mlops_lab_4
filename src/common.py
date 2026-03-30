from __future__ import annotations

import hashlib
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def resolve_tracking_uri(tracking_uri: str | None) -> str:
    if tracking_uri and str(tracking_uri).lower() != "none":
        return tracking_uri

    tracking_dir = PROJECT_ROOT / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    return tracking_dir.resolve().as_uri()


def load_hydra_config(overrides: list[str] | None = None, config_name: str = "config"):
    config_dir = PROJECT_ROOT / "config"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        return compose(config_name=config_name, overrides=overrides or sys.argv[1:])


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit_hash() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def file_md5(path: str | Path) -> str:
    digest = hashlib.md5()
    with resolve_path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value)


def _ensure_series(value: Any, name: str = "target") -> pd.Series:
    if isinstance(value, pd.Series):
        return value.copy()
    return pd.Series(value, name=name)


def load_processed_data(
    path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    abs_path = resolve_path(path)
    suffix = abs_path.suffix.lower()

    if suffix in {".pkl", ".pickle"}:
        obj = joblib.load(abs_path)
        if isinstance(obj, dict):
            if {"X_train", "X_test", "y_train", "y_test"}.issubset(obj.keys()):
                return (
                    _ensure_frame(obj["X_train"]),
                    _ensure_frame(obj["X_test"]),
                    _ensure_series(obj["y_train"]),
                    _ensure_series(obj["y_test"]),
                )

            if {"X", "y"}.issubset(obj.keys()):
                X = _ensure_frame(obj["X"])
                y = _ensure_series(obj["y"])
                split = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                return (
                    _ensure_frame(split[0]),
                    _ensure_frame(split[1]),
                    _ensure_series(split[2]),
                    _ensure_series(split[3]),
                )

        if isinstance(obj, pd.DataFrame):
            if "target" not in obj.columns:
                raise ValueError("Processed DataFrame must contain a 'target' column.")
            X = obj.drop(columns=["target"])
            y = obj["target"]
            split = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            return (
                _ensure_frame(split[0]),
                _ensure_frame(split[1]),
                _ensure_series(split[2]),
                _ensure_series(split[3]),
            )

        raise ValueError(
            "Unsupported pickle format. Expected dict or pandas.DataFrame."
        )

    if suffix == ".csv":
        df = pd.read_csv(abs_path)
        if "target" not in df.columns:
            raise ValueError("CSV must contain a 'target' column.")
        X = df.drop(columns=["target"])
        y = df["target"]
        split = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return (
            _ensure_frame(split[0]),
            _ensure_frame(split[1]),
            _ensure_series(split[2]),
            _ensure_series(split[3]),
        )

    raise ValueError("Unsupported data format. Use .pickle/.pkl or .csv.")


def build_preprocessor(
    X_frame: pd.DataFrame, scale_numeric: bool = False
) -> ColumnTransformer:
    categorical_cols = X_frame.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric_cols = X_frame.select_dtypes(include=["number"]).columns.tolist()

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_model_pipeline(
    model_type: str, params: dict[str, Any], X_sample: pd.DataFrame, seed: int
) -> Pipeline:
    if model_type == "random_forest":
        max_depth = params.get("max_depth")
        if max_depth in ("None", "null", None):
            max_depth = None
        estimator = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 250)),
            max_depth=None if max_depth is None else int(max_depth),
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        preprocessor = build_preprocessor(X_sample, scale_numeric=False)
    elif model_type == "logistic_regression":
        solver = cast(
            Literal[
                "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
            ],
            str(params.get("solver", "liblinear")),
        )
        penalty = cast(
            Literal["l1", "l2", "elasticnet"] | None, params.get("penalty", "l2")
        )
        estimator = LogisticRegression(
            C=float(params.get("C", 1.0)),
            solver=solver,
            penalty=penalty,
            max_iter=1000,
            random_state=seed,
        )
        preprocessor = build_preprocessor(X_sample, scale_numeric=True)
    else:
        raise ValueError(f"Unknown model.type='{model_type}'.")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def compute_primary_metric(
    model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series, metric: str
) -> float:
    y_pred = model.predict(X_eval)

    if metric == "f1":
        average = "binary" if y_eval.nunique() == 2 else "weighted"
        return float(f1_score(y_eval, y_pred, average=average, zero_division=0))

    if metric == "roc_auc":
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_eval)
            if y_eval.nunique() > 2:
                return float(
                    roc_auc_score(
                        y_eval, probabilities, multi_class="ovr", average="weighted"
                    )
                )
            return float(roc_auc_score(y_eval, probabilities[:, 1]))

        scores = model.decision_function(X_eval)
        return float(roc_auc_score(y_eval, scores))

    raise ValueError("Unsupported metric. Use 'f1' or 'roc_auc'.")


def compute_classification_metrics(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    average = "weighted"

    metrics = {
        "accuracy_train": float(accuracy_score(y_train, y_pred_train)),
        "accuracy_test": float(accuracy_score(y_test, y_pred_test)),
        "f1_train": float(
            f1_score(y_train, y_pred_train, average=average, zero_division=0)
        ),
        "f1_test": float(
            f1_score(y_test, y_pred_test, average=average, zero_division=0)
        ),
        "precision_train": float(
            precision_score(y_train, y_pred_train, average=average, zero_division=0)
        ),
        "precision_test": float(
            precision_score(y_test, y_pred_test, average=average, zero_division=0)
        ),
        "recall_train": float(
            recall_score(y_train, y_pred_train, average=average, zero_division=0)
        ),
        "recall_test": float(
            recall_score(y_test, y_pred_test, average=average, zero_division=0)
        ),
    }

    if hasattr(model, "predict_proba"):
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc_train"] = float(roc_auc_score(y_train, train_proba))
        metrics["roc_auc_test"] = float(roc_auc_score(y_test, test_proba))

    return metrics


def get_feature_names(model: Pipeline) -> list[str] | None:
    try:
        return model.named_steps["preprocessor"].get_feature_names_out().tolist()
    except Exception:
        return None

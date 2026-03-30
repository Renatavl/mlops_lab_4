from __future__ import annotations

import json
from typing import Any, cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as mlflow_sklearn
from omegaconf import OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from common import (
    build_model_pipeline,
    compute_classification_metrics,
    file_md5,
    get_git_commit_hash,
    load_hydra_config,
    load_processed_data,
    resolve_path,
    resolve_tracking_uri,
    set_global_seed,
)


def main() -> None:
    cfg = load_hydra_config()
    set_global_seed(int(cfg.seed))

    X_train, X_test, y_train, y_test = load_processed_data(str(cfg.data.processed_path))

    params_path = cfg.training.params_path
    if params_path:
        resolved_params_path = resolve_path(str(params_path))
        model_params = cast(
            dict[str, Any], json.loads(resolved_params_path.read_text(encoding="utf-8"))
        )
        params_source = str(resolved_params_path)
    else:
        model_params = cast(
            dict[str, Any],
            OmegaConf.to_container(cfg.model.default_params, resolve=True),
        )
        params_source = "config.model.default_params"

    model = build_model_pipeline(
        str(cfg.model.type), model_params, X_train, int(cfg.seed)
    )
    model.fit(X_train, y_train)
    metrics = compute_classification_metrics(model, X_train, y_train, X_test, y_test)

    trained_model_path = resolve_path(str(cfg.output.trained_model_path))
    metrics_path = resolve_path(str(cfg.output.final_metrics_path))
    ci_model_path = resolve_path(str(cfg.output.ci_model_path))
    ci_metrics_path = resolve_path(str(cfg.output.ci_metrics_path))
    ci_confusion_matrix_path = resolve_path(str(cfg.output.ci_confusion_matrix_path))

    trained_model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    ci_model_path.parent.mkdir(parents=True, exist_ok=True)
    ci_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    ci_confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, trained_model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, ci_model_path)
    ci_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    labels = sorted(y_test.unique().tolist())
    cm = confusion_matrix(y_test, model.predict(X_test), labels=labels)
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=axis, colorbar=False
    )
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    figure.savefig(ci_confusion_matrix_path, dpi=150)
    plt.close(figure)

    if bool(cfg.training.log_to_mlflow):
        mlflow.set_tracking_uri(resolve_tracking_uri(cfg.mlflow.tracking_uri))
        mlflow.set_experiment(str(cfg.mlflow.experiment_name))
        resolved_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))

        with mlflow.start_run(run_name=str(cfg.training.run_name)):
            mlflow.set_tag("model_type", str(cfg.model.type))
            mlflow.set_tag("params_source", params_source)
            mlflow.set_tag("git_commit_hash", get_git_commit_hash())
            mlflow.set_tag("raw_data_md5", file_md5(str(cfg.data.raw_path)))
            mlflow.log_params({key: value for key, value in model_params.items()})
            mlflow.log_params(
                {
                    "seed": int(cfg.seed),
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                }
            )
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.log_dict(resolved_cfg, "config_resolved.json")
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(trained_model_path))
            mlflow.log_artifact(str(ci_metrics_path))
            mlflow.log_artifact(str(ci_confusion_matrix_path))
            if bool(cfg.mlflow.log_model):
                mlflow_sklearn.log_model(model, artifact_path="trained_model")

    print("Training finished successfully.")
    print(f"Model artifact: {trained_model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"CI model artifact: {ci_model_path}")
    print(f"CI metrics: {ci_metrics_path}")
    print(f"CI confusion matrix: {ci_confusion_matrix_path}")


if __name__ == "__main__":
    main()

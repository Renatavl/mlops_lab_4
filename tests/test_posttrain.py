import json
import os


def test_artifacts_exist() -> None:
    assert os.path.exists("model.pkl"), "model.pkl not found"
    assert os.path.exists("metrics.json"), "metrics.json not found"
    assert os.path.exists("confusion_matrix.png"), "confusion_matrix.png not found"


def test_quality_gate_f1() -> None:
    threshold = float(os.getenv("F1_THRESHOLD", "0.70"))

    with open("metrics.json", "r", encoding="utf-8") as file:
        metrics = json.load(file)

    f1 = float(metrics["f1_test"])
    assert f1 >= threshold, f"Quality Gate not passed: f1={f1:.4f} < {threshold:.2f}"

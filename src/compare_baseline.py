from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _to_float(value: Any) -> float:
    return float(value)


def main() -> None:
    baseline_path = Path("baseline/metrics.json")
    current_path = Path("metrics.json")
    output_path = Path("reports/baseline_comparison.json")

    baseline = _load_json(baseline_path)
    current = _load_json(current_path)

    metric_names = [
        "accuracy_test",
        "f1_test",
        "precision_test",
        "recall_test",
        "roc_auc_test",
    ]
    comparison: dict[str, dict[str, float]] = {}

    for name in metric_names:
        baseline_value = _to_float(baseline.get(name, 0.0))
        current_value = _to_float(current.get(name, 0.0))
        comparison[name] = {
            "baseline": baseline_value,
            "current": current_value,
            "delta": current_value - baseline_value,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("| Metric | Baseline | Current | Δ |")
    print("|---|---:|---:|---:|")
    for name in metric_names:
        values = comparison[name]
        print(
            f"| {name} | {values['baseline']:.4f} | {values['current']:.4f} | {values['delta']:+.4f} |"
        )


if __name__ == "__main__":
    main()

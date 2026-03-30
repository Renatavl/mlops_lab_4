from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Optuna sampler summaries")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Summary JSON files produced by optimize.py",
    )
    parser.add_argument(
        "--output-json", required=True, help="Where to save comparison JSON"
    )
    parser.add_argument(
        "--output-md", required=True, help="Where to save comparison Markdown"
    )
    return parser.parse_args()


def read_summary(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_markdown(comparison: dict) -> str:
    rows = [
        "# Sampler comparison",
        "",
        "| Study | Sampler | Best validation metric | Final test metric | Mean trial metric | Median trial metric | Duration (s) | Trials |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in comparison["studies"]:
        rows.append(
            f"| {item['study_name']} | {item['sampler']} | {item['best_value']:.4f} | {item['final_metric_value']:.4f} | {item['mean_trial_value']:.4f} | {item['median_trial_value']:.4f} | {item['duration_seconds']:.2f} | {item['n_trials_completed']} |"
        )

    rows.extend(
        [
            "",
            f"Winner by best validation metric: **{comparison['winner']['study_name']}** ({comparison['winner']['sampler']}).",
            f"Best validation metric: **{comparison['winner']['best_value']:.4f}**.",
            f"Best final test metric: **{comparison['best_final']['study_name']}** with **{comparison['best_final']['final_metric_value']:.4f}**.",
        ]
    )
    return "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    summaries = [read_summary(path) for path in args.inputs]
    summaries.sort(key=lambda item: item["best_value"], reverse=True)

    best_final = max(summaries, key=lambda item: item["final_metric_value"])
    comparison = {
        "winner": {
            "study_name": summaries[0]["study_name"],
            "sampler": summaries[0]["sampler"],
            "best_value": summaries[0]["best_value"],
        },
        "best_final": {
            "study_name": best_final["study_name"],
            "sampler": best_final["sampler"],
            "final_metric_value": best_final["final_metric_value"],
        },
        "studies": summaries,
    }

    output_json_path = Path(args.output_json)
    output_md_path = Path(args.output_md)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    output_json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    output_md_path.write_text(build_markdown(comparison), encoding="utf-8")

    print(f"Comparison JSON: {output_json_path}")
    print(f"Comparison Markdown: {output_md_path}")


if __name__ == "__main__":
    main()

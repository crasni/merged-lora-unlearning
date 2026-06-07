#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


RUNS = {
    0.05: "1_5b_scale_05",
    0.10: "1_5b_scale_10",
    0.20: "1_5b_paper",
    0.40: "1_5b_scale_40",
}
METHODS = ("ga", "grad_diff", "npo", "npo_grad_diff", "simnpo", "simnpo_grad_diff")


def main() -> None:
    summaries = {}
    for ratio, run in RUNS.items():
        path = Path("outputs/runs") / run / "report" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed report: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries[ratio] = summary
    rows = []
    for ratio, summary in summaries.items():
        by_model = {row["model"]: row for row in summary}
        for model in METHODS:
            row = by_model.get(model)
            if row is None:
                rows.append(
                    {
                        "forget_request_ratio": ratio,
                        "model": model,
                        "result": "failed",
                        "forget_match": None,
                        "retain_match": None,
                        "privacy_distance_to_oracle": None,
                        "general_utility_match": None,
                    }
                )
                continue
            rows.append(
                {
                    "forget_request_ratio": ratio,
                    "model": row["model"],
                    "result": row["result"],
                    "forget_match": row["c2_forget_match"],
                    "retain_match": row["c4_retain_match"],
                    "privacy_distance_to_oracle": row["c3_distance_to_oracle"],
                    "general_utility_match": row["general_utility_match"],
                }
            )
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "1_5b_scalability.json"
    csv_path = output_dir / "1_5b_scalability.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)


if __name__ == "__main__":
    main()

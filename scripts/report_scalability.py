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


def main() -> None:
    summaries = {}
    common_models = None
    for ratio, run in RUNS.items():
        path = Path("outputs/runs") / run / "report" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed report: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        summaries[ratio] = summary
        models = {row["model"] for row in summary} - {"base", "acquisition_adapter"}
        common_models = models if common_models is None else common_models & models

    rows = []
    for ratio, summary in summaries.items():
        for row in summary:
            if row["model"] not in common_models:
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

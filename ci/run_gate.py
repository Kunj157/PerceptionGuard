from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.regression_checker import get_baseline, init_db, overall_verdict


def run_gate(commit_sha: str, output_path: Path) -> dict:
    """Run the full evaluation pipeline and return gate results.

    This is a placeholder that works with synthetic/empty data until
    real models and datasets are available.
    """
    db_path = Path("data/metrics.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(db_path)

    baseline = get_baseline(conn)

    results = {
        "commit_sha": commit_sha,
        "overall": "pass",
        "details": [],
    }

    if baseline is not None and not baseline.empty:
        details_df = baseline[["weather", "scene", "time_of_day", "mean_ap", "verdict"]].copy()
        details_df["delta_ap"] = 0.0
        results["details"] = details_df.to_dict(orient="records")

    verdict = overall_verdict(
        pd.DataFrame(results["details"]) if results["details"]
        else pd.DataFrame([{"verdict": "pass"}])
    )
    results["overall"] = verdict

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="PerceptionGuard CI gate runner")
    parser.add_argument("--commit-sha", required=True, help="Git commit SHA")
    parser.add_argument("--output", default="gate_result.json", help="Output JSON path")
    args = parser.parse_args()

    result = run_gate(args.commit_sha, Path(args.output))
    print(f"Gate result: {result['overall']}")
    sys.exit(0 if result["overall"] != "critical" else 1)


if __name__ == "__main__":
    main()

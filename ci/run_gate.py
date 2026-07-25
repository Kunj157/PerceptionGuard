from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from src.coverage_analyzer import compute_coverage, load_thresholds
from src.eval_harness import aggregate_bucket_metrics, compute_map_per_bucket
from src.regression_checker import (
    compare_against_baseline,
    get_baseline,
    init_db,
    insert_run,
    overall_verdict,
    load_thresholds as load_regression_thresholds,
)
from src.scenario_tagger import build_metadata
from src.test_suite import build_manifest, validate_manifest


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

    results["overall"] = "pass"
    results["details"] = []

    verdict = overall_verdict(
        __import__("pandas").DataFrame(results["details"])
        if results["details"]
        else __import__("pandas").DataFrame([{"verdict": "pass"}])
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

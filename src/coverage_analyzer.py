from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

DEFAULT_THRESHOLDS = {
    "critical": 0,
    "low": 10,
}


def load_thresholds(config_path: Path | None = None) -> dict[str, int]:
    """Load coverage thresholds from YAML config. Falls back to defaults."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("coverage_thresholds", DEFAULT_THRESHOLDS)
    return DEFAULT_THRESHOLDS.copy()


def compute_coverage(df: pd.DataFrame, thresholds: dict[str, int] | None = None) -> pd.DataFrame:
    """Compute per-bucket coverage counts and flag status.

    Args:
        df: Metadata DataFrame with weather, scene, time_of_day columns.
        thresholds: Mapping of status -> minimum sample count.
                    Buckets below 'low' threshold but above 'critical' get 'low'.
                    Buckets at or below 'critical' threshold get 'critical'.
                    Everything else gets 'adequate'.

    Returns:
        DataFrame with columns: weather, scene, time_of_day, count, status.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    critical_max = thresholds.get("critical", 0)
    low_max = thresholds.get("low", 10)

    group_cols = ["weather", "scene", "time_of_day"]
    counts = df.groupby(group_cols, dropna=False).size().reset_index(name="count")

    def _flag(n: int) -> str:
        if n <= critical_max:
            return "critical"
        if n <= low_max:
            return "low"
        return "adequate"

    counts["status"] = counts["count"].apply(_flag)
    return counts.sort_values(["count", "weather", "scene", "time_of_day"]).reset_index(drop=True)


def coverage_summary(coverage_table: pd.DataFrame) -> dict[str, int]:
    """Return counts of buckets per status."""
    return coverage_table["status"].value_counts().to_dict()


def save_coverage_table(table: pd.DataFrame, output_path: Path) -> None:
    """Save coverage table to CSV."""
    table.to_csv(output_path, index=False)

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_THRESHOLDS = {
    "critical_drop": 0.10,
    "warning_drop": 0.05,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_sha TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    weather TEXT NOT NULL,
    scene TEXT NOT NULL,
    time_of_day TEXT NOT NULL,
    mean_ap REAL NOT NULL,
    verdict TEXT NOT NULL
);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create or open the metrics database and ensure schema exists."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    return conn


def insert_run(
    conn: sqlite3.Connection,
    commit_sha: str,
    weather: str,
    scene: str,
    time_of_day: str,
    mean_ap: float,
    verdict: str,
) -> None:
    """Insert a single scenario bucket result into the runs table."""
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO runs (commit_sha, timestamp, weather, scene, time_of_day, mean_ap, verdict) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (commit_sha, ts, weather, scene, time_of_day, mean_ap, verdict),
    )
    conn.commit()


def get_baseline(conn: sqlite3.Connection) -> pd.DataFrame | None:
    """Get the most recent run's results as baseline.

    Returns None if no prior runs exist.
    """
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY timestamp DESC", conn)
    if df.empty:
        return None
    latest_ts = df["timestamp"].iloc[0]
    return df[df["timestamp"] == latest_ts].copy()


def load_thresholds(config_path: Path | None = None) -> dict[str, float]:
    """Load regression thresholds from YAML. Falls back to defaults."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("regression_thresholds", DEFAULT_THRESHOLDS)
    return DEFAULT_THRESHOLDS.copy()


def compare_against_baseline(
    current: pd.DataFrame,
    baseline: pd.DataFrame | None,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compare current results against baseline and assign verdicts.

    Args:
        current: DataFrame with weather, scene, time_of_day, mean_ap.
        baseline: DataFrame with same columns from a prior run. None = first run.
        thresholds: dict with 'critical_drop' and 'warning_drop' (fractional).

    Returns:
        DataFrame with current metrics plus 'delta_ap' and 'verdict' columns.
        Verdict: 'pass', 'warning', or 'critical'.
    """
    if thresholds is None:
        thresholds = load_thresholds()

    critical_drop = thresholds.get("critical_drop", 0.10)
    warning_drop = thresholds.get("warning_drop", 0.05)

    result = current.copy()
    result["delta_ap"] = 0.0
    result["verdict"] = "pass"

    if baseline is None:
        result["verdict"] = "pass"
        return result

    group_cols = ["weather", "scene", "time_of_day"]
    baseline_map = baseline.set_index(group_cols)["mean_ap"].to_dict()

    for idx, row in result.iterrows():
        key = (row["weather"], row["scene"], row["time_of_day"])
        if key in baseline_map:
            delta = baseline_map[key] - row["mean_ap"]
            result.at[idx, "delta_ap"] = delta
            if delta >= critical_drop:
                result.at[idx, "verdict"] = "critical"
            elif delta >= warning_drop:
                result.at[idx, "verdict"] = "warning"
            else:
                result.at[idx, "verdict"] = "pass"
        else:
            result.at[idx, "verdict"] = "pass"

    return result


def overall_verdict(results: pd.DataFrame) -> str:
    """Return 'pass' unless any bucket is critical or warning."""
    if (results["verdict"] == "critical").any():
        return "critical"
    if (results["verdict"] == "warning").any():
        return "warning"
    return "pass"

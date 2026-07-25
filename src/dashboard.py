from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def get_runs(db_path: Path) -> pd.DataFrame:
    """Load all runs from the metrics database."""
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY timestamp", conn)
    conn.close()
    return df


def get_latest_run(db_path: Path) -> pd.DataFrame:
    """Load the most recent run's results."""
    df = get_runs(db_path)
    if df.empty:
        return df
    latest_ts = df["timestamp"].iloc[-1]
    return df[df["timestamp"] == latest_ts].copy()


def build_accuracy_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot runs into a timestamp x scenario matrix of mAP values."""
    if runs_df.empty:
        return runs_df
    runs_df["scenario"] = (
        runs_df["weather"] + " / " + runs_df["scene"] + " / " + runs_df["time_of_day"]
    )
    pivot = runs_df.pivot_table(
        index="timestamp", columns="scenario", values="mean_ap", aggfunc="first"
    )
    return pivot


def build_coverage_heatmap(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Build a weather x scene matrix of mean AP from the latest run."""
    if runs_df.empty:
        return runs_df
    latest_ts = runs_df["timestamp"].iloc[-1]
    latest = runs_df[runs_df["timestamp"] == latest_ts]
    heatmap = latest.pivot_table(
        index="weather", columns="scene", values="mean_ap", aggfunc="first"
    )
    return heatmap

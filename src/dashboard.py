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
    """Load the most recent run's results (all buckets from latest batch)."""
    df = get_runs(db_path)
    if df.empty:
        return df
    if df["commit_sha"].nunique() == 1:
        return df.copy()
    latest_commit = df["commit_sha"].iloc[-1]
    return df[df["commit_sha"] == latest_commit].copy()


def get_coverage_data(data_dir: Path) -> pd.DataFrame:
    """Load coverage table from CSV."""
    coverage_path = data_dir / "coverage.csv"
    if not coverage_path.exists():
        return pd.DataFrame()
    return pd.read_csv(coverage_path)


def get_stats(runs_df: pd.DataFrame, coverage_df: pd.DataFrame) -> dict:
    """Compute KPI statistics for the dashboard."""
    if runs_df.empty:
        return {
            "total_images": 0,
            "total_buckets": 0,
            "avg_map": 0.0,
            "coverage_gaps": 0,
            "verdict": "unknown",
        }
    latest_ts = runs_df["timestamp"].iloc[-1]
    latest = runs_df[runs_df["timestamp"] == latest_ts]
    n_buckets = len(runs_df.groupby(["weather", "scene", "time_of_day"]))
    avg_map = float(runs_df["mean_ap"].mean())
    verdict = str(latest["verdict"].iloc[0]) if "verdict" in latest.columns else "unknown"
    coverage_gaps = (
        int(len(coverage_df[coverage_df["status"].isin(["critical", "low"])]))
        if not coverage_df.empty
        else 0
    )
    return {
        "total_images": len(runs_df),
        "total_buckets": n_buckets,
        "avg_map": avg_map,
        "coverage_gaps": coverage_gaps,
        "verdict": verdict,
    }


def get_bucket_summary(latest_run: pd.DataFrame) -> pd.DataFrame:
    """Get a clean summary of the latest run's bucket results."""
    if latest_run.empty:
        return pd.DataFrame()
    cols = ["weather", "scene", "time_of_day", "mean_ap", "verdict"]
    available = [c for c in cols if c in latest_run.columns]
    return latest_run[available].copy()


def build_accuracy_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot runs into a timestamp x scenario matrix of mAP values."""
    if runs_df.empty:
        return runs_df
    runs_df = runs_df.copy()
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

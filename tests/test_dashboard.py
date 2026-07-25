from pathlib import Path

import pandas as pd

from src.dashboard import build_accuracy_table, build_coverage_heatmap, get_latest_run, get_runs
from src.regression_checker import init_db, insert_run


def _populate_db(db_path: Path):
    conn = init_db(db_path)
    insert_run(conn, "sha1", "clear", "highway", "day", 0.85, "pass")
    insert_run(conn, "sha1", "rainy", "city street", "night", 0.72, "warning")
    insert_run(conn, "sha2", "clear", "highway", "day", 0.80, "pass")
    conn.close()


def test_get_runs(tmp_path):
    db = tmp_path / "test.db"
    _populate_db(db)
    df = get_runs(db)
    assert len(df) == 3


def test_get_runs_missing_db(tmp_path):
    df = get_runs(tmp_path / "nonexistent.db")
    assert len(df) == 0


def test_get_latest_run(tmp_path):
    db = tmp_path / "test.db"
    _populate_db(db)
    latest = get_latest_run(db)
    assert len(latest) >= 1
    assert "commit_sha" in latest.columns


def test_build_accuracy_table(tmp_path):
    db = tmp_path / "test.db"
    _populate_db(db)
    runs = get_runs(db)
    table = build_accuracy_table(runs)
    assert not table.empty
    assert table.shape[0] >= 1


def test_build_coverage_heatmap(tmp_path):
    db = tmp_path / "test.db"
    _populate_db(db)
    runs = get_runs(db)
    heatmap = build_coverage_heatmap(runs)
    assert not heatmap.empty
    assert "highway" in heatmap.columns or "city street" in heatmap.columns


def test_build_accuracy_table_empty():
    df = build_accuracy_table(pd.DataFrame())
    assert len(df) == 0


def test_build_coverage_heatmap_empty():
    df = build_coverage_heatmap(pd.DataFrame())
    assert len(df) == 0

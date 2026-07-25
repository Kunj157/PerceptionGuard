from pathlib import Path

import pandas as pd
import yaml

from src.coverage_analyzer import compute_coverage, coverage_summary, load_thresholds


def _make_metadata(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def test_compute_coverage_basic():
    df = _make_metadata([
        {"weather": "clear", "scene": "highway", "time_of_day": "day"} for _ in range(20)
    ] + [
        {"weather": "rainy", "scene": "city street", "time_of_day": "night"} for _ in range(5)
    ])
    table = compute_coverage(df)
    assert len(table) == 2
    assert set(table.columns) == {"weather", "scene", "time_of_day", "count", "status"}

    clear_row = table[table["weather"] == "clear"].iloc[0]
    assert clear_row["status"] == "adequate"

    rainy_row = table[table["weather"] == "rainy"].iloc[0]
    assert rainy_row["status"] == "low"


def test_compute_coverage_critical_bucket():
    records = [{"weather": "foggy", "scene": "tunnel", "time_of_day": "dawn/dusk"}]
    records += [{"weather": "clear", "scene": "highway", "time_of_day": "day"} for _ in range(20)]
    df = _make_metadata(records)
    table = compute_coverage(df, thresholds={"critical": 5, "low": 10})
    foggy_row = table[table["weather"] == "foggy"].iloc[0]
    assert foggy_row["status"] == "critical"
    assert foggy_row["count"] == 1


def test_compute_coverage_empty_dataframe():
    df = pd.DataFrame(columns=["weather", "scene", "time_of_day"])
    table = compute_coverage(df)
    assert len(table) == 0


def test_compute_coverage_custom_thresholds():
    df = _make_metadata([
        {"weather": "clear", "scene": "highway", "time_of_day": "day"} for _ in range(50)
    ])
    table = compute_coverage(df, thresholds={"critical": 0, "low": 100})
    assert table.iloc[0]["status"] == "low"


def test_coverage_summary():
    df = _make_metadata([
        {"weather": "clear", "scene": "highway", "time_of_day": "day"} for _ in range(20)
    ] + [
        {"weather": "foggy", "scene": "tunnel", "time_of_day": "night"},
    ])
    table = compute_coverage(df)
    summary = coverage_summary(table)
    assert summary["adequate"] == 1
    assert summary["low"] == 1


def test_load_thresholds_default():
    thresholds = load_thresholds()
    assert thresholds == {"critical": 0, "low": 10}


def test_load_thresholds_from_yaml(tmp_path):
    cfg = {"coverage_thresholds": {"critical": 5, "low": 50}}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(cfg))
    thresholds = load_thresholds(config_file)
    assert thresholds == {"critical": 5, "low": 50}


def test_load_thresholds_missing_file():
    thresholds = load_thresholds(Path("/nonexistent/config.yaml"))
    assert thresholds == {"critical": 0, "low": 10}

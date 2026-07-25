import pandas as pd
import yaml

from src.regression_checker import (
    compare_against_baseline,
    get_baseline,
    init_db,
    insert_run,
    load_thresholds,
    overall_verdict,
)


def _make_results(ap_map: dict[tuple, float]) -> pd.DataFrame:
    records = [
        {"weather": w, "scene": s, "time_of_day": t, "mean_ap": ap}
        for (w, s, t), ap in ap_map.items()
    ]
    return pd.DataFrame(records)


def test_init_db_creates_table(tmp_path):
    conn = init_db(tmp_path / "test.db")
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    assert ("runs",) in tables


def test_insert_and_get_baseline(tmp_path):
    conn = init_db(tmp_path / "test.db")
    insert_run(conn, "abc123", "clear", "highway", "day", 0.85, "pass")
    baseline = get_baseline(conn)
    conn.close()
    assert baseline is not None
    assert len(baseline) == 1
    assert baseline.iloc[0]["mean_ap"] == 0.85


def test_get_baseline_empty_db(tmp_path):
    conn = init_db(tmp_path / "test.db")
    baseline = get_baseline(conn)
    conn.close()
    assert baseline is None


def test_compare_no_baseline():
    current = _make_results({("clear", "highway", "day"): 0.8})
    result = compare_against_baseline(current, None)
    assert (result["verdict"] == "pass").all()
    assert (result["delta_ap"] == 0.0).all()


def test_compare_pass():
    current = _make_results({("clear", "highway", "day"): 0.80})
    baseline = _make_results({("clear", "highway", "day"): 0.82})
    result = compare_against_baseline(current, baseline)
    assert result.iloc[0]["verdict"] == "pass"
    assert abs(result.iloc[0]["delta_ap"] - 0.02) < 1e-6


def test_compare_warning():
    current = _make_results({("clear", "highway", "day"): 0.75})
    baseline = _make_results({("clear", "highway", "day"): 0.82})
    result = compare_against_baseline(current, baseline)
    assert result.iloc[0]["verdict"] == "warning"


def test_compare_critical():
    current = _make_results({("clear", "highway", "day"): 0.65})
    baseline = _make_results({("clear", "highway", "day"): 0.82})
    result = compare_against_baseline(current, baseline)
    assert result.iloc[0]["verdict"] == "critical"


def test_compare_new_bucket_no_baseline():
    current = _make_results({("foggy", "tunnel", "night"): 0.70})
    baseline = _make_results({("clear", "highway", "day"): 0.80})
    result = compare_against_baseline(current, baseline)
    assert result.iloc[0]["verdict"] == "pass"


def test_overall_verdict_pass():
    df = pd.DataFrame([{"verdict": "pass"}, {"verdict": "pass"}])
    assert overall_verdict(df) == "pass"


def test_overall_verdict_warning():
    df = pd.DataFrame([{"verdict": "pass"}, {"verdict": "warning"}])
    assert overall_verdict(df) == "warning"


def test_overall_verdict_critical():
    df = pd.DataFrame([{"verdict": "warning"}, {"verdict": "critical"}])
    assert overall_verdict(df) == "critical"


def test_load_thresholds_default():
    t = load_thresholds()
    assert t["critical_drop"] == 0.10
    assert t["warning_drop"] == 0.05


def test_load_thresholds_from_yaml(tmp_path):
    cfg = {"regression_thresholds": {"critical_drop": 0.20, "warning_drop": 0.10}}
    f = tmp_path / "config.yaml"
    f.write_text(yaml.dump(cfg))
    t = load_thresholds(f)
    assert t["critical_drop"] == 0.20

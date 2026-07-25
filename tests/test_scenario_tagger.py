import json
import sqlite3
from pathlib import Path

import pandas as pd

from src.scenario_tagger import build_metadata, parse_label_file, save_metadata


def _write_label(path: Path, weather="clear", scene="city street", time_of_day="day"):
    data = {
        "name": path.stem + ".jpg",
        "attributes": {
            "weather": weather,
            "scene": scene,
            "time of day": time_of_day,
        },
    }
    path.write_text(json.dumps(data))


def test_parse_label_file(tmp_path):
    label = tmp_path / "0000.json"
    _write_label(label, weather="rainy", scene="highway", time_of_day="night")
    result = parse_label_file(label)
    assert result["weather"] == "rainy"
    assert result["scene"] == "highway"
    assert result["time_of_day"] == "night"


def test_parse_label_file_missing_attributes(tmp_path):
    label = tmp_path / "0001.json"
    label.write_text(json.dumps({"name": "0001.jpg", "attributes": {}}))
    result = parse_label_file(label)
    assert result["weather"] == "unknown"
    assert result["scene"] == "unknown"
    assert result["time_of_day"] == "unknown"


def test_build_metadata(tmp_path):
    _write_label(tmp_path / "a.json", weather="foggy", scene="residential")
    _write_label(tmp_path / "b.json", weather="clear", time_of_day="night")
    df = build_metadata(tmp_path)
    assert len(df) == 2
    assert set(df.columns) == {"name", "weather", "scene", "time_of_day"}


def test_build_metadata_invalid_values(tmp_path):
    label = tmp_path / "bad.json"
    _write_label(label, weather="tornado", scene="space", time_of_day="midnight")
    df = build_metadata(tmp_path)
    assert df.iloc[0]["weather"] == "unknown"
    assert df.iloc[0]["scene"] == "unknown"
    assert df.iloc[0]["time_of_day"] == "unknown"


def test_build_metadata_empty_directory(tmp_path):
    df = build_metadata(tmp_path)
    assert len(df) == 0
    assert list(df.columns) == ["name", "weather", "scene", "time_of_day"]


def test_save_metadata_csv(tmp_path):
    df = pd.DataFrame(
        [{"name": "a.jpg", "weather": "clear", "scene": "highway", "time_of_day": "day"}]
    )
    out = tmp_path / "meta.csv"
    save_metadata(df, out)
    loaded = pd.read_csv(out)
    assert len(loaded) == 1
    assert loaded.iloc[0]["weather"] == "clear"


def test_save_metadata_sqlite(tmp_path):
    df = pd.DataFrame(
        [{"name": "a.jpg", "weather": "rainy", "scene": "city street", "time_of_day": "night"}]
    )
    out = tmp_path / "meta.db"
    save_metadata(df, out)
    conn = sqlite3.connect(str(out))
    rows = conn.execute("SELECT * FROM metadata").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][1] == "rainy"


def test_save_metadata_unsupported_format(tmp_path):
    df = pd.DataFrame()
    try:
        save_metadata(df, tmp_path / "meta.txt")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

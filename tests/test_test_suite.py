import json
from pathlib import Path

import pandas as pd

from src.test_suite import build_manifest, curate_test_suite, save_manifest, validate_manifest


def _make_metadata(n_per_bucket: int = 5) -> pd.DataFrame:
    records = []
    for weather in ["clear", "rainy"]:
        for scene in ["highway", "city street"]:
            for time_of_day in ["day", "night"]:
                for i in range(n_per_bucket):
                    records.append({
                        "name": f"{weather}_{scene}_{time_of_day}_{i}.jpg",
                        "weather": weather,
                        "scene": scene,
                        "time_of_day": time_of_day,
                    })
    return pd.DataFrame(records)


def test_curate_test_suite_balances_buckets(tmp_path):
    df = _make_metadata(n_per_bucket=20)
    for name in df["name"]:
        (tmp_path / name).touch()
    suite = curate_test_suite(df, tmp_path, target_per_bucket=5)
    counts = suite.groupby(["weather", "scene", "time_of_day"]).size()
    assert counts.max() == 5
    assert counts.min() == 5


def test_curate_test_suite_includes_small_buckets():
    df = pd.DataFrame([
        {"name": "a.jpg", "weather": "foggy", "scene": "tunnel", "time_of_day": "night"},
        {"name": "b.jpg", "weather": "clear", "scene": "highway", "time_of_day": "day"},
    ] + [
        {"name": f"c_{i}.jpg", "weather": "clear", "scene": "highway", "time_of_day": "day"}
        for i in range(30)
    ])
    suite = curate_test_suite(df, Path("."), target_per_bucket=5)
    foggy_count = len(suite[suite["weather"] == "foggy"])
    assert foggy_count == 1


def test_curate_test_suite_adds_image_path(tmp_path):
    df = _make_metadata(n_per_bucket=2)
    for name in df["name"]:
        (tmp_path / name).touch()
    suite = curate_test_suite(df, tmp_path, target_per_bucket=2)
    assert all(suite["image_path"].str.startswith(str(tmp_path)))


def test_build_manifest(tmp_path):
    df = _make_metadata(n_per_bucket=2)
    for name in df["name"]:
        (tmp_path / name).touch()
    suite = curate_test_suite(df, tmp_path, target_per_bucket=2)
    manifest = build_manifest(suite)
    assert manifest["version"] == "v1"
    assert manifest["total_images"] == len(suite)
    assert len(manifest["images"]) == len(suite)
    assert len(manifest["buckets"]) > 0


def test_save_and_validate_manifest(tmp_path):
    df = _make_metadata(n_per_bucket=2)
    for name in df["name"]:
        (tmp_path / name).touch()
    suite = curate_test_suite(df, tmp_path, target_per_bucket=2)
    manifest = build_manifest(suite)
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)

    loaded = json.loads(manifest_path.read_text())
    errors = validate_manifest(loaded, tmp_path)
    assert errors == []


def test_validate_manifest_missing_files(tmp_path):
    img_entry = {
        "name": "nonexistent.jpg",
        "weather": "clear",
        "scene": "highway",
        "time_of_day": "day",
    }
    manifest = {
        "version": "v1",
        "total_images": 1,
        "buckets": [],
        "images": [img_entry],
    }
    errors = validate_manifest(manifest, tmp_path)
    assert len(errors) == 1
    assert "Missing file" in errors[0]


def test_validate_manifest_duplicates(tmp_path):
    img = tmp_path / "dup.jpg"
    img.touch()
    manifest = {
        "version": "v1",
        "total_images": 2,
        "buckets": [],
        "images": [
            {"name": "dup.jpg", "weather": "clear", "scene": "highway", "time_of_day": "day"},
            {"name": "dup.jpg", "weather": "clear", "scene": "highway", "time_of_day": "day"},
        ],
    }
    errors = validate_manifest(manifest, tmp_path)
    assert any("Duplicate" in e for e in errors)


def test_validate_manifest_count_mismatch(tmp_path):
    img = tmp_path / "a.jpg"
    img.touch()
    manifest = {
        "version": "v1",
        "total_images": 5,
        "buckets": [],
        "images": [{"name": "a.jpg", "weather": "clear", "scene": "highway", "time_of_day": "day"}],
    }
    errors = validate_manifest(manifest, tmp_path)
    assert any("total_images" in e for e in errors)

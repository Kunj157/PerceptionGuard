from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

VALID_WEATHER = {
    "rainy",
    "cloudy",
    "overcast",
    "foggy",
    "partly cloudy",
    "clear",
    "unknown",
}

VALID_SCENE = {
    "tunnel",
    "residential",
    "parking lot",
    "city street",
    "gas stations",
    "highway",
    "unknown",
}

VALID_TIME_OF_DAY = {
    "dawn/dusk",
    "day",
    "night",
    "unknown",
}

REQUIRED_ATTRIBUTES = ["weather", "scene", "time of day"]


def parse_label_file(path: Path) -> dict:
    """Parse a single BDD100K JSON label file and return its attributes."""
    with open(path) as f:
        data = json.load(f)

    attrs = data.get("attributes", {})
    return {
        "name": data.get("name", path.stem),
        "weather": attrs.get("weather", "unknown"),
        "scene": attrs.get("scene", "unknown"),
        "time_of_day": attrs.get("time of day", "unknown"),
    }


def build_metadata(label_dir: Path) -> pd.DataFrame:
    """Parse all label files in a directory and return a metadata DataFrame."""
    label_files = sorted(label_dir.glob("*.json"))
    if not label_files:
        return pd.DataFrame(columns=["name", "weather", "scene", "time_of_day"])

    records = [parse_label_file(f) for f in label_files]
    df = pd.DataFrame(records)

    df["weather"] = df["weather"].where(df["weather"].isin(VALID_WEATHER), "unknown")
    df["scene"] = df["scene"].where(df["scene"].isin(VALID_SCENE), "unknown")
    df["time_of_day"] = df["time_of_day"].where(
        df["time_of_day"].isin(VALID_TIME_OF_DAY), "unknown"
    )

    return df


def save_metadata(df: pd.DataFrame, output_path: Path) -> None:
    """Save metadata DataFrame to CSV or SQLite based on file extension."""
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix == ".db":
        import sqlite3

        conn = sqlite3.connect(str(output_path))
        df.to_sql("metadata", conn, if_exists="replace", index=False)
        conn.close()
    else:
        raise ValueError(f"Unsupported output format: {suffix}")

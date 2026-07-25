"""Download and prepare BDD100K dataset for PerceptionGuard pipeline.

BDD100K requires registration at https://bdd-data.berkeley.edu/
After downloading, place files in data/raw/ and run this script.

Usage:
    python scripts/download_data.py --setup
    python scripts/download_data.py --prepare
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

TRAIN_SIZE = 500
VAL_SIZE = 100


def check_prerequisites() -> bool:
    """Check if raw BDD100K data exists."""
    if not RAW_DIR.exists():
        return False
    has_images = any((RAW_DIR / "images").glob("*.jpg")) if (RAW_DIR / "images").exists() else False
    has_labels = any((RAW_DIR / "labels").glob("*.json")) if (RAW_DIR / "labels").exists() else False
    return has_images and has_labels


def print_setup_instructions():
    """Print manual download instructions."""
    print("=" * 60)
    print("BDD100K Download Instructions")
    print("=" * 60)
    print()
    print("1. Go to https://bdd-data.berkeley.edu/")
    print("2. Register and download the following:")
    print("   - 100K images (train/val split)")
    print("   - Detection labels (100K)")
    print()
    print("3. Extract into the following structure:")
    print(f"   {RAW_DIR}/")
    print("   ├── images/")
    print("   │   ├── train/")
    print("   │   └── val/")
    print("   └── labels/")
    print("       ├── train/")
    print("       └── val/")
    print()
    print("4. Run: python scripts/download_data.py --prepare")
    print("=" * 60)


def prepare_data(max_images: int | None = None) -> None:
    """Parse raw BDD100K labels into metadata CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_labels = RAW_DIR / "labels"
    if not raw_labels.exists():
        print(f"Error: {raw_labels} not found. Run --setup first.")
        sys.exit(1)

    all_records = []
    label_files = sorted(raw_labels.rglob("*.json"))

    if not label_files:
        print(f"Error: No JSON label files found in {raw_labels}")
        sys.exit(1)

    print(f"Found {len(label_files)} label files")

    for label_file in label_files:
        if max_images and len(all_records) >= max_images:
            break

        try:
            with open(label_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        attrs = data.get("attributes", {})
        record = {
            "name": data.get("name", label_file.stem),
            "weather": attrs.get("weather", "unknown"),
            "scene": attrs.get("scene", "unknown"),
            "time_of_day": attrs.get("time of day", "unknown"),
            "source_split": label_file.parent.name,
        }
        all_records.append(record)

    import pandas as pd

    df = pd.DataFrame(all_records)
    output_path = DATA_DIR / "metadata.csv"
    df.to_csv(output_path, index=False)

    print(f"\nPrepared {len(df)} records -> {output_path}")
    print(f"\nScenario distribution:")
    for col in ["weather", "scene", "time_of_day"]:
        print(f"\n  {col}:")
        for val, count in df[col].value_counts().items():
            print(f"    {val}: {count}")


def main():
    parser = argparse.ArgumentParser(description="BDD100K data download and preparation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--setup", action="store_true", help="Show download instructions")
    group.add_argument("--prepare", action="store_true", help="Parse labels into metadata CSV")
    group.add_argument("--check", action="store_true", help="Check if data is available")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process")
    args = parser.parse_args()

    if args.setup:
        print_setup_instructions()
    elif args.check:
        if check_prerequisites():
            print("BDD100K data found.")
        else:
            print("BDD100K data not found. Run --setup for instructions.")
    elif args.prepare:
        prepare_data(max_images=args.max_images)


if __name__ == "__main__":
    main()

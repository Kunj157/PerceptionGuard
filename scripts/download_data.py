"""Download and prepare BDD100K dataset for PerceptionGuard pipeline.

Downloads images and detection labels from the official ETH Zurich mirror,
then parses attributes into metadata CSV.

Usage:
    python scripts/download_data.py                 # download + prepare
    python scripts/download_data.py --prepare-only  # skip download, just parse
    python scripts/download_data.py --check         # check if data exists
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"

BASE_URL = "https://dl.cv.ethz.ch/bdd100k/data"

DOWNLOADS = {
    "100k_images_train.zip": f"{BASE_URL}/100k_images_train.zip",
    "100k_images_val.zip": f"{BASE_URL}/100k_images_val.zip",
    "bdd100k_det_20_labels_trainval.zip": f"{BASE_URL}/bdd100k_det_20_labels_trainval.zip",
}

# Use a small subset for local dev — keep full numbers for production
MAX_IMAGES_DEFAULT = 500


def download_file(url: str, dest: Path) -> None:
    """Download a file using wget with progress."""
    print(f"  Downloading {dest.name}...")
    result = subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(dest), url],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  Error downloading {dest.name}. Check your network.")
        sys.exit(1)


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file."""
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def download_all() -> None:
    """Download all required BDD100K files."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in DOWNLOADS.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"  {filename} already exists, skipping.")
            continue
        download_file(url, dest)

    print("\nExtracting...")
    for filename in DOWNLOADS:
        zip_path = RAW_DIR / filename
        if zip_path.exists():
            extract_zip(zip_path, RAW_DIR)

    print("Download complete.")


def check_data() -> bool:
    """Check if BDD100K data is available."""
    train_images = RAW_DIR / "bdd100k" / "images" / "100k" / "train"
    val_images = RAW_DIR / "bdd100k" / "images" / "100k" / "val"
    labels_dir = RAW_DIR / "bdd100k" / "labels"

    if not train_images.exists():
        print(f"Missing: {train_images}")
        return False
    if not val_images.exists():
        print(f"Missing: {val_images}")
        return False
    if not labels_dir.exists():
        print(f"Missing: {labels_dir}")
        return False

    n_train = len(list(train_images.glob("*.jpg")))
    n_val = len(list(val_images.glob("*.jpg")))
    print(f"Found {n_train} train images, {n_val} val images")
    return True


def prepare_metadata(max_images: int = MAX_IMAGES_DEFAULT) -> None:
    """Parse BDD100K detection labels into metadata CSV."""
    import pandas as pd

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    det_labels = RAW_DIR / "bdd100k" / "labels" / "det_20"
    if not det_labels.exists():
        # Try alternate location from older format
        alt_labels = RAW_DIR / "bdd100k" / "labels"
        if alt_labels.exists():
            label_files = list(alt_labels.glob("*.json"))
        else:
            print(f"Error: labels not found at {det_labels}")
            sys.exit(1)
    else:
        label_files = sorted(det_labels.glob("*.json"))

    if not label_files:
        print(f"Error: no label JSON files found")
        sys.exit(1)

    print(f"Found {len(label_files)} label files")

    all_records = []
    for lf in label_files:
        print(f"  Parsing {lf.name}...")
        with open(lf) as f:
            data = json.load(f)

        for item in data:
            if max_images and len(all_records) >= max_images:
                break

            attrs = item.get("attributes", {})
            name = item.get("name", "")
            if not name:
                continue

            all_records.append({
                "name": name,
                "weather": attrs.get("weather", "unknown"),
                "scene": attrs.get("scene", "unknown"),
                "time_of_day": attrs.get("time of day", "unknown"),
            })

        if max_images and len(all_records) >= max_images:
            break

    df = pd.DataFrame(all_records)
    output_path = DATA_DIR / "metadata.csv"
    df.to_csv(output_path, index=False)

    print(f"\nWrote {len(df)} records -> {output_path}")
    print("\nDistribution:")
    for col in ["weather", "scene", "time_of_day"]:
        print(f"\n  {col}:")
        for val, count in df[col].value_counts().items():
            print(f"    {val}: {count}")


def main():
    parser = argparse.ArgumentParser(description="BDD100K download and preparation")
    parser.add_argument("--prepare-only", action="store_true", help="Skip download, just parse labels")
    parser.add_argument("--check", action="store_true", help="Check if data is available")
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES_DEFAULT, help="Max images to process")
    args = parser.parse_args()

    if args.check:
        if check_data():
            print("BDD100K data is ready.")
        else:
            print("BDD100K data not found. Run without --check to download.")
        return

    if not args.prepare_only:
        print("Downloading BDD100K...")
        download_all()

    print("\nPreparing metadata...")
    prepare_metadata(max_images=args.max_images)


if __name__ == "__main__":
    main()

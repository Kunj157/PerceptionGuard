from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def curate_test_suite(
    df: pd.DataFrame,
    images_dir: Path,
    target_per_bucket: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample a balanced test suite from metadata, ensuring all buckets are represented.

    Low-coverage buckets get all their samples included. Adequate buckets are
    downsampled to target_per_bucket.

    Args:
        df: Metadata DataFrame with name, weather, scene, time_of_day columns.
        images_dir: Directory containing the actual image files.
        target_per_bucket: Desired sample count per unique scenario combination.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame of selected samples with an added 'image_path' column.
    """
    group_cols = ["weather", "scene", "time_of_day"]
    selected = []

    for _, group in df.groupby(group_cols, dropna=False):
        if len(group) <= target_per_bucket:
            selected.append(group)
        else:
            selected.append(group.sample(n=target_per_bucket, random_state=seed))

    result = pd.concat(selected, ignore_index=True)
    result["image_path"] = result["name"].apply(lambda n: str(images_dir / n))
    return result


def build_manifest(suite_df: pd.DataFrame) -> dict:
    """Convert a test suite DataFrame into a JSON-serializable manifest."""
    return {
        "version": "v1",
        "total_images": len(suite_df),
        "buckets": suite_df.groupby(["weather", "scene", "time_of_day"], dropna=False)
        .size()
        .reset_index(name="count")
        .to_dict(orient="records"),
        "images": suite_df[["name", "weather", "scene", "time_of_day", "image_path"]].to_dict(
            orient="records"
        ),
    }


def save_manifest(manifest: dict, output_path: Path) -> None:
    """Write manifest dict to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def validate_manifest(manifest: dict, images_dir: Path) -> list[str]:
    """Check manifest integrity. Returns list of error strings (empty = valid)."""
    errors = []
    seen_names = set()

    for img in manifest.get("images", []):
        name = img.get("name")
        if not name:
            errors.append("Entry missing 'name' field")
            continue
        if name in seen_names:
            errors.append(f"Duplicate entry: {name}")
        seen_names.add(name)

        img_path = images_dir / name
        if not img_path.exists():
            errors.append(f"Missing file: {img_path}")

    if manifest.get("total_images") != len(manifest.get("images", [])):
        errors.append(
            f"total_images ({manifest.get('total_images')}) != "
            f"actual count ({len(manifest.get('images', []))})"
        )

    return errors

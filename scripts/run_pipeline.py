"""End-to-end pipeline runner with real BDD100K data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from ultralytics import YOLO

from src.coverage_analyzer import compute_coverage, coverage_summary, save_coverage_table
from src.eval_harness import aggregate_bucket_metrics, compute_map_per_bucket
from src.regression_checker import (
    compare_against_baseline,
    get_baseline,
    init_db,
    insert_run,
    overall_verdict,
)
from src.test_suite import build_manifest, curate_test_suite, save_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
METADATA_CSV = DATA_DIR / "metadata.csv"
IMAGES_DIR = DATA_DIR / "bdd100k_raw" / "data"
SAMPLES_JSON = DATA_DIR / "bdd100k_raw" / "samples.json"
DB_PATH = DATA_DIR / "metrics.db"
COVERAGE_CSV = DATA_DIR / "coverage.csv"
MANIFEST_JSON = DATA_DIR / "test_suite.json"

BDD_CLASS_MAP = {
    "bike": 0, "bus": 1, "car": 2, "motor": 3, "person": 4,
    "rider": 5, "traffic light": 6, "traffic sign": 7, "train": 8, "truck": 9,
}


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_CSV)
    df = df.rename(columns={"filename": "name", "timeofday": "time_of_day"})
    df["weather"] = df["weather"].replace({"undefined": "unknown"})
    df["time_of_day"] = df["time_of_day"].replace({"daytime": "day", "undefined": "unknown"})
    df["scene"] = df["scene"].replace({"undefined": "unknown"})
    df["image_path"] = df["filepath"]
    return df


def load_ground_truth_map() -> dict[str, list[dict]]:
    with open(SAMPLES_JSON) as f:
        data = json.load(f)
    gt_map = {}
    for s in data["samples"]:
        fname = Path(s["filepath"]).name
        dets = []
        for d in s["detections"]["detections"]:
            label = d["label"]
            if label not in BDD_CLASS_MAP:
                continue
            x, y, w, h = d["bounding_box"]
            img_w = s["metadata"]["width"]
            img_h = s["metadata"]["height"]
            x1 = x * img_w
            y1 = y * img_h
            x2 = (x + w) * img_w
            y2 = (y + h) * img_h
            dets.append({"class_id": BDD_CLASS_MAP[label], "bbox": [x1, y1, x2, y2]})
        gt_map[fname] = dets
    return gt_map


def run_inference(model: YOLO, image_paths: list[str]) -> list[dict]:
    results = model.predict(source=image_paths, imgsz=640, conf=0.25, verbose=False)
    preds = []
    for img_path, result in zip(image_paths, results):
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({"class_id": cls_id, "confidence": conf, "bbox": xyxy})
        preds.append({
            "name": Path(img_path).name,
            "image_path": img_path,
            "detections": detections,
        })
    return preds


def build_ground_truth(metadata_df: pd.DataFrame, gt_map: dict) -> list[dict]:
    gt = []
    for _, row in metadata_df.iterrows():
        name = row["name"]
        gt.append({"name": name, "detections": gt_map.get(name, [])})
    return gt


def main():
    print("=== PerceptionGuard End-to-End Pipeline ===\n")

    metadata = load_metadata()
    print(f"Loaded {len(metadata)} images with metadata")

    coverage = compute_coverage(metadata)
    save_coverage_table(coverage, COVERAGE_CSV)
    summary = coverage_summary(coverage)
    print(f"Coverage: {summary}")

    suite = curate_test_suite(metadata, IMAGES_DIR, target_per_bucket=5)
    manifest = build_manifest(suite)
    save_manifest(manifest, MANIFEST_JSON)
    print(f"Test suite: {len(suite)} images across {len(manifest['buckets'])} buckets")

    gt_map = load_ground_truth_map()
    print(f"Loaded ground truth for {len(gt_map)} images")

    print("\nLoading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print(f"Running inference on {len(suite)} images...")
    image_paths = suite["image_path"].tolist()
    predictions = run_inference(model, image_paths)

    gt = build_ground_truth(suite, gt_map)
    total_gt_boxes = sum(len(g["detections"]) for g in gt)
    print(f"Ground truth: {total_gt_boxes} total boxes across {len(gt)} images")

    image_metrics = compute_map_per_bucket(predictions, gt)
    bucket_metrics = aggregate_bucket_metrics(image_metrics, metadata)
    print("\nPer-bucket mAP:")
    print(bucket_metrics.to_string(index=False))

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(DB_PATH)
    baseline = get_baseline(conn)

    bucket_metrics_renamed = bucket_metrics.rename(columns={"mean_ap": "mean_ap"})
    if "weather" not in bucket_metrics_renamed.columns:
        print("Warning: no bucket metrics to store")
        conn.close()
        return

    results = compare_against_baseline(bucket_metrics_renamed, baseline)
    verdict = overall_verdict(results)
    print(f"\nOverall verdict: {verdict}")

    for _, row in results.iterrows():
        insert_run(
            conn,
            commit_sha="local-run",
            weather=row["weather"],
            scene=row["scene"],
            time_of_day=row["time_of_day"],
            mean_ap=row["mean_ap"],
            verdict=row["verdict"],
        )
    conn.close()

    print(f"\nResults stored in {DB_PATH}")
    print("Run `streamlit run src/app.py` to view the dashboard")


if __name__ == "__main__":
    main()

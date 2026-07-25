from __future__ import annotations

from pathlib import Path

import pandas as pd


def compute_map_per_bucket(
    predictions: list[dict],
    ground_truth: list[dict],
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute average precision per scenario bucket.

    Args:
        predictions: List of dicts with keys: name, image_path, detections.
                     Each detection: {class_id, confidence, bbox}.
        ground_truth: List of dicts with keys: name, detections.
                      Each detection: {class_id, bbox}.
        iou_threshold: IoU threshold for a true positive match.

    Returns:
        DataFrame with columns: name, bucket_key, ap.
    """
    results = []
    gt_by_name = {g["name"]: g["detections"] for g in ground_truth}

    for pred in predictions:
        name = pred["name"]
        preds = pred.get("detections", [])
        gts = gt_by_name.get(name, [])

        ap = _compute_ap(preds, gts, iou_threshold)
        results.append({"name": name, "ap": ap})

    return pd.DataFrame(results)


def _compute_ap(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float,
) -> float:
    """Compute average precision for a single image."""
    if not ground_truths:
        return 1.0 if not predictions else 0.0
    if not predictions:
        return 0.0

    sorted_preds = sorted(predictions, key=lambda d: d["confidence"], reverse=True)
    matched_gt = set()
    tp_count = 0

    for pred in sorted_preds:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(ground_truths):
            if idx in matched_gt:
                continue
            if pred["class_id"] != gt["class_id"]:
                continue
            iou = _compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx >= 0:
            tp_count += 1
            matched_gt.add(best_idx)

    precision = tp_count / len(sorted_preds) if sorted_preds else 0.0
    recall = tp_count / len(ground_truths) if ground_truths else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def aggregate_bucket_metrics(
    image_metrics: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Merge image-level AP with metadata and compute per-bucket mean AP."""
    merged = image_metrics.merge(metadata, on="name", how="inner")
    bucket_cols = ["weather", "scene", "time_of_day"]
    bucket_means = (
        merged.groupby(bucket_cols, dropna=False)["ap"]
        .mean()
        .reset_index(name="mean_ap")
    )
    return bucket_means

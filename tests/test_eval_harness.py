import pandas as pd

from src.eval_harness import (
    _compute_ap,
    _compute_iou,
    aggregate_bucket_metrics,
    compute_map_per_bucket,
)


def test_compute_iou_perfect_overlap():
    assert _compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_compute_iou_no_overlap():
    assert _compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_compute_iou_partial_overlap():
    # intersection=25, union=175, IoU=1/7
    iou = _compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert abs(iou - 1 / 7) < 1e-6


def test_compute_ap_perfect_predictions():
    preds = [{"class_id": 0, "confidence": 0.9, "bbox": [0, 0, 10, 10]}]
    gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
    assert _compute_ap(preds, gts, 0.5) == 1.0


def test_compute_ap_no_predictions():
    gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
    assert _compute_ap([], gts, 0.5) == 0.0


def test_compute_ap_no_ground_truth():
    preds = [{"class_id": 0, "confidence": 0.9, "bbox": [0, 0, 10, 10]}]
    assert _compute_ap(preds, [], 0.5) == 0.0


def test_compute_ap_mismatched_class():
    preds = [{"class_id": 1, "confidence": 0.9, "bbox": [0, 0, 10, 10]}]
    gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
    assert _compute_ap(preds, gts, 0.5) == 0.0


def test_compute_map_per_bucket():
    predictions = [
        {"name": "a.jpg", "detections": [
            {"class_id": 0, "confidence": 0.9, "bbox": [0, 0, 10, 10]},
        ]},
    ]
    ground_truth = [
        {"name": "a.jpg", "detections": [
            {"class_id": 0, "bbox": [0, 0, 10, 10]},
        ]},
    ]
    result = compute_map_per_bucket(predictions, ground_truth)
    assert len(result) == 1
    assert result.iloc[0]["ap"] == 1.0


def test_aggregate_bucket_metrics():
    image_metrics = pd.DataFrame([
        {"name": "a.jpg", "ap": 1.0},
        {"name": "b.jpg", "ap": 0.5},
    ])
    metadata = pd.DataFrame([
        {"name": "a.jpg", "weather": "clear", "scene": "highway", "time_of_day": "day"},
        {"name": "b.jpg", "weather": "rainy", "scene": "city street", "time_of_day": "night"},
    ])
    result = aggregate_bucket_metrics(image_metrics, metadata)
    assert len(result) == 2
    assert "mean_ap" in result.columns

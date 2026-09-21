"""Detection decode/filter (offline, synthetic predictions).

Regression suite for the COCO path: in 0.1.0 the detection branch returned scores for
all 8,400 candidate rows while the boxes had already been filtered, which crashed NMS
(index out of bounds) the first time a COCO model ran. The invariant tested here is
simple and non-negotiable: scores and boxes must always have the same length.
"""

import numpy as np

from streamcount.detector import COCO80, select_detections


def _coco_row(x, y, w, h, cls_index, conf, num_classes=80):
    row = np.zeros(4 + num_classes, dtype=np.float32)
    row[:4] = (x, y, w, h)
    row[4 + cls_index] = conf
    return row


def test_pose_decode_keeps_scores_aligned():
    pred = np.zeros((3, 56), dtype=np.float32)
    pred[0, :5] = [320, 320, 40, 80, 0.9]
    pred[1, :5] = [100, 100, 30, 60, 0.4]
    pred[2, :5] = [10, 10, 10, 10, 0.1]
    xywh, scores, labels = select_detections(pred, 0.25, {0}, ["person"], is_pose=True)
    assert len(xywh) == len(scores) == len(labels) == 2
    assert labels == ["person", "person"]


def test_coco_decode_scores_match_boxes():
    # two cars + one person + one below-threshold car
    pred = np.stack([
        _coco_row(300, 300, 100, 50, cls_index=2, conf=0.80),   # car
        _coco_row(500, 300, 120, 60, cls_index=2, conf=0.55),   # car
        _coco_row(700, 200, 40, 90, cls_index=0, conf=0.90),    # person
        _coco_row(100, 300, 90, 45, cls_index=2, conf=0.10),    # car, below conf
    ])
    xywh, scores, labels = select_detections(pred, 0.25, {2, 3, 5, 7}, COCO80, is_pose=False)
    assert len(xywh) == len(scores) == len(labels) == 2
    assert labels == ["car", "car"]
    assert list(np.round(scores, 2)) == [0.80, 0.55]


def test_coco_decode_respects_class_filter():
    pred = np.stack([
        _coco_row(300, 300, 100, 50, cls_index=2, conf=0.80),   # car
        _coco_row(700, 200, 40, 90, cls_index=0, conf=0.90),    # person
    ])
    _, _, only_people = select_detections(pred, 0.25, {0}, COCO80, is_pose=False)
    assert only_people == ["person"]
    _, _, only_cars = select_detections(pred, 0.25, {2}, COCO80, is_pose=False)
    assert only_cars == ["car"]


def test_coco_decode_empty_when_nothing_passes():
    pred = np.stack([_coco_row(300, 300, 100, 50, cls_index=2, conf=0.05)])
    xywh, scores, labels = select_detections(pred, 0.25, {2}, COCO80, is_pose=False)
    assert len(xywh) == len(scores) == len(labels) == 0

# Copyright 2025-present Kensho Technologies, LLC.
"""Tests for the bounding box utilities."""

import torch

from potatr.geometry import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, iou, rescale_bboxes


def test_iou():
    # Identical boxes
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    # No overlap
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    # Half of each box overlaps the other: intersection 50, union 150
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-6


def test_box_conversions_are_inverses():
    cxcywh = torch.tensor([[50.0, 40.0, 20.0, 10.0]])
    xyxy = box_cxcywh_to_xyxy(cxcywh)
    assert xyxy.tolist() == [[40.0, 35.0, 60.0, 45.0]]
    assert box_xyxy_to_cxcywh(xyxy).tolist() == cxcywh.tolist()


def test_rescale_bboxes_to_pixels():
    # A normalized (cx, cy, w, h) box on a page 1000 wide and 800 tall
    boxes = rescale_bboxes(torch.tensor([[0.5, 0.5, 0.2, 0.4]]), (1000, 800))
    assert torch.allclose(boxes, torch.tensor([[400.0, 240.0, 600.0, 560.0]]))

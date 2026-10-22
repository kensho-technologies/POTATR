# Copyright 2025-present Kensho Technologies, LLC.
"""Bounding box geometry utilities."""

from collections.abc import Sequence

import torch


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) to (x0, y0, x1, y1) format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x0, y0, x1, y1) to (cx, cy, w, h) format."""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def rescale_bboxes(out_bbox: torch.Tensor, size: Sequence[float]) -> torch.Tensor:
    """Rescale normalized bboxes (cx, cy, w, h) to pixel coordinates (x0, y0, x1, y1)."""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Compute IoU between two [x0, y0, x1, y1] bounding boxes."""
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection == 0:
        return 0.0

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    return intersection / union

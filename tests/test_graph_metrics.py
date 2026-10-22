# Copyright 2025-present Kensho Technologies, LLC.
"""Tests for the edge prediction metrics."""

import pytest
import torch

from potatr.graph_metrics import (
    compute_predicted_edges,
    compute_target_edges,
    evaluate_edge_prediction,
)
from potatr.postprocess import class_idx2name


PAGE_SIZE = (1000, 800)  # (width, height)
# A table and its caption, as normalized (cx, cy, w, h) boxes.
BOXES = torch.tensor([[0.3, 0.3, 0.4, 0.2], [0.3, 0.15, 0.4, 0.05]])
LABELS = [class_idx2name.index("table"), class_idx2name.index("table caption")]


def _target():
    return {
        "boxes": BOXES,
        "labels": torch.tensor(LABELS),
        "orig_size": torch.tensor([PAGE_SIZE[1], PAGE_SIZE[0]]),  # (height, width)
        "adjacencies": torch.tensor([[0, 1], [1, 0]]),
    }


def _predictions(edge_class):
    node_logits = torch.zeros(2, len(class_idx2name))
    node_logits[0, LABELS[0]] = 10.0
    node_logits[1, LABELS[1]] = 10.0
    edge_logits = torch.zeros(2, 2, 2)
    edge_logits[0, 1, edge_class] = 10.0
    edge_logits[1, 0, edge_class] = 10.0
    return node_logits, BOXES.clone(), edge_logits


def test_target_edges_link_both_directions():
    edges = compute_target_edges(_target(), class_idx2name)
    assert [(e[0]["label"], e[1]["label"]) for e in edges] == [
        ("table", "table caption"),
        ("table caption", "table"),
    ]
    assert edges[0][0]["bbox"] == pytest.approx([100.0, 160.0, 500.0, 320.0])  # pixels


def test_correct_edge_predictions_are_true_positives():
    target_edges = compute_target_edges(_target(), class_idx2name)
    predicted_edges = compute_predicted_edges(
        *_predictions(edge_class=1), PAGE_SIZE, class_idx2name, node_score_threshold=0.8
    )
    assert len(predicted_edges) == 2
    assert len(evaluate_edge_prediction(target_edges, predicted_edges, iou_threshold=0.8)) == 2


def test_no_edge_predicted():
    predicted_edges = compute_predicted_edges(
        *_predictions(edge_class=0), PAGE_SIZE, class_idx2name
    )
    assert predicted_edges == []

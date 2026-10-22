# Copyright 2025-present Kensho Technologies, LLC.

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from potatr.geometry import iou, rescale_bboxes


# An edge is a pair of nodes, each a dict with a "label" and a "bbox".
Edge = list[dict[str, Any]]


def compute_target_edges(target: dict[str, Any], class_idx2name: list[str]) -> list[Edge]:
    """Return a target's ground-truth edges, as pairs of {"label", "bbox"} nodes.

    There is one edge for each nonzero off-diagonal entry of the adjacency matrix, so both
    directions of a link are included. Boxes are in pixels of the original image.
    """
    target_bboxes = target["boxes"]
    img_size = target["orig_size"].cpu().numpy().tolist()[::-1]
    target_bboxes = [elem.tolist() for elem in rescale_bboxes(target_bboxes, img_size)]
    target_class_labels = [class_idx2name[elem] for elem in target["labels"].numpy().tolist()]
    target_adjacency_matrix = target["adjacencies"].numpy().tolist()
    target_edges = []
    for row_num in range(len(target_adjacency_matrix)):
        bbox1 = target_bboxes[row_num]
        class_label1 = target_class_labels[row_num]
        for col_num in range(len(target_adjacency_matrix[row_num])):
            if row_num == col_num:
                continue
            if target_adjacency_matrix[row_num][col_num] == 1:
                bbox2 = target_bboxes[col_num]
                class_label2 = target_class_labels[col_num]
                target_edges.append(
                    [{"label": class_label1, "bbox": bbox1}, {"label": class_label2, "bbox": bbox2}]
                )
    return target_edges


def compute_predicted_edges(
    pred_node_logits: torch.Tensor,
    pred_bboxes: torch.Tensor,
    pred_edge_logits: torch.Tensor,
    img_size: Sequence[int],
    class_idx2name: list[str],
    node_score_threshold: float = 0.5,
    edge_score_threshold: float = 0.5,
) -> list[Edge]:
    """Return one image's predicted edges, in the same format as compute_target_edges.

    The nodes are the queries whose most likely class isn't "no object" and has probability at
    least node_score_threshold. There is an edge from node i to node j when the highest
    probability of an edge class (any class but 0, no edge) is at least edge_score_threshold.
    Boxes are converted to pixels using img_size, (width, height).
    """
    m = pred_node_logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.numpy())
    pred_labels = [class_idx2name[label_idx] for label_idx in pred_labels]

    bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    pred_adj_matrix = pred_edge_logits.squeeze().softmax(-1)[:, :, 1:].max(-1).values.numpy()

    pred_scores = list(m.values.numpy())
    object_indices = [
        idx
        for idx, score in enumerate(pred_scores)
        if (score >= node_score_threshold and not pred_labels[idx] == "no object")
    ]

    predicted_edges = []
    for object1_idx in object_indices:
        bbox1 = bboxes[object1_idx]
        class_label1 = pred_labels[object1_idx]
        for object2_idx in object_indices:
            if object1_idx == object2_idx:
                continue
            if pred_adj_matrix[object1_idx, object2_idx] >= edge_score_threshold:
                bbox2 = bboxes[object2_idx]
                class_label2 = pred_labels[object2_idx]
                predicted_edges.append(
                    [{"label": class_label1, "bbox": bbox1}, {"label": class_label2, "bbox": bbox2}]
                )
    return predicted_edges


def score(edge1: Edge, edge2: Edge, iou_threshold: float) -> int:
    """Return 1 if two edges match, otherwise 0.

    Edges match if both ends have the same labels and boxes with an IoU of at least iou_threshold.
    """
    if not edge1[0]["label"] == edge2[0]["label"] or not edge1[1]["label"] == edge2[1]["label"]:
        return 0

    node1_iou = iou(edge1[0]["bbox"], edge2[0]["bbox"])
    node2_iou = iou(edge1[1]["bbox"], edge2[1]["bbox"])

    if node1_iou >= iou_threshold and node2_iou >= iou_threshold:
        return 1

    return 0


def compute_score_matrix(
    target_edges: list[Edge], predicted_edges: list[Edge], iou_threshold: float
) -> np.ndarray:
    """Return the score() of every target edge (rows) against every predicted edge (columns)."""
    score_matrix = np.zeros((len(target_edges), len(predicted_edges)))
    for t_idx, target_edge in enumerate(target_edges):
        for p_idx, predicted_edge in enumerate(predicted_edges):
            s = score(target_edge, predicted_edge, iou_threshold)
            score_matrix[t_idx, p_idx] = s
    return score_matrix


def evaluate_edge_prediction(
    target_edges: list[Edge], predicted_edges: list[Edge], iou_threshold: float
) -> list[Edge]:
    """Return the target edges matched by a predicted edge, at one IoU threshold.

    Target and predicted edges are matched one-to-one (Hungarian matching) to maximize the number
    of matches. The matched target edges are the true positives of the edge F1.
    """
    score_matrix = compute_score_matrix(target_edges, predicted_edges, iou_threshold)
    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)
    true_positives = [target_edges[r] for r, c in zip(row_ind, col_ind) if score_matrix[r, c] > 0]

    return true_positives

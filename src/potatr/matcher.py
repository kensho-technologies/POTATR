# Copyright 2025-present Kensho Technologies, LLC.
# Portions copyright (C) 2023 Microsoft Corporation, from LOBSTR
# (https://github.com/microsoft/LOBSTR), MIT License.
# Portions derived from DETR (https://github.com/facebookresearch/detr),
# Copyright (c) Facebook, Inc. and its affiliates, Apache License 2.0.
# Third-party portions have been modified. See THIRD_PARTY_NOTICES.md.
"""Modules to compute the matching cost and solve the corresponding LSAP."""

from typing import Any

from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from potatr.detr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """Compute an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
    ) -> None:
        """Create the matcher.

        Params:
            cost_class: relative weight of the classification error in the matching cost
            cost_bbox: relative weight of the L1 error of the bounding box coordinates
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        assert cost_class != 0 or cost_bbox != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, torch.Tensor], targets: list[dict[str, Any]]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Perform the matching.

        Params:
            outputs: dict containing at least these entries:
                 "pred_node_logits": Tensor [batch_size, num_queries, num_classes]
                 "pred_boxes": Tensor [batch_size, num_queries, 4]

            targets: list of targets (len(targets) = batch_size), each dict containing:
                 "labels": Tensor [num_target_boxes]
                 "boxes": Tensor [num_target_boxes, 4]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
        """
        bs, num_queries = outputs["pred_node_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_node_logits"].flatten(0, 1).softmax(-1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]

        # Create cost matrix
        C = self.cost_class * cost_class

        if self.cost_bbox > 0:
            out_bbox = outputs["pred_boxes"].flatten(0, 1)
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            target_bbox_loss_mask = torch.cat([v["bbox_loss_mask"] for v in targets])
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_bbox = target_bbox_loss_mask.unsqueeze(0) * cost_bbox
            C += self.cost_bbox * cost_bbox

        if self.cost_giou > 0:
            out_bbox = outputs["pred_boxes"].flatten(0, 1)
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            target_bbox_loss_mask = torch.cat([v["bbox_loss_mask"] for v in targets])
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
            )
            cost_giou = target_bbox_loss_mask.unsqueeze(0) * cost_giou
            C += self.cost_giou * cost_giou

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


def build_matcher(args: Any) -> HungarianMatcher:
    """Build the Hungarian matcher from config args."""
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
    )

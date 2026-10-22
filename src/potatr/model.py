# Copyright 2025-present Kensho Technologies, LLC.
# Portions copyright (C) 2023 Microsoft Corporation, from LOBSTR
# (https://github.com/microsoft/LOBSTR), MIT License.
# Portions derived from DETR (https://github.com/facebookresearch/detr),
# Copyright (c) Facebook, Inc. and its affiliates, Apache License 2.0.
# Third-party portions have been modified. See THIRD_PARTY_NOTICES.md.
"""POTATR model and criterion classes."""

from collections import defaultdict
from collections.abc import Iterable
import logging
from typing import Any, cast

import numpy as np
from safetensors.torch import load_file
import torch
from torch import nn
import torch.nn.functional as F

from potatr.config import has_loss_settings
from potatr.detr.models.backbone import build_backbone
from potatr.detr.models.transformer import Transformer, build_transformer
from potatr.detr.util import box_ops
from potatr.detr.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from potatr.matcher import build_matcher


logger = logging.getLogger(__name__)


class POTATR(nn.Module):
    """This is the POTATR module that performs graph recognition."""

    def __init__(
        self,
        backbone: nn.Module,
        transformer: Transformer,
        num_node_classes: int,
        num_edge_classes: int,
        num_queries: int,
        node_embed_hidden_layers: int = 3,
        node_embed_hidden_dim: int = 256,
        edge_embed_hidden_layers: int = 3,
        edge_embed_hidden_dim: int = 256,
        bbox_embed_hidden_layers: int = 3,
        bbox_embed_hidden_dim: int = 256,
        pre_adj_dim: int = 256,
        aux_loss: bool = False,
    ) -> None:
        """Initialize the model.

        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_node_classes: number of node classes
            num_edge_classes: number of edge classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of
                         objects POTATR can detect in a single image.
            pre_adj_dim: size each query embedding is projected to before edge prediction. No
                         projection is applied if it equals the transformer's hidden size.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_edge_classes = num_edge_classes

        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.node_class_embed: nn.Module
        if node_embed_hidden_layers > 1:
            self.node_class_embed = MLP(
                hidden_dim, node_embed_hidden_dim, num_node_classes + 1, node_embed_hidden_layers
            )
        else:
            self.node_class_embed = nn.Linear(hidden_dim, num_node_classes + 1)

        # Edges are predicted for every ordered pair of queries, from their concatenated
        # (optionally projected) embeddings.
        self.edge_class_embed: nn.Module
        if edge_embed_hidden_layers > 1:
            self.edge_class_embed = MLP(
                pre_adj_dim * 2,
                edge_embed_hidden_dim,
                num_edge_classes + 1,
                edge_embed_hidden_layers,
            )
        else:
            self.edge_class_embed = nn.Linear(pre_adj_dim * 2, num_edge_classes + 1)

        self.pre_adj_layer: nn.Module
        if pre_adj_dim != hidden_dim:
            self.pre_adj_layer = nn.Linear(hidden_dim, pre_adj_dim)
        else:
            self.pre_adj_layer = nn.Identity()

        self.bbox_embed: nn.Module
        if bbox_embed_hidden_layers > 1:
            self.bbox_embed = MLP(hidden_dim, bbox_embed_hidden_dim, 4, bbox_embed_hidden_layers)
        else:
            self.bbox_embed = nn.Linear(hidden_dim, 4)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # build_backbone sets num_channels on the backbone module, so mypy can't see it.
        num_channels = cast(int, backbone.num_channels)
        self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor | torch.Tensor | list[torch.Tensor]) -> dict[str, Any]:
        """Forward pass.

        The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_node_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
           - "pred_edge_logits": the edge classification logits for all query pairs.
        """
        if isinstance(samples, (list, torch.Tensor)):
            # A batched tensor works too: it's iterated over like a list of images.
            samples = nested_tensor_from_tensor_list(samples)  # type: ignore[arg-type]
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        # Edge logits for every ordered pair of queries, from their concatenated embeddings.
        hs1 = self.pre_adj_layer(hs)
        hs2 = torch.cat(
            [
                hs1.repeat(1, 1, self.num_queries, 1),
                torch.repeat_interleave(hs1, self.num_queries, 2),
            ],
            dim=3,
        )
        outputs_edge_class = self.edge_class_embed(hs2)
        outputs_edge_class = outputs_edge_class.reshape(
            outputs_edge_class.shape[0],
            outputs_edge_class.shape[1],
            self.num_queries,
            self.num_queries,
            self.num_edge_classes + 1,
        )

        outputs_node_class = self.node_class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {
            "pred_node_logits": outputs_node_class[-1],
            "pred_edge_logits": outputs_edge_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_node_class, outputs_edge_class, outputs_coord
            )
        return out

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_node_class: torch.Tensor,
        outputs_edge_class: torch.Tensor,
        outputs_coord: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Set auxiliary loss outputs from intermediate decoder layers."""
        return [
            {"pred_node_logits": a, "pred_edge_logits": b, "pred_boxes": c}
            for a, b, c in zip(
                outputs_node_class[:-1],
                outputs_edge_class[:-1],
                outputs_coord[:-1],
            )
        ]


class SetCriterion(nn.Module):
    """Compute the loss for POTATR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    empty_weight: torch.Tensor  # registered as a buffer in __init__

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: dict[str, float],
        eos_coef: float,
        losses: list[str],
        adj_class_weights: list[float] | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        """Create the criterion.

        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        if adj_class_weights is None:
            adj_class_weights = [0.2, 2]
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.adj_class_weights = torch.as_tensor(adj_class_weights).to(device)

    def loss_labels(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_objects: float,
        log: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Classification loss (Negative Log Likelihood).

        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_node_logits" in outputs
        src_logits = outputs["pred_node_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_objects: float,
    ) -> dict[str, torch.Tensor]:
        """Compute the cardinality error.

        This is not really a loss, it is intended for logging purposes only.
        It doesn't propagate gradients.
        """
        pred_logits = outputs["pred_node_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_objects: float,
    ) -> dict[str, torch.Tensor]:
        """Compute the bounding box losses: L1 regression loss and GIoU loss.

        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4].
        The target boxes are expected in format (center_x, center_y, w, h), normalized by image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_loss_mask = torch.cat(
            [t["bbox_loss_mask"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = target_loss_mask.unsqueeze(1) * loss_bbox

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_objects

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)
            )
        )
        loss_giou = target_loss_mask * loss_giou
        losses["loss_giou"] = loss_giou.sum() / num_objects

        return losses

    def loss_adjacencies(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_objects: float,
    ) -> dict[str, torch.Tensor]:
        """Adjacency loss.

        targets dicts must contain the key "adjacencies".
        """
        assert "pred_edge_logits" in outputs

        batch_size = len(indices)

        loss_adj = torch.tensor(0.0, device=outputs["pred_edge_logits"].device)
        for batch_num in range(batch_size):
            prediction_idx = indices[batch_num][0]
            target_idx = indices[batch_num][1]

            src_adjacencies = outputs["pred_edge_logits"][batch_num]
            src_adjacencies = src_adjacencies[
                np.ix_(
                    prediction_idx.numpy(),
                    prediction_idx.numpy(),
                    np.arange(src_adjacencies.shape[2]),
                )
            ]

            selected_targets = targets[batch_num]["adjacencies"][
                np.ix_(target_idx.numpy(), target_idx.numpy())
            ]

            for query_num in range(src_adjacencies.shape[0]):
                this_src = src_adjacencies[query_num, :, :].squeeze()
                this_target = selected_targets[query_num, :].squeeze().long()
                loss_adj += F.cross_entropy(this_src, this_target, weight=self.adj_class_weights)

        losses = {"loss_adj": loss_adj / num_objects}

        return losses

    def _get_src_permutation_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Permute predictions following indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(
        self,
        loss: str,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_objects: float,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Dispatch to the appropriate loss function."""
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "adj": self.loss_adjacencies,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_objects, **kwargs)

    def forward(
        self, outputs: dict[str, Any], targets: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """Perform the loss computation.

        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization
        num_objects_tensor = torch.as_tensor(
            [sum(len(t["labels"]) for t in targets)],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects_tensor)
        num_objects = torch.clamp(num_objects_tensor / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_objects))

        # In case of auxiliary losses, repeat with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_objects, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """Convert the model's output into the format expected by the COCO API."""

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, Any], target_sizes: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        """Perform the computation.

        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each image
        """
        out_logits, out_bbox = outputs["pred_node_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        """Initialize the MLP."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def summarize_keys(keys: Iterable[str]) -> str:
    """Summarize state dict keys by top-level module, e.g. 'bbox_embed (6), pre_adj_layer (2)'."""
    counts: defaultdict[str, int] = defaultdict(int)
    for key in keys:
        counts[key.split(".")[0]] += 1
    return ", ".join(f"{module} ({count})" for module, count in sorted(counts.items()))


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a checkpoint file: model weights in a .safetensors file, or a PyTorch .pth file.

    A .pth file can hold a plain state dict or a full training checkpoint (see load_model_weights).
    A .safetensors file is read on the CPU; load_model_weights copies its tensors to the model.
    """
    if path.endswith(".safetensors"):
        return load_file(path)
    checkpoint: dict[str, Any] = torch.load(path, map_location=map_location)
    return checkpoint


def load_model_weights(
    model: nn.Module, checkpoint: dict[str, Any], require_all: bool = False
) -> None:
    """Load the tensors of a checkpoint that match the model by name and shape.

    The checkpoint can be a plain state dict or a full training checkpoint, whose weights are
    under "model_state_dict". Tensors that don't match are reported. Model tensors that aren't
    loaded keep their initial values, which is expected when fine-tuning from a checkpoint
    without POTATR's heads; with require_all, they raise an error instead.
    """
    loaded_state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state_dict = model.state_dict()
    matching = {
        k: v
        for k, v in loaded_state_dict.items()
        if k in model_state_dict and model_state_dict[k].shape == v.shape
    }
    unexpected = [k for k in loaded_state_dict if k not in model_state_dict]
    mismatched = [
        k for k, v in loaded_state_dict.items() if k in model_state_dict and k not in matching
    ]
    missing = [k for k in model_state_dict if k not in matching]

    model_state_dict.update(matching)
    model.load_state_dict(model_state_dict, strict=True)

    logger.info(
        "Loaded %d of %d model tensors from checkpoint", len(matching), len(model_state_dict)
    )
    if unexpected:
        logger.warning(
            "Ignored %d checkpoint tensors not in the model: %s",
            len(unexpected),
            summarize_keys(unexpected),
        )
    if mismatched:
        logger.warning(
            "Ignored %d checkpoint tensors with a different shape: %s",
            len(mismatched),
            summarize_keys(mismatched),
        )
    if missing:
        message = (
            f"{len(missing)} model tensors were not in the checkpoint and keep their initial "
            f"values: {summarize_keys(missing)}"
        )
        if require_all:
            raise ValueError(message)
        logger.warning(message)


def build(args: Any) -> tuple[POTATR, SetCriterion | None, dict[str, nn.Module]]:
    """Build the POTATR model, criterion and postprocessors from config args.

    The criterion is None if the config has no loss settings (possible for evaluation).
    """
    num_classes = args.num_node_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = POTATR(
        backbone,
        transformer,
        num_node_classes=args.num_node_classes,
        num_edge_classes=args.num_edge_classes,
        num_queries=args.num_queries,
        node_embed_hidden_layers=args.node_embed_hidden_layers,
        node_embed_hidden_dim=args.node_embed_hidden_dim,
        edge_embed_hidden_layers=args.edge_embed_hidden_layers,
        edge_embed_hidden_dim=args.edge_embed_hidden_dim,
        bbox_embed_hidden_layers=args.bbox_embed_hidden_layers,
        bbox_embed_hidden_dim=args.bbox_embed_hidden_dim,
        pre_adj_dim=args.pre_adj_dim,
        aux_loss=args.aux_loss,
    )
    postprocessors: dict[str, nn.Module] = {"bbox": PostProcess()}

    # The loss settings are optional for evaluation; without them there is no criterion.
    if not has_loss_settings(vars(args)):
        return model, None, postprocessors

    matcher = build_matcher(args)
    weight_dict = {
        "loss_ce": args.ce_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_adj": args.adj_loss_coef,
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["cardinality", "labels"]
    if args.bbox_loss_coef > 0:
        losses.append("boxes")
    if args.adj_loss_coef > 0:
        losses.append("adj")

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        adj_class_weights=args.adj_class_weights,
        device=args.device,
    )
    criterion.to(device)

    return model, criterion, postprocessors

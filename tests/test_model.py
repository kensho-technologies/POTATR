# Copyright 2025-present Kensho Technologies, LLC.
"""Smoke tests for POTATR model instantiation."""

import torch

from potatr import build
from potatr.config import LOSS_KEYS


def _make_args():
    """Create a minimal args object for model construction."""
    return type(
        "Args",
        (object,),
        {
            "device": "cpu",
            "backbone": "resnet18",
            "dilation": False,
            "position_embedding": "sine",
            "enc_layers": 2,
            "dec_layers": 2,
            "dim_feedforward": 512,
            "hidden_dim": 128,
            "dropout": 0.1,
            "nheads": 4,
            "num_queries": 10,
            "pre_norm": True,
            "num_node_classes": 4,
            "num_edge_classes": 1,
            "node_embed_hidden_layers": 1,
            "node_embed_hidden_dim": 128,
            "edge_embed_hidden_layers": 2,
            "edge_embed_hidden_dim": 128,
            "bbox_embed_hidden_layers": 2,
            "bbox_embed_hidden_dim": 128,
            "pre_adj_dim": 64,
            "aux_loss": False,
            "masks": False,
            "set_cost_class": 1,
            "set_cost_bbox": 5,
            "set_cost_giou": 2,
            "ce_loss_coef": 1,
            "bbox_loss_coef": 5,
            "giou_loss_coef": 2,
            "adj_loss_coef": 0.05,
            "eos_coef": 0.1,
            "adj_class_weights": [1.0, 1.0],
            "lr_backbone": 1e-5,
        },
    )


def test_model_builds():
    """Test that the POTATR model can be instantiated."""
    args = _make_args()
    model, criterion, postprocessors = build(args)

    assert model is not None
    assert criterion is not None
    assert "bbox" in postprocessors


def test_model_builds_without_loss_settings():
    """Without loss settings (possible for evaluation), there is no criterion."""
    settings = {
        k: v
        for k, v in vars(_make_args()).items()
        if not k.startswith("__") and k not in LOSS_KEYS - {"aux_loss"}
    }
    model, criterion, postprocessors = build(type("Args", (object,), settings))

    assert model is not None
    assert criterion is None
    assert "bbox" in postprocessors


def test_model_forward_pass():
    """Test that the model can perform a forward pass on CPU."""
    args = _make_args()
    model, criterion, postprocessors = build(args)
    model.eval()

    # Create a dummy input: batch of 1 image, 3 channels, 100x100
    dummy_input = torch.randn(1, 3, 100, 100)

    with torch.no_grad():
        outputs = model(dummy_input)

    assert "pred_node_logits" in outputs
    assert "pred_boxes" in outputs
    assert "pred_edge_logits" in outputs
    assert outputs["pred_node_logits"].shape == (1, 10, 5)  # num_queries, num_classes+1
    assert outputs["pred_boxes"].shape == (1, 10, 4)


def test_criterion_computes_finite_losses():
    """The loss for a batch of one image with two target objects linked by an edge."""
    args = _make_args()
    model, criterion, _ = build(args)
    model.eval()
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.3], [0.5, 0.3, 0.4, 0.05]]),
            "labels": torch.tensor([0, 3]),
            "adjacencies": torch.tensor([[0, 1], [1, 0]]),
            "bbox_loss_mask": torch.ones(2, dtype=torch.int64),
        }
    ]

    with torch.no_grad():
        losses = criterion(model(torch.randn(1, 3, 100, 100)), targets)

    for name in ("loss_ce", "loss_bbox", "loss_giou", "loss_adj", "cardinality_error"):
        assert name in losses
        assert torch.isfinite(losses[name])
    total = sum(losses[k] * criterion.weight_dict[k] for k in losses if k in criterion.weight_dict)
    assert total > 0


def test_load_checkpoint_reads_pth_and_safetensors(tmp_path):
    from safetensors.torch import save_file

    from potatr.model import load_checkpoint

    weights = {"query_embed.weight": torch.randn(3, 4), "class_embed.bias": torch.randn(5)}
    save_file(weights, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    torch.save({"epoch": 2, "model_state_dict": weights}, str(tmp_path / "model.pth"))

    from_safetensors = load_checkpoint(str(tmp_path / "model.safetensors"))
    from_pth = load_checkpoint(str(tmp_path / "model.pth"))

    assert from_safetensors.keys() == weights.keys()
    assert all(torch.equal(from_safetensors[k], weights[k]) for k in weights)
    assert from_pth["epoch"] == 2
    assert all(torch.equal(from_pth["model_state_dict"][k], weights[k]) for k in weights)

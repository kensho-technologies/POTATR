# Copyright 2025-present Kensho Technologies, LLC.
"""Tests for running POTATR on page images."""

import json

import pytest
import torch

from potatr.config import MODEL_KEYS
from potatr.inference import load_model, tables_from_outputs
from potatr.postprocess import class_idx2name


PAGE_SIZE = (1000, 800)  # (width, height)


def _outputs(objects, edges):
    """Make model outputs for one page: (class name, box, logit) objects, and the linked pairs."""
    width, height = PAGE_SIZE
    logits = torch.zeros(1, len(objects), len(class_idx2name))
    boxes = torch.zeros(1, len(objects), 4)
    for i, (label, (x0, y0, x1, y1), logit) in enumerate(objects):
        logits[0, i, class_idx2name.index(label)] = logit
        boxes[0, i] = torch.tensor(
            [(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height]
        )
    edge_logits = torch.zeros(1, len(objects), len(objects), 2)
    edge_logits[..., 0] = 5.0
    for i, j in edges:
        edge_logits[0, i, j, 1] = edge_logits[0, j, i, 1] = 10.0
    return {"pred_node_logits": logits, "pred_boxes": boxes, "pred_edge_logits": edge_logits}


def _word(text, bbox):
    return {"text": text, "bbox": bbox, "block_num": 0, "line_num": 0, "span_num": 0}


def test_each_table_gets_its_own_caption():
    # The second table is more confident than the first, so ordering by score would swap them.
    outputs = _outputs(
        [
            ("table", [100.0, 100.0, 400.0, 300.0], 4.0),
            ("table caption", [100.0, 60.0, 400.0, 90.0], 10.0),
            ("table", [500.0, 400.0, 900.0, 700.0], 10.0),
            ("table caption", [500.0, 360.0, 900.0, 390.0], 10.0),
        ],
        edges=[(0, 1), (2, 3)],
    )
    tokens = [
        _word("First", [150.0, 65.0, 250.0, 85.0]),
        _word("Second", [550.0, 365.0, 650.0, 385.0]),
    ]

    tables = tables_from_outputs(outputs, PAGE_SIZE, tokens, class_idx2name)

    assert [table["caption"] for table in tables] == ["First", "Second"]
    assert tables[0]["bbox"] == pytest.approx([100.0, 100.0, 400.0, 300.0], abs=1e-3)
    assert tables[0]["score"] < tables[1]["score"]
    assert tables[0]["caption_parts"][0]["edge_score"] > 0.99


# A small model, so the test is quick.
SMALL_MODEL = {
    "backbone": "resnet18",
    "dilation": False,
    "masks": False,
    "position_embedding": "sine",
    "enc_layers": 1,
    "dec_layers": 1,
    "dim_feedforward": 64,
    "hidden_dim": 32,
    "nheads": 4,
    "num_queries": 5,
    "pre_norm": True,
    "num_node_classes": 4,
    "num_edge_classes": 1,
    "node_embed_hidden_layers": 1,
    "node_embed_hidden_dim": 32,
    "edge_embed_hidden_layers": 2,
    "edge_embed_hidden_dim": 32,
    "bbox_embed_hidden_layers": 2,
    "bbox_embed_hidden_dim": 32,
    "pre_adj_dim": 16,
}


def test_load_model_loads_every_tensor(tmp_path):
    assert set(SMALL_MODEL) <= MODEL_KEYS
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(SMALL_MODEL))
    model, _ = load_model([str(config_path)], _save_random_weights(tmp_path, config_path), "cpu")
    reloaded, _ = load_model([str(config_path)], str(tmp_path / "weights.pth"), "cpu")

    assert not model.training
    for (name, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert torch.equal(a, b), name

    # A checkpoint missing a tensor is an error, not a silently random tensor.
    state_dict = torch.load(tmp_path / "weights.pth")
    del state_dict["bbox_embed.layers.0.weight"]
    torch.save(state_dict, tmp_path / "incomplete.pth")
    with pytest.raises(ValueError, match="bbox_embed"):
        load_model([str(config_path)], str(tmp_path / "incomplete.pth"), "cpu")


def _save_random_weights(tmp_path, config_path):
    """Save the weights of a randomly initialized model with the given config."""
    from potatr.model import build

    config = json.loads(config_path.read_text())
    config.update(device="cpu", dropout=0.0, lr_backbone=0.0, aux_loss=False)
    model, _, _ = build(type("Args", (object,), config))
    path = tmp_path / "weights.pth"
    torch.save(model.state_dict(), path)
    return str(path)

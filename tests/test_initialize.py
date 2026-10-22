# Copyright 2025-present Kensho Technologies, LLC.
"""Tests for making POTATR's starting checkpoint from Table Transformer weights."""

import pytest
import torch

from potatr.initialize import file_sha256, make_initial_checkpoint


def _tatr_state_dict():
    return {
        "backbone.0.body.conv1.weight": torch.randn(4, 3, 7, 7),
        "transformer.encoder.layers.0.linear1.weight": torch.randn(8, 4),
        "bbox_embed.layers.0.weight": torch.randn(4, 4),
        "class_embed.weight": torch.randn(7, 4),
        "class_embed.bias": torch.randn(7),
        "query_embed.weight": torch.randn(5, 4),
    }


def test_copies_tatr_tensors_and_adds_queries():
    tatr = _tatr_state_dict()
    state_dict = make_initial_checkpoint(tatr, num_queries=12)

    assert "class_embed.weight" not in state_dict
    assert "class_embed.bias" not in state_dict
    for name in ("backbone.0.body.conv1.weight", "transformer.encoder.layers.0.linear1.weight"):
        assert torch.equal(state_dict[name], tatr[name])
    queries = state_dict["query_embed.weight"]
    assert queries.shape == (12, 4)
    assert torch.equal(queries[:5], tatr["query_embed.weight"])


def test_new_queries_depend_only_on_the_seed():
    tatr = _tatr_state_dict()
    torch.manual_seed(0)
    first = make_initial_checkpoint(tatr, num_queries=12)["query_embed.weight"]
    torch.manual_seed(1)  # the global random state doesn't matter
    second = make_initial_checkpoint(tatr, num_queries=12)["query_embed.weight"]
    other_seed = make_initial_checkpoint(tatr, num_queries=12, seed=7)["query_embed.weight"]

    assert torch.equal(first, second)
    assert not torch.equal(first[5:], other_seed[5:])


def test_fewer_queries_than_tatr_raises():
    with pytest.raises(ValueError, match="num_queries"):
        make_initial_checkpoint(_tatr_state_dict(), num_queries=3)


def test_file_sha256(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")
    assert file_sha256(str(path)) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )

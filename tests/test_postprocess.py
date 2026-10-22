# Copyright 2025-present Kensho Technologies, LLC.
"""Tests for converting model outputs into table cells."""

import copy

import pytest
import torch

from potatr.postprocess import (
    class_idx2name,
    potatr_outputs_to_objects,
    potatr_outputs_to_tables_cells,
)


PAGE_SIZE = (1000, 800)  # (width, height)


def _outputs(objects):
    """Make model outputs that confidently predict the given (class name, page box) objects."""
    width, height = PAGE_SIZE
    logits = torch.zeros(len(objects), len(class_idx2name))
    boxes = torch.zeros(len(objects), 4)
    for i, (label, (x0, y0, x1, y1)) in enumerate(objects):
        logits[i, class_idx2name.index(label)] = 10.0
        boxes[i] = torch.tensor(
            [(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height]
        )
    return logits, boxes


def _word(text, bbox):
    return {"text": text, "bbox": bbox, "block_num": 0, "line_num": 0, "span_num": 0}


def test_rotated_table_does_not_change_the_callers_words():
    # A rotated table that is a single cell, plus a word inside it and a word outside it.
    table_box = [100.0, 100.0, 300.0, 500.0]
    logits, boxes = _outputs(
        [
            ("table rotated", table_box),
            ("table row rotated", table_box),
            ("table column rotated", table_box),
        ]
    )
    tokens = [
        _word("Revenue", [150.0, 200.0, 250.0, 300.0]),
        _word("2024", [700.0, 300.0, 740.0, 320.0]),
    ]
    tokens_before = copy.deepcopy(tokens)

    tables_cells = potatr_outputs_to_tables_cells(logits, boxes, tokens, PAGE_SIZE)

    assert tokens == tokens_before
    assert [[cell["cell_text"] for cell in cells] for cells in tables_cells] == [["Revenue"]]


def test_outputs_to_objects_keeps_confident_objects_in_pixels():
    logits, boxes = _outputs(
        [
            ("table", [100.0, 80.0, 500.0, 400.0]),
            ("no object", [0.0, 0.0, 10.0, 10.0]),
            ("table row", [100.0, 80.0, 500.0, 160.0]),
        ]
    )
    logits[2] = 0.0  # every class equally likely, so no class is above 0.5

    objects = potatr_outputs_to_objects(logits, boxes, PAGE_SIZE)

    assert [(obj["idx"], obj["label"]) for obj in objects] == [(0, "table")]
    assert objects[0]["score"] > 0.99
    assert objects[0]["bbox"] == pytest.approx([100.0, 80.0, 500.0, 400.0], abs=1e-3)


def test_table_structure_to_cells():
    # A 2x2 table with a column header over the first row, and one word in each cell.
    logits, boxes = _outputs(
        [
            ("table", [100.0, 100.0, 500.0, 300.0]),
            ("table row", [100.0, 100.0, 500.0, 200.0]),
            ("table row", [100.0, 200.0, 500.0, 300.0]),
            ("table column", [100.0, 100.0, 300.0, 300.0]),
            ("table column", [300.0, 100.0, 500.0, 300.0]),
            ("table column header", [100.0, 100.0, 500.0, 200.0]),
        ]
    )
    tokens = [
        _word("Name", [150.0, 140.0, 250.0, 160.0]),
        _word("Value", [350.0, 140.0, 450.0, 160.0]),
        _word("a", [150.0, 240.0, 170.0, 260.0]),
        _word("1", [350.0, 240.0, 370.0, 260.0]),
    ]

    tables_cells = potatr_outputs_to_tables_cells(logits, boxes, tokens, PAGE_SIZE)

    assert len(tables_cells) == 1
    cells = {(cell["row_nums"][0], cell["column_nums"][0]): cell for cell in tables_cells[0]}
    assert {k: cell["cell_text"] for k, cell in cells.items()} == {
        (0, 0): "Name",
        (0, 1): "Value",
        (1, 0): "a",
        (1, 1): "1",
    }
    assert {k: cell["header"] for k, cell in cells.items()} == {
        (0, 0): True,
        (0, 1): True,
        (1, 0): False,
        (1, 1): False,
    }

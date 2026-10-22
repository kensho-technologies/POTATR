# Copyright 2025-present Kensho Technologies, LLC.
"""Tests for the table structure recognition metrics."""

from grits import TableCell, cell_list_to_grid_con, cell_list_to_grid_top
import pytest

from potatr.metrics import (
    cells_to_html,
    compute_grits_metrics,
    compute_metrics_summary,
    compute_teds_metrics,
    lcs_length,
)


# A 2x2 table whose header cell spans both columns.
CELLS = [
    {"row_nums": [0], "column_nums": [0, 1], "cell_text": "Results", "header": True},
    {"row_nums": [1], "column_nums": [0], "cell_text": "a", "header": False},
    {"row_nums": [1], "column_nums": [1], "cell_text": "1", "header": False},
]


def test_lcs_length():
    assert lcs_length("", "abc") == 0
    assert lcs_length("abc", "abc") == 3
    assert lcs_length("abcde", "ace") == 3
    assert lcs_length("abc", "xyz") == 0


def test_cells_to_html():
    assert cells_to_html(CELLS) == (
        "<html><table>"
        '<thead><tr><th colspan="2">Results</th></tr></thead>'
        "<tbody><tr><td>a</td><td>1</td></tr></tbody>"
        "</table></html>"
    )


def test_grits_of_a_perfect_prediction():
    table_cells = [TableCell.from_dict(c) for c in CELLS]
    true_grids_con = [cell_list_to_grid_con(table_cells)]
    true_grids_top = [cell_list_to_grid_top(table_cells)]

    metrics, matched = compute_grits_metrics(true_grids_con, true_grids_top, [CELLS])

    for measure in ("grits_top", "grits_con"):
        assert metrics[f"{measure}_num_true"] == 4  # grid cells
        assert metrics[f"{measure}_num_positive"] == 4
        assert metrics[f"{measure}_true_positive_score"] == 4
        assert metrics[f"{measure}_num_exact_match"] == 1
    assert [list(indices) for indices in matched["con"]] == [[0], [0]]


def test_grits_with_no_predicted_table():
    table_cells = [TableCell.from_dict(c) for c in CELLS]
    metrics, _ = compute_grits_metrics(
        [cell_list_to_grid_con(table_cells)], [cell_list_to_grid_top(table_cells)], []
    )
    assert metrics["grits_con_num_true"] == 4
    assert metrics["grits_con_num_positive"] == 0
    assert metrics["grits_con_true_positive_score"] == 0


def test_teds_of_a_perfect_prediction():
    metrics = compute_teds_metrics([cells_to_html(CELLS)], [CELLS])
    for measure in ("teds", "teds_struct"):
        assert metrics[f"{measure}_num_true"] == 1
        assert metrics[f"{measure}_num_positive"] == 1
        assert metrics[f"{measure}_true_positive_score"] == 1.0
        assert metrics[f"{measure}_num_exact_match"] == 1


def _sample_metrics(caption_true, caption_positive, caption_tp):
    """Per-page metrics with the given caption counts (characters) and no other content."""
    metrics = {"num_tables": 1}
    for name in ("grits_top", "grits_con", "teds", "teds_struct", "edges@iou=0.5", "edges@iou=0.8"):
        metrics.update(
            {f"{name}_true_positive_score": 0, f"{name}_num_true": 0, f"{name}_num_positive": 0}
        )
    for text_type, (true, positive, tp) in [
        ("caption", (caption_true, caption_positive, caption_tp)),
        ("footer", (0, 0, 0)),
    ]:
        metrics[f"{text_type}_num_true"] = true
        metrics[f"{text_type}_num_positive"] = positive
        for matching in ("iou", "con", "top"):
            metrics[f"{text_type}_lcs_tp_{matching}"] = tp
    return metrics


def test_caption_text_f1_pools_characters_over_pages():
    # 15 of 20 true characters found, among 20 predicted: precision and recall 0.75.
    summary = compute_metrics_summary([_sample_metrics(10, 8, 6), _sample_metrics(10, 12, 9)])
    assert summary["caption_text_precision@iou"] == pytest.approx(0.75)
    assert summary["caption_text_recall@iou"] == pytest.approx(0.75)
    assert summary["caption_text_f1@iou"] == pytest.approx(0.75)
    assert summary["caption_num_true_chars"] == 20

# Copyright 2025-present Kensho Technologies, LLC.
# Portions copyright (C) 2021 Microsoft Corporation, from Table Transformer
# (https://github.com/microsoft/table-transformer), MIT License.
# Third-party portions have been modified. See THIRD_PARTY_NOTICES.md.
"""Pure metric and formatting functions for table structure recognition evaluation.

Extracted from evaluate.py — no model, GPU, or I/O dependencies.
"""

import logging
from typing import Any
import xml.etree.ElementTree as ET

from grits import (
    TableCell,
    cell_list_to_grid_con,
    cell_list_to_grid_top,
    compute_fscore,
    hungarian_grits_con_matching,
    hungarian_grits_top_matching,
)

from potatr import teds


logger = logging.getLogger(__name__)


def lcs_length(s1: str, s2: str) -> int:
    """Compute the length of the longest common subsequence of two strings."""
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def cells_to_html(cells: list[dict[str, Any]]) -> str:
    """Convert table cells to an HTML table string, the form TEDS compares.

    Cells are placed in order of their first row and first column, with rowspan and colspan for
    the rows and columns they cover. A row whose first cell is a header cell goes in <thead>, with
    <th> cells; the other rows go in <tbody>.
    """
    cells = sorted(cells, key=lambda k: min(k["column_nums"]))
    cells = sorted(cells, key=lambda k: min(k["row_nums"]))

    html = ET.Element("html")
    table = ET.SubElement(html, "table")
    current_row = -1
    header = None
    body = None

    for cell in cells:
        this_row = min(cell["row_nums"])

        attrib = {}
        colspan = len(cell["column_nums"])
        if colspan > 1:
            attrib["colspan"] = str(colspan)
        rowspan = len(cell["row_nums"])
        if rowspan > 1:
            attrib["rowspan"] = str(rowspan)
        if this_row > current_row:
            current_row = this_row
            if "header" in cell and cell["header"]:
                cell_tag = "th"
                if header is None:
                    header = ET.SubElement(table, "thead")
                row = ET.SubElement(header, "tr")
            else:
                cell_tag = "td"
                if body is None:
                    body = ET.SubElement(table, "tbody")
                row = ET.SubElement(body, "tr")
        tcell = ET.SubElement(row, cell_tag, attrib=attrib)
        tcell.text = cell["cell_text"]

    return str(ET.tostring(html, encoding="unicode", short_empty_elements=False))


def compute_grits_metrics(
    true_grids_con: list[Any],
    true_grids_top: list[Any],
    tables_pred_cells: list[list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    """Compute GriTS_Top and GriTS_Con for one page's predicted tables against its true tables.

    Returns (metrics_dict, matched_indices). For each measure, metrics_dict holds the
    true-positive score, the true and predicted grid-cell counts and the number of exactly
    matching tables, summed over the page. matched_indices contains the Hungarian assignment
    indices for both GriTS_Con and GriTS_Top.
    """
    metrics = {}

    tables_pred_table_cells = [
        [TableCell.from_dict(c) for c in pred_cells] for pred_cells in tables_pred_cells
    ]
    pred_grids_top = [cell_list_to_grid_top(tc) for tc in tables_pred_table_cells]
    pred_grids_con = [cell_list_to_grid_con(tc) for tc in tables_pred_table_cells]

    top_result = hungarian_grits_top_matching(true_grids_top, pred_grids_top)
    metrics["grits_top_true_positive_score"] = top_result.true_positive_score
    metrics["grits_top_num_true"] = sum(len(g) * len(g[0]) for g in true_grids_top if len(g) > 0)
    metrics["grits_top_num_positive"] = sum(
        len(g) * len(g[0]) for g in pred_grids_top if len(g) > 0
    )
    metrics["grits_top_num_exact_match"] = top_result.num_exact_grid_matches

    con_result = hungarian_grits_con_matching(true_grids_con, pred_grids_con)
    metrics["grits_con_true_positive_score"] = con_result.true_positive_score
    metrics["grits_con_num_true"] = sum(len(g) * len(g[0]) for g in true_grids_con if len(g) > 0)
    metrics["grits_con_num_positive"] = sum(
        len(g) * len(g[0]) for g in pred_grids_con if len(g) > 0
    )
    metrics["grits_con_num_exact_match"] = con_result.num_exact_grid_matches

    matched_indices = {
        "con": (con_result.matched_true_indices, con_result.matched_pred_indices),
        "top": (top_result.matched_true_indices, top_result.matched_pred_indices),
    }

    return metrics, matched_indices


def compute_teds_metrics(
    true_htmls: list[str], tables_pred_cells: list[list[dict[str, Any]]]
) -> dict[str, Any]:
    """Compute TEDS metrics"""
    metrics = {}

    pred_htmls = [cells_to_html(pred_cells) for pred_cells in tables_pred_cells]

    # Compute TEDS
    (
        metrics["teds_num_true"],
        metrics["teds_num_positive"],
        metrics["teds_true_positive_score"],
        metrics["teds_num_exact_match"],
    ) = teds.hungarian_teds(true_htmls, pred_htmls, structure_only=False)

    # Compute TEDS
    (
        metrics["teds_struct_num_true"],
        metrics["teds_struct_num_positive"],
        metrics["teds_struct_true_positive_score"],
        metrics["teds_struct_num_exact_match"],
    ) = teds.hungarian_teds(true_htmls, pred_htmls, structure_only=True)

    return metrics


def compute_metrics_summary(sample_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute dataset-level metrics from the per-sample metrics, and return them as a dict.

    The F1, precision and recall of each GriTS, TEDS and edge measure pool the true positives and
    counts of all samples. Also computes the fraction of tables matched exactly, and the caption
    and footer text F1 under each table-matching method.
    """

    metrics_summary = {}

    metric_names = [
        "grits_top",
        "grits_con",
        "teds",
        "teds_struct",
        "edges@iou=0.5",
        "edges@iou=0.8",
    ]

    total_tables = sum([elem["num_tables"] for elem in sample_metrics])

    if len(sample_metrics) > 0:
        for metric_name in metric_names:
            total_true_positives = sum(
                [elem[metric_name + "_true_positive_score"] for elem in sample_metrics]
            )
            total_true = sum([elem[metric_name + "_num_true"] for elem in sample_metrics])
            total_positive = sum([elem[metric_name + "_num_positive"] for elem in sample_metrics])

            fscore, precision, recall = compute_fscore(
                total_true_positives, total_true, total_positive
            )
            metrics_summary[metric_name] = fscore
            metrics_summary[metric_name + "_precision"] = precision
            metrics_summary[metric_name + "_recall"] = recall

            if metric_name + "_num_exact_match" in sample_metrics[0]:
                total_exact_matches = sum(
                    [elem[metric_name + "_num_exact_match"] for elem in sample_metrics]
                )
                accuracy = total_exact_matches / total_tables
                metrics_summary["acc_" + metric_name] = accuracy

        # Caption/footer text F1 under each matching criterion
        for text_type in ["caption", "footer"]:
            total_true = sum(elem[f"{text_type}_num_true"] for elem in sample_metrics)
            total_positive = sum(elem[f"{text_type}_num_positive"] for elem in sample_metrics)
            metrics_summary[f"{text_type}_num_true_chars"] = total_true
            metrics_summary[f"{text_type}_num_positive_chars"] = total_positive
            for matching in ["iou", "con", "top"]:
                total_lcs_tp = sum(
                    elem[f"{text_type}_lcs_tp_{matching}"] for elem in sample_metrics
                )
                fscore, precision, recall = compute_fscore(total_lcs_tp, total_true, total_positive)
                metrics_summary[f"{text_type}_text_f1@{matching}"] = fscore
                metrics_summary[f"{text_type}_text_precision@{matching}"] = precision
                metrics_summary[f"{text_type}_text_recall@{matching}"] = recall
                metrics_summary[f"{text_type}_lcs_tp@{matching}"] = total_lcs_tp

    return metrics_summary


PAD = 40


def log_metrics_line(name: str, metrics_dict: dict[str, Any], key: str, fmt: str = ".4f") -> None:
    """Log one metric with a right-aligned name, or nothing if the metric is missing."""
    padded = name.rjust(PAD)
    try:
        logger.info("%s: %s", padded, format(metrics_dict[key], fmt))
    except KeyError:
        pass


def log_metrics_summary(metrics: dict[str, Any]) -> None:
    """Log the dataset-level metrics from compute_metrics_summary, one per line, at info level.

    Metrics that are missing are skipped.
    """

    logger.info("%s", "-" * 60)

    log_metrics_line("GriTS_Top", metrics, "grits_top")
    log_metrics_line("GriTS_Con", metrics, "grits_con")
    log_metrics_line("TEDS", metrics, "teds")
    log_metrics_line("TEDS_Struct", metrics, "teds_struct")

    log_metrics_line("Edge F1 @ IoU=0.5", metrics, "edges@iou=0.5")
    log_metrics_line("Edge Prec @ IoU=0.5", metrics, "edges@iou=0.5_precision")
    log_metrics_line("Edge Rec @ IoU=0.5", metrics, "edges@iou=0.5_recall")
    log_metrics_line("Edge F1 @ IoU=0.8", metrics, "edges@iou=0.8")
    log_metrics_line("Edge Prec @ IoU=0.8", metrics, "edges@iou=0.8_precision")
    log_metrics_line("Edge Rec @ IoU=0.8", metrics, "edges@iou=0.8_recall")

    if "grits_loc" in metrics:
        log_metrics_line("GriTS_Loc", metrics, "grits_loc")

    log_metrics_line("Accuracy_GriTS_Top", metrics, "acc_grits_top")
    log_metrics_line("Accuracy_GriTS_Con", metrics, "acc_grits_con")
    log_metrics_line("Accuracy_TEDS", metrics, "acc_teds")
    log_metrics_line("Accuracy_TEDS_Struct", metrics, "acc_teds_struct")
    log_metrics_line("Accuracy_Loc", metrics, "acc_grits_loc")

    matching_labels = {
        "iou": "IoU table matching",
        "con": "GriTS_Con table matching",
        "top": "GriTS_Top table matching",
    }
    for text_type, label in [("caption", "Caption"), ("footer", "Footer")]:
        gt_chars = metrics.get(f"{text_type}_num_true_chars", None)
        if gt_chars is None:
            continue
        log_metrics_line(f"{label} GT chars", metrics, f"{text_type}_num_true_chars", fmt="d")
        log_metrics_line(f"{label} Pred chars", metrics, f"{text_type}_num_positive_chars", fmt="d")
        for matching, mlabel in matching_labels.items():
            lcs_key = f"{text_type}_lcs_tp@{matching}"
            if lcs_key not in metrics:
                continue
            log_metrics_line(f"{label} LCS TP (w/ {mlabel})", metrics, lcs_key, fmt="d")
            log_metrics_line(
                f"{label} F1 (w/ {mlabel})", metrics, f"{text_type}_text_f1@{matching}"
            )
            log_metrics_line(
                f"{label} Prec (w/ {mlabel})", metrics, f"{text_type}_text_precision@{matching}"
            )
            log_metrics_line(
                f"{label} Rec (w/ {mlabel})", metrics, f"{text_type}_text_recall@{matching}"
            )

    log_metrics_line("TSR_runtime", metrics, "tsr_runtime")
    log_metrics_line("TSR_post_runtime", metrics, "tsr_post_runtime")
    logger.info("%s", "-" * 60)

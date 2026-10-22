# Copyright 2025-present Kensho Technologies, LLC.
# Portions copyright (C) 2021 Microsoft Corporation, from Table Transformer
# (https://github.com/microsoft/table-transformer), MIT License.
# Portions derived from DETR (https://github.com/facebookresearch/detr),
# Copyright (c) Facebook, Inc. and its affiliates, Apache License 2.0.
# Third-party portions have been modified. See THIRD_PARTY_NOTICES.md.
"""Evaluation harness for POTATR table structure recognition."""

from datetime import datetime
import itertools
import json
import logging
import os
import time
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from potatr import graph_metrics, postprocess
from potatr.datasets import TableGraphDataset
from potatr.detr.datasets.coco_eval import CocoEvaluator
import potatr.detr.util.misc as utils
from potatr.geometry import iou as compute_iou
from potatr.graph_metrics import compute_predicted_edges, compute_target_edges
from potatr.metrics import (
    compute_grits_metrics,
    compute_metrics_summary,
    compute_teds_metrics,
    lcs_length,
    log_metrics_summary,
)
from potatr.model import SetCriterion


logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    args: Any,
    model: nn.Module,
    criterion: SetCriterion | None,
    postprocessors: dict[str, nn.Module],
    data_loader: DataLoader,
    base_ds: TableGraphDataset,
    device: torch.device,
) -> tuple[dict[str, Any], CocoEvaluator, list[dict[str, Any]]]:
    """Run full evaluation on a dataset, computing COCO, GriTS, TEDS, and edge metrics."""
    st_time = datetime.now()
    model.eval()
    # Test-time losses are computed only when a criterion is given (the config has loss settings).
    if criterion is not None:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if criterion is not None:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    tsr_metrics = []

    rank = dist.get_rank()
    if rank == 0:
        print_freq = 100
    else:
        print_freq = 10000000000

    num_batches = len(data_loader)
    warmup_batches = min(10, num_batches // 4)
    batch_idx = 0
    total_inference_time = 0.0
    total_inference_images = 0
    total_postprocess_time = 0.0
    total_postprocess_images = 0
    total_file_io_time = 0.0
    total_target_edges_time = 0.0
    total_grits_time = 0.0
    total_teds_time = 0.0
    total_edge_metrics_time = 0.0
    total_caption_footer_time = 0.0
    total_loss_coco_time = 0.0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        for t in targets:
            for k, v in t.items():
                if not k == "img_path":
                    t[k] = v.to(device)

        past_warmup = batch_idx >= warmup_batches

        torch.cuda.synchronize()
        batch_start_time = time.perf_counter()

        outputs = model(samples)

        torch.cuda.synchronize()
        batch_inference_time = time.perf_counter() - batch_start_time
        batch_inference_time_per_image = batch_inference_time / len(targets)

        if past_warmup:
            total_inference_time += batch_inference_time
            total_inference_images += len(targets)

        batch_idx += 1

        loss_coco_start = time.perf_counter()

        if criterion is not None:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {
                k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
            }
            loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
            metric_logger.update(
                loss=sum(loss_dict_reduced_scaled.values()),
                **loss_dict_reduced_scaled,
                **loss_dict_reduced_unscaled,
            )
            metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if past_warmup:
            total_loss_coco_time += time.perf_counter() - loss_coco_start

        for target in targets:
            for k, v in target.items():
                if not k == "img_path":
                    target[k] = v.cpu()
            img_filepath = target["img_path"]

            file_io_start = time.perf_counter()

            tables_filepath = os.path.join(
                img_filepath.replace("images", "tables").replace(".jpg", ".json")
            )
            if os.path.exists(tables_filepath):
                with open(tables_filepath) as in_file:
                    tables_data = json.load(in_file)
                    target["html"] = [table["html"] for table in tables_data]
                    target["grits_top"] = [table["grid_top"] for table in tables_data]
                    target["grits_con"] = [table["grid_con"] for table in tables_data]

            img_words_filepath = img_filepath.replace("images", args.table_words_dir).replace(
                ".jpg", ".json"
            )
            if not os.path.exists(img_words_filepath):
                img_words_filepath = img_filepath.replace("images", args.table_words_dir).replace(
                    ".jpg", "_words.json"
                )
            with open(img_words_filepath) as f:
                tokens = json.load(f)
            target["tokens"] = tokens

            if past_warmup:
                total_file_io_time += time.perf_counter() - file_io_start

            target_edges_start = time.perf_counter()
            target_edges = compute_target_edges(target, base_ds.class_list)
            target["edges"] = target_edges
            if past_warmup:
                total_target_edges_time += time.perf_counter() - target_edges_start

        for target, pred_logits, pred_bboxes, pred_edge_logits in zip(
            targets,
            outputs["pred_node_logits"],
            outputs["pred_boxes"],
            outputs["pred_edge_logits"],
        ):
            true_img_size = list(reversed(target["orig_size"].tolist()))

            postprocess_start = time.perf_counter()

            pred_tables_cells = postprocess.potatr_outputs_to_tables_cells(
                pred_logits.detach().cpu(),
                pred_bboxes.detach().cpu(),
                target["tokens"],
                true_img_size,
            )

            sample_postprocess_time = time.perf_counter() - postprocess_start

            if past_warmup:
                total_postprocess_time += sample_postprocess_time
                total_postprocess_images += 1

            predicted_edges = compute_predicted_edges(
                pred_logits.detach().cpu(),
                pred_bboxes.detach().cpu(),
                pred_edge_logits.detach().cpu(),
                true_img_size,
                base_ds.class_list,
                node_score_threshold=0.8,
                edge_score_threshold=0.5,
            )

            grits_start = time.perf_counter()
            grits_metrics, grits_matched_indices = compute_grits_metrics(
                target["grits_con"], target["grits_top"], pred_tables_cells
            )
            if past_warmup:
                total_grits_time += time.perf_counter() - grits_start

            teds_start = time.perf_counter()
            teds_metrics = compute_teds_metrics(target["html"], pred_tables_cells)
            if past_warmup:
                total_teds_time += time.perf_counter() - teds_start

            edge_metrics_start = time.perf_counter()
            edge_metrics = {}
            edges_true_positives = graph_metrics.evaluate_edge_prediction(
                target["edges"], predicted_edges, iou_threshold=0.5
            )
            edge_metrics["edges@iou=0.5_num_true"] = len(target["edges"])
            edge_metrics["edges@iou=0.5_num_positive"] = len(predicted_edges)
            edge_metrics["edges@iou=0.5_true_positive_score"] = len(edges_true_positives)
            edges_true_positives = graph_metrics.evaluate_edge_prediction(
                target["edges"], predicted_edges, iou_threshold=0.8
            )
            edge_metrics["edges@iou=0.8_num_true"] = len(target["edges"])
            edge_metrics["edges@iou=0.8_num_positive"] = len(predicted_edges)
            edge_metrics["edges@iou=0.8_true_positive_score"] = len(edges_true_positives)

            edge_class_pairs = [
                (
                    "table-caption",
                    {"table", "table rotated"},
                    {"table caption", "table caption rotated"},
                ),
                (
                    "table-footer",
                    {"table", "table rotated"},
                    {"table footer", "table footer rotated"},
                ),
                (
                    "table-row",
                    {"table", "table rotated"},
                    {"table row", "table row rotated"},
                ),
                (
                    "table-column",
                    {"table", "table rotated"},
                    {"table column", "table column rotated"},
                ),
            ]
            for pair_name, src_classes, dst_classes in edge_class_pairs:
                target_subset = [
                    e
                    for e in target["edges"]
                    if e[0]["label"] in src_classes and e[1]["label"] in dst_classes
                ]
                pred_subset = [
                    e
                    for e in predicted_edges
                    if e[0]["label"] in src_classes and e[1]["label"] in dst_classes
                ]
                for iou_thresh in [0.5, 0.8]:
                    tp = graph_metrics.evaluate_edge_prediction(
                        target_subset, pred_subset, iou_threshold=iou_thresh
                    )
                    prefix = f"edges-{pair_name}@iou={iou_thresh}"
                    edge_metrics[f"{prefix}_num_true"] = len(target_subset)
                    edge_metrics[f"{prefix}_num_positive"] = len(pred_subset)
                    edge_metrics[f"{prefix}_true_positive_score"] = len(tp)

            if past_warmup:
                total_edge_metrics_time += time.perf_counter() - edge_metrics_start

            caption_footer_start = time.perf_counter()
            table_classes = {"table", "table rotated"}
            caption_classes = {"table caption", "table caption rotated"}
            footer_classes = {"table footer", "table footer rotated"}

            # GT: collect caption/footer text per table from GT edges, sorted top-to-bottom
            gt_caption_texts = []
            gt_footer_texts = []
            gt_table_bboxes = []
            for edge in target["edges"]:
                if edge[0]["label"] in table_classes:
                    if edge[0]["bbox"] not in gt_table_bboxes:
                        gt_table_bboxes.append(edge[0]["bbox"])
            for table_bbox in gt_table_bboxes:
                caption_parts = []
                footer_parts = []
                for edge in target["edges"]:
                    if edge[0]["label"] in table_classes and edge[0]["bbox"] == table_bbox:
                        if edge[1]["label"] in caption_classes:
                            text, _ = postprocess.extract_text_inside_bbox(
                                target["tokens"], edge[1]["bbox"]
                            )
                            if text.strip():
                                caption_parts.append((edge[1]["bbox"][1], text.strip()))
                        elif edge[1]["label"] in footer_classes:
                            text, _ = postprocess.extract_text_inside_bbox(
                                target["tokens"], edge[1]["bbox"]
                            )
                            if text.strip():
                                footer_parts.append((edge[1]["bbox"][1], text.strip()))
                caption_parts.sort(key=lambda x: x[0])
                footer_parts.sort(key=lambda x: x[0])
                gt_caption_texts.append(" ".join(t for _, t in caption_parts))
                gt_footer_texts.append(" ".join(t for _, t in footer_parts))

            # Predicted: associate caption/footer objects to tables
            pred_objects = postprocess.potatr_outputs_to_objects(
                pred_logits.detach().cpu(), pred_bboxes.detach().cpu(), true_img_size
            )
            pred_edge_probs = (
                pred_edge_logits.detach()
                .cpu()
                .reshape(args.num_queries, args.num_queries, -1)
                .softmax(-1)[:, :, 1:]
                .max(-1)
                .values.numpy()
            )

            pred_table_objects = [obj for obj in pred_objects if obj["label"] in table_classes]
            pred_caption_texts = []
            pred_footer_texts = []
            pred_table_bboxes = [obj["bbox"] for obj in pred_table_objects]

            for table_obj in pred_table_objects:
                caption_parts = []
                footer_parts = []
                for obj in pred_objects:
                    if obj["idx"] == table_obj["idx"]:
                        continue
                    avg_edge_prob = (
                        pred_edge_probs[table_obj["idx"], obj["idx"]]
                        + pred_edge_probs[obj["idx"], table_obj["idx"]]
                    ) / 2
                    if avg_edge_prob < 0.5:
                        continue
                    if obj["label"] in caption_classes:
                        text, _ = postprocess.extract_text_inside_bbox(
                            target["tokens"], obj["bbox"]
                        )
                        if text.strip():
                            caption_parts.append((obj["bbox"][1], text.strip()))
                    elif obj["label"] in footer_classes:
                        text, _ = postprocess.extract_text_inside_bbox(
                            target["tokens"], obj["bbox"]
                        )
                        if text.strip():
                            footer_parts.append((obj["bbox"][1], text.strip()))
                caption_parts.sort(key=lambda x: x[0])
                footer_parts.sort(key=lambda x: x[0])
                pred_caption_texts.append(" ".join(t for _, t in caption_parts))
                pred_footer_texts.append(" ".join(t for _, t in footer_parts))

            # Compute caption/footer LCS metrics
            caption_footer_metrics = {}
            caption_num_true = sum(len(t) for t in gt_caption_texts)
            caption_num_positive = sum(len(t) for t in pred_caption_texts)
            footer_num_true = sum(len(t) for t in gt_footer_texts)
            footer_num_positive = sum(len(t) for t in pred_footer_texts)

            caption_footer_metrics["caption_num_true"] = caption_num_true
            caption_footer_metrics["caption_num_positive"] = caption_num_positive
            caption_footer_metrics["footer_num_true"] = footer_num_true
            caption_footer_metrics["footer_num_positive"] = footer_num_positive

            # IoU matching
            caption_lcs_tp_iou = 0
            footer_lcs_tp_iou = 0
            if gt_table_bboxes and pred_table_bboxes:
                iou_matrix = np.zeros((len(gt_table_bboxes), len(pred_table_bboxes)))
                for i, gt_bbox in enumerate(gt_table_bboxes):
                    for j, pred_bbox in enumerate(pred_table_bboxes):
                        iou_matrix[i, j] = compute_iou(gt_bbox, pred_bbox)
                row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
                for gt_idx, pred_idx in zip(row_ind, col_ind):
                    caption_lcs_tp_iou += lcs_length(
                        gt_caption_texts[gt_idx], pred_caption_texts[pred_idx]
                    )
                    footer_lcs_tp_iou += lcs_length(
                        gt_footer_texts[gt_idx], pred_footer_texts[pred_idx]
                    )
            caption_footer_metrics["caption_lcs_tp_iou"] = caption_lcs_tp_iou
            caption_footer_metrics["footer_lcs_tp_iou"] = footer_lcs_tp_iou

            # GriTS_Con matching
            caption_lcs_tp_con = 0
            footer_lcs_tp_con = 0
            con_true_indices, con_pred_indices = grits_matched_indices["con"]
            for gt_idx, pred_idx in zip(con_true_indices, con_pred_indices):
                if gt_idx < len(gt_caption_texts) and pred_idx < len(pred_caption_texts):
                    caption_lcs_tp_con += lcs_length(
                        gt_caption_texts[gt_idx], pred_caption_texts[pred_idx]
                    )
                if gt_idx < len(gt_footer_texts) and pred_idx < len(pred_footer_texts):
                    footer_lcs_tp_con += lcs_length(
                        gt_footer_texts[gt_idx], pred_footer_texts[pred_idx]
                    )
            caption_footer_metrics["caption_lcs_tp_con"] = caption_lcs_tp_con
            caption_footer_metrics["footer_lcs_tp_con"] = footer_lcs_tp_con

            # GriTS_Top matching
            caption_lcs_tp_top = 0
            footer_lcs_tp_top = 0
            top_true_indices, top_pred_indices = grits_matched_indices["top"]
            for gt_idx, pred_idx in zip(top_true_indices, top_pred_indices):
                if gt_idx < len(gt_caption_texts) and pred_idx < len(pred_caption_texts):
                    caption_lcs_tp_top += lcs_length(
                        gt_caption_texts[gt_idx], pred_caption_texts[pred_idx]
                    )
                if gt_idx < len(gt_footer_texts) and pred_idx < len(pred_footer_texts):
                    footer_lcs_tp_top += lcs_length(
                        gt_footer_texts[gt_idx], pred_footer_texts[pred_idx]
                    )
            caption_footer_metrics["caption_lcs_tp_top"] = caption_lcs_tp_top
            caption_footer_metrics["footer_lcs_tp_top"] = footer_lcs_tp_top

            if past_warmup:
                total_caption_footer_time += time.perf_counter() - caption_footer_start

            metrics = grits_metrics
            metrics.update(teds_metrics)
            metrics.update(edge_metrics)
            metrics.update(caption_footer_metrics)
            metrics["num_tables"] = len(target["grits_con"])

            metrics["id"] = target["img_path"].split("/")[-1].split(".")[0]
            metrics["inference_ms"] = batch_inference_time_per_image * 1000
            metrics["postprocess_ms"] = sample_postprocess_time * 1000
            metrics["exclude_warmup"] = past_warmup
            tsr_metrics.append(metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

    # Inference timing (per-GPU, excluding warm-up)
    if total_inference_images > 0:
        ms_per_image = (total_inference_time / total_inference_images) * 1000
        batch_size = data_loader.batch_size
        stats["inference_ms_per_image"] = ms_per_image
        stats["inference_batch_size"] = batch_size
        stats["inference_total_seconds"] = total_inference_time
        stats["inference_num_images"] = total_inference_images

        if total_postprocess_images > 0:
            postprocess_ms_per_image = (total_postprocess_time / total_postprocess_images) * 1000
            stats["postprocess_ms_per_image"] = postprocess_ms_per_image
            stats["postprocess_total_seconds"] = total_postprocess_time
            stats["postprocess_num_images"] = total_postprocess_images

    world_size = dist.get_world_size()
    gathered_tsr_metrics: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_tsr_metrics, tsr_metrics)
    tsr_metrics = list(itertools.chain(*gathered_tsr_metrics))

    rank = dist.get_rank()
    if rank == 0:
        metrics_summary = compute_metrics_summary(tsr_metrics)
        log_metrics_summary(metrics_summary)

        if total_inference_images > 0:
            PAD = 40
            batch_size = data_loader.batch_size
            ms_per_image = (total_inference_time / total_inference_images) * 1000
            logger.info("")
            logger.info("%s", "Inference Timing".rjust(PAD))
            logger.info("%s: %d", "Batch size".rjust(PAD), batch_size)
            logger.info("%s: %d", "Images timed".rjust(PAD), total_inference_images)
            logger.info("%s: %.2f ms/image", "Forward pass".rjust(PAD), ms_per_image)
            if total_postprocess_images > 0:
                postprocess_ms_per_image = (
                    total_postprocess_time / total_postprocess_images
                ) * 1000
                combined_ms = ms_per_image + postprocess_ms_per_image
                logger.info("%s: %.2f ms/image", "Postprocess".rjust(PAD), postprocess_ms_per_image)
                logger.info("%s: %.2f ms/image", "Combined".rjust(PAD), combined_ms)
                logger.info("%s: %.1f images/s", "Throughput".rjust(PAD), 1000 / combined_ms)

    logger.info("Total time taken for %d samples: %s", len(base_ds), datetime.now() - st_time)

    return stats, coco_evaluator, tsr_metrics

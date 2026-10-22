# Copyright 2025-present Kensho Technologies, LLC.
r"""Run POTATR on page images: find each table, its cells, and its caption and footer text.

Example:
    potatr-infer \
        --config_file configs/model/potatr_v1.json \
        --model_load_path <weights>.pth \
        --images_dir pages/images \
        --words_dir pages/words \
        --output_dir output/tables
"""

import argparse
from collections.abc import Sequence
import json
import logging
import os
import time
from typing import Any

from PIL import Image
import torch
from torch import nn

from potatr import postprocess
from potatr.config import load_config_files
from potatr.datasets import get_transform
from potatr.metrics import cells_to_html
from potatr.model import build, load_checkpoint, load_model_weights


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TABLE_CLASSES = {"table", "table rotated"}
CAPTION_CLASSES = {"table caption", "table caption rotated"}
FOOTER_CLASSES = {"table footer", "table footer rotated"}


def load_model(config_files: list[str], weights_path: str, device: str) -> tuple[nn.Module, Any]:
    """Build the model from its config files and load its weights, for inference.

    Every model tensor must be in the checkpoint. Returns (model, args), where args holds the
    merged config settings.
    """
    config = load_config_files(config_files)
    config["device"] = device
    # Training settings that building the model reads. They have no effect in eval mode.
    config.setdefault("dropout", 0.0)
    config.setdefault("lr_backbone", 0.0)
    config.setdefault("aux_loss", False)
    args = type("Args", (object,), config)

    model, _, _ = build(args)
    model.to(torch.device(device))
    checkpoint = load_checkpoint(weights_path, map_location=device)
    load_model_weights(model, checkpoint, require_all=True)
    model.eval()
    return model, args


def linked_text_parts(
    table: postprocess.Object,
    objects: list[postprocess.Object],
    edge_probs: Any,
    tokens: list[postprocess.Object],
    classes: set[str],
    edge_threshold: float,
) -> list[dict[str, Any]]:
    """Return the text of the objects of the given classes that are linked to a table.

    An object is linked to the table if the mean of the two edge probabilities between them is at
    least edge_threshold. Objects without text are skipped, and the rest are sorted top to bottom.
    """
    parts = []
    for obj in objects:
        if obj["label"] not in classes:
            continue
        edge_prob = (
            edge_probs[table["idx"], obj["idx"]] + edge_probs[obj["idx"], table["idx"]]
        ) / 2
        if edge_prob < edge_threshold:
            continue
        text, _ = postprocess.extract_text_inside_bbox(tokens, obj["bbox"])
        if text.strip():
            parts.append(
                {
                    "text": text.strip(),
                    "bbox": obj["bbox"],
                    "score": obj["score"],
                    "edge_score": float(edge_prob),
                }
            )
    parts.sort(key=lambda part: part["bbox"][1])
    return parts


def tables_from_outputs(
    outputs: dict[str, Any],
    img_size: Sequence[int],
    tokens: list[postprocess.Object],
    class_names: list[str],
    edge_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Turn the model's outputs for one page into the page's tables.

    outputs are the outputs for a batch of one page, img_size is the page's (width, height), and
    tokens are the page's words. Each table has its HTML, its cells (without their words), its
    caption and footer text with the objects they came from, and its box and score.
    """
    pred_logits = outputs["pred_node_logits"][0].detach().cpu()
    pred_boxes = outputs["pred_boxes"][0].detach().cpu()
    pred_edge_logits = outputs["pred_edge_logits"][0].detach().cpu()

    objects = postprocess.potatr_outputs_to_objects(pred_logits, pred_boxes, img_size, class_names)
    # For each pair of queries, the probability of the most likely edge class (class 0 is no edge).
    edge_probs = pred_edge_logits.softmax(-1)[:, :, 1:].max(-1).values.numpy()
    tables_cells = postprocess.potatr_outputs_to_tables_cells(
        pred_logits, pred_boxes, tokens, img_size, class_names
    )
    # potatr_outputs_to_tables_cells handles the tables in the same order as objects.
    table_objects = [obj for obj in objects if obj["label"] in TABLE_CLASSES]
    assert len(table_objects) == len(tables_cells)

    tables = []
    for table, cells in zip(table_objects, tables_cells):
        caption_parts = linked_text_parts(
            table, objects, edge_probs, tokens, CAPTION_CLASSES, edge_threshold
        )
        footer_parts = linked_text_parts(
            table, objects, edge_probs, tokens, FOOTER_CLASSES, edge_threshold
        )
        tables.append(
            {
                "html": cells_to_html(cells) if cells else "",
                "cells": [{k: v for k, v in cell.items() if k != "spans"} for cell in cells],
                "caption": " ".join(part["text"] for part in caption_parts),
                "footer": " ".join(part["text"] for part in footer_parts),
                "caption_parts": caption_parts,
                "footer_parts": footer_parts,
                "label": table["label"],
                "bbox": table["bbox"],
                "score": table["score"],
            }
        )
    return tables


def run_on_page(
    model: nn.Module,
    args: Any,
    image: Image.Image,
    tokens: list[postprocess.Object],
    edge_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run POTATR on one page image with its words. Returns (tables, timing in milliseconds).

    The image is preprocessed as for evaluation.
    """
    image = image.convert("RGB")
    img_tensor, _ = get_transform("eval")(image, {"boxes": torch.empty((0, 4))})
    device = torch.device(args.device)
    on_gpu = device.type == "cuda"

    if on_gpu:
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model([img_tensor.to(device)])
    if on_gpu:
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start

    start = time.perf_counter()
    tables = tables_from_outputs(outputs, image.size, tokens, args.classes, edge_threshold)
    postprocess_seconds = time.perf_counter() - start

    timing = {
        "inference_ms": inference_seconds * 1000,
        "postprocess_ms": postprocess_seconds * 1000,
    }
    return tables, timing


def process_directory(
    model: nn.Module,
    args: Any,
    images_dir: str,
    words_dir: str,
    output_dir: str,
    edge_threshold: float = 0.5,
) -> None:
    """Run POTATR on every page image in a directory, and write one JSON file of tables per page.

    A page's words are read from <words_dir>/<name>.json (or <name>_words.json). Pages without a
    words file are skipped. The time taken per page is written to timing.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(
        f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    logger.info("Found %d images in %s", len(image_files), images_dir)

    all_timing = {}
    for i, filename in enumerate(image_files, start=1):
        stem = os.path.splitext(filename)[0]
        words_path = os.path.join(words_dir, f"{stem}.json")
        if not os.path.exists(words_path):
            words_path = os.path.join(words_dir, f"{stem}_words.json")
        if not os.path.exists(words_path):
            logger.warning("Skipping %s: no words file in %s", filename, words_dir)
            continue
        with open(words_path) as f:
            tokens = json.load(f)

        with Image.open(os.path.join(images_dir, filename)) as image:
            tables, timing = run_on_page(model, args, image, tokens, edge_threshold)
        all_timing[stem] = timing
        with open(os.path.join(output_dir, f"{stem}.json"), "w") as f:
            json.dump(tables, f, indent=2)
        logger.info(
            "[%d/%d] %s: %d table(s), %.1f ms",
            i,
            len(image_files),
            filename,
            len(tables),
            timing["inference_ms"] + timing["postprocess_ms"],
        )

    with open(os.path.join(output_dir, "timing.json"), "w") as f:
        json.dump(all_timing, f, indent=2)
    logger.info("Results written to %s", output_dir)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run POTATR table extraction on page images")
    parser.add_argument("--config_file", required=True, nargs="+", help="The model config file(s)")
    parser.add_argument("--model_load_path", required=True, help="The model weights")
    parser.add_argument("--images_dir", required=True, help="Directory of page images")
    parser.add_argument("--words_dir", required=True, help="Directory of the pages' words (JSON)")
    parser.add_argument("--output_dir", required=True, help="Directory to write the tables to")
    parser.add_argument(
        "--device", default="cuda", help="Device to run on, for example cuda or cpu"
    )
    parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.5,
        help="Edge probability needed to link a caption or footer to a table",
    )
    return parser.parse_args()


def cli_main() -> None:
    """CLI entry point for inference."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cli_args = get_args()
    model, args = load_model(cli_args.config_file, cli_args.model_load_path, cli_args.device)
    logger.info("Model loaded (%d parameters)", sum(p.numel() for p in model.parameters()))
    process_directory(
        model,
        args,
        cli_args.images_dir,
        cli_args.words_dir,
        cli_args.output_dir,
        cli_args.edge_threshold,
    )


if __name__ == "__main__":
    cli_main()

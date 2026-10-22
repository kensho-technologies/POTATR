# Copyright 2025-present Kensho Technologies, LLC.
# Portions copyright (C) 2021 Microsoft Corporation, from Table Transformer
# (https://github.com/microsoft/table-transformer), MIT License.
# Third-party portions have been modified. See THIRD_PARTY_NOTICES.md.
"""POTATR training entry point with Distributed Data Parallel (DDP) support."""

import argparse
from datetime import datetime
import io
import json
import logging
import os
import random
import sys
from typing import Any

from grits import compute_fscore
import numpy as np
from safetensors.torch import save_file
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from potatr import build as build_model
from potatr.config import has_loss_settings, load_config_files, unused_config_keys
from potatr.datasets import TableGraphDataset, get_transform
from potatr.detr.engine import evaluate, train_one_epoch
import potatr.detr.util.misc as utils
from potatr.evaluate import evaluate as ev
from potatr.metrics import compute_metrics_summary
from potatr.model import POTATR, SetCriterion, load_checkpoint, load_model_weights


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def ddp_setup(rank: int, world_size: int, port: int = 12355) -> None:
    """Set up DDP process group.

    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        port: Port for inter-process communication
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        nargs="+",
        help="One or more config files, merged in order; a setting may appear in only one of them",
    )
    parser.add_argument("--model_load_path", help="The path to trained model")
    parser.add_argument("--load_weights_only", action="store_true")
    parser.add_argument(
        "--model_save_dir", help="The output directory for saving model params and checkpoints"
    )
    parser.add_argument(
        "--table_words_dir",
        help="Folder containing the bboxes of table words",
        default="words",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Modes: training (train) and evaluation (test)",
    )
    parser.add_argument("--device")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_drop", type=int)
    parser.add_argument("--lr_gamma", type=float)
    parser.add_argument("--adj_loss_coef", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--start_epoch", type=int)
    parser.add_argument(
        "--checkpoint_freq", type=int, help="Save a numbered checkpoint every N epochs (default: 1)"
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--train_max_size", type=int)
    parser.add_argument("--val_max_size", type=int)
    parser.add_argument("--test_max_size", type=int)
    parser.add_argument("--max_batches_per_epoch", type=int)
    parser.add_argument("--train_all_weights", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--port", type=int, default=10101)

    return parser.parse_args()


def get_class_map(args: Any) -> dict[str, int]:
    """Build class name to index mapping."""
    classes = args.classes
    class_map = {k: v for v, k in enumerate(classes)}
    return class_map


def get_max_size(dataset: dict[str, Any], key: str, override: int | None) -> int | None:
    """Return a dataset's size limit for one split (train_max_size, val_max_size or test_max_size).

    Missing or null means the full split, 0 means skip the split, and N > 0 means a seeded random
    subset of N images. A command-line value overrides the config for every dataset, but a split
    that the config skips (0) stays skipped, since some collections don't have every split.
    """
    max_size = dataset.get(key)
    if override is not None and max_size != 0:
        max_size = override
    if max_size is not None and max_size < 0:
        raise ValueError(f"{key} must be non-negative, got {max_size}")
    return max_size


def default_output_directory() -> str:
    """Return a new output/<timestamp> directory path, the same on every DDP rank.

    Every rank must call this, since rank 0 broadcasts its timestamp to the others.
    """
    run_date = [datetime.now().strftime("%Y%m%d%H%M%S")]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(run_date, src=0)
    return os.path.join("output", run_date[0])


def get_train_data(args: Any) -> tuple[list[dict[str, Any]], list[TableGraphDataset], int]:
    """Load training and validation datasets."""
    logger.info("Loading data")

    class_map = get_class_map(args)

    datasets_val = []

    logger.info("%d datasets.", len(args.datasets))
    datasets = args.datasets
    len_per_epoch = 0
    for dataset in datasets:
        dataset_train = TableGraphDataset(
            os.path.join(dataset["data_root_dir"], "train"),
            image_extension=dataset.get("img_extension", ".jpg"),
            class_map=class_map,
            transforms=get_transform(dataset["train_transform"]),
            page_list=dataset.get("train_page_list"),
        )
        dataset["data"] = dataset_train
        # The training loop takes the first train_max_size samples of each epoch's shuffle,
        # so a missing or null value (stored as None) uses every sample.
        train_max_size = get_max_size(dataset, "train_max_size", args.train_max_size)
        dataset["train_max_size"] = train_max_size
        if train_max_size is None:
            len_per_epoch += len(dataset_train)
        else:
            len_per_epoch += min(train_max_size, len(dataset_train))

        val_max_size = get_max_size(dataset, "val_max_size", args.val_max_size)
        if val_max_size == 0:
            logger.info("Skipping val split of %s (val_max_size is 0)", dataset["data_root_dir"])
            continue
        dataset_val = TableGraphDataset(
            os.path.join(dataset["data_root_dir"], "val"),
            image_extension=dataset.get("img_extension", ".jpg"),
            class_map=class_map,
            transforms=get_transform(dataset["val_transform"]),
            make_coco=True,
            max_size=val_max_size,
            page_list=dataset.get("val_page_list"),
        )
        datasets_val.append(dataset_val)

    if len_per_epoch == 0:
        raise ValueError("No training data: train_max_size is 0 for every dataset")

    return datasets, datasets_val, len_per_epoch


def get_eval_data(args: Any) -> tuple[list[DataLoader], list[TableGraphDataset]]:
    """Return test data loaders and datasets for evaluation."""
    class_map = get_class_map(args)

    datasets = args.datasets

    datasets_test = []
    data_loaders_test = []

    for dataset in datasets:
        test_max_size = get_max_size(dataset, "test_max_size", args.test_max_size)
        if test_max_size == 0:
            logger.info("Skipping test split of %s (test_max_size is 0)", dataset["data_root_dir"])
            continue

        logger.info(
            "Test dataset with %s transform, %s",
            dataset["val_transform"],
            "full split" if test_max_size is None else f"up to {test_max_size} images",
        )

        dataset_test = TableGraphDataset(
            os.path.join(dataset["data_root_dir"], "test"),
            image_extension=dataset.get("img_extension", ".jpg"),
            class_map=class_map,
            transforms=get_transform(dataset["val_transform"]),
            make_coco=True,
            include_eval=True,
            max_size=test_max_size,
            page_list=dataset.get("test_page_list"),
        )
        dataset_test.name = dataset.get("name")
        datasets_test.append(dataset_test)

        sampler_test: DistributedSampler[Any] = DistributedSampler(dataset_test)
        data_loader_test = DataLoader(
            dataset_test,
            sampler=sampler_test,
            collate_fn=utils.collate_fn,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        sampler_test.set_epoch(0)
        data_loaders_test.append(data_loader_test)

    return data_loaders_test, datasets_test


def get_model(
    args: Any, device: torch.device
) -> tuple[POTATR, SetCriterion | None, dict[str, nn.Module]]:
    """Load POTATR model on to the device specified.

    If a load path is specified, the model weights are loaded from it. In test mode, every model
    tensor must be in the checkpoint.
    """
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.model_load_path:
        logger.info("Loading model from checkpoint %s", args.model_load_path)
        checkpoint = load_checkpoint(args.model_load_path, map_location=device)
        load_model_weights(model, checkpoint, require_all=args.mode == "test")

    if args.train_all_weights:
        logger.info("Training all weights")
        for param in model.parameters():
            param.requires_grad = True

    return model, criterion, postprocessors


def train(
    args: Any,
    model: DDP,
    criterion: SetCriterion,
    postprocessors: dict[str, nn.Module],
    device: torch.device,
) -> None:
    """Run the training loop."""
    logger.info("Loading data")
    dataloading_time = datetime.now()
    dataset_dicts_train, datasets_val, train_len = get_train_data(args)
    logger.info("Finished loading data in: %s", datetime.now() - dataloading_time)

    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_drop, gamma=args.lr_gamma
    )

    max_batches_per_epoch = int(train_len / args.batch_size)
    if args.max_batches_per_epoch > 0 and args.max_batches_per_epoch < max_batches_per_epoch:
        max_batches_per_epoch = args.max_batches_per_epoch
    logger.info("Max batches per epoch: %d", max_batches_per_epoch)

    resume_checkpoint = False
    if args.model_load_path:
        # The model weights were already loaded by get_model. Load the checkpoint again here only
        # for the optimizer, learning rate scheduler and epoch of a full training checkpoint.
        checkpoint = load_checkpoint(args.model_load_path)
        if not args.load_weights_only:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Loaded optimizer state")
                resume_checkpoint = True
            if "lr_scheduler_state_dict" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                logger.info("Loaded learning rate scheduler state")
        elif args.load_weights_only:
            logger.warning(
                "Resuming training and ignoring optimizer state. "
                "Training will resume with new initialized values. "
                "To use current optimizer state, remove the --load_weights_only flag."
            )

        if not args.load_weights_only and "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1
        elif args.load_weights_only:
            logger.warning(
                "Resuming training and ignoring previously saved epoch. "
                "To resume from previously saved epoch, remove the --load_weights_only flag."
            )
        else:
            logger.warning(
                "Epoch of saved model not found. Starting at epoch %d.", args.start_epoch
            )

    # Use user-specified save directory, if specified
    if args.model_save_dir:
        output_directory = args.model_save_dir
    # If resuming from a checkpoint with optimizer state, save into same directory
    elif args.model_load_path and resume_checkpoint:
        output_directory = os.path.split(args.model_load_path)[0]
    # Create new save directory
    else:
        output_directory = default_output_directory()

    # Every rank runs this, so the directory may already exist by the time a rank creates it.
    os.makedirs(output_directory, exist_ok=True)
    logger.info("Output directory: %s", output_directory)
    model_save_path = os.path.join(output_directory, "model.pth")
    logger.info("Output model path: %s", model_save_path)
    if not resume_checkpoint and os.path.exists(model_save_path):
        logger.warning(
            "Output model path exists but is not being used to resume training; "
            "training will overwrite it."
        )

    if args.start_epoch >= args.epochs:
        logger.warning(
            "Starting epoch (%d) is greater or equal to the number of training epochs (%d).",
            args.start_epoch,
            args.epochs,
        )

    logger.info("Start training")
    start_time = datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        logger.info("-" * 100)

        datasets_train = []
        for elem in dataset_dicts_train:
            dataset_train = elem["data"]
            sample_nums = list(range(len(dataset_train)))
            random.shuffle(sample_nums)
            sample_nums = sample_nums[: elem["train_max_size"]]
            dataset_train = Subset(dataset_train, sample_nums)
            datasets_train.append(dataset_train)

        dataset_train = ConcatDataset(datasets_train)

        sampler_train: DistributedSampler[Any] = DistributedSampler(dataset_train)
        data_loader_train = DataLoader(
            dataset_train,
            sampler=sampler_train,
            collate_fn=utils.collate_fn,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        sampler_train.set_epoch(epoch)

        epoch_timing = datetime.now()

        rank = dist.get_rank()
        if rank == 0:
            print_freq = 100
        else:
            print_freq = 10000000000

        train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            max_batches_per_epoch=max_batches_per_epoch,
            print_freq=print_freq,
        )
        logger.info("Epoch completed in %s", datetime.now() - epoch_timing)

        lr_scheduler.step()

        if rank == 0:
            # Save current model training progress
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                },
                model_save_path,
            )

            # Save checkpoint for evaluation
            if (epoch + 1) % args.checkpoint_freq == 0:
                model_save_path_epoch = os.path.join(
                    output_directory, "model_" + str(epoch + 1) + ".pth"
                )
                torch.save(model.module.state_dict(), model_save_path_epoch)
                logger.info("Saved checkpoint to %s", model_save_path_epoch)

            # Save HF model to safetensors
            safetensors_save_path = os.path.join(output_directory, "model.safetensors")
            model_state_dict = model.module.state_dict()
            model_state_dict = {k: v.contiguous() for k, v in model_state_dict.items()}
            save_file(model_state_dict, safetensors_save_path, metadata={"format": "pt"})

        for dataset_val in datasets_val:
            sampler_val: DistributedSampler[Any] = DistributedSampler(dataset_val)
            data_loader_val = DataLoader(
                dataset_val,
                sampler=sampler_val,
                collate_fn=utils.collate_fn,
                shuffle=False,
                batch_size=1,
                num_workers=args.num_workers,
            )
            sampler_val.set_epoch(epoch)

            stats, coco_evaluator = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                dataset_val,
                device,
                None,
                print_freq=print_freq,
            )

            if rank == 0:
                logger.info(
                    "val: AP50: %.3f, AP75: %.3f, AP: %.3f, AR: %.3f",
                    stats["coco_eval_bbox"][1],
                    stats["coco_eval_bbox"][2],
                    stats["coco_eval_bbox"][0],
                    stats["coco_eval_bbox"][8],
                )

    logger.info("Total training time: %s", datetime.now() - start_time)


def main(rank: int, world_size: int) -> None:
    """Main function run by each DDP process."""
    # DETR's progress lines are printed; write them line by line even when output goes to a file.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    cmd_args = get_args().__dict__
    config_args = load_config_files(cmd_args["config_file"])
    unused = unused_config_keys(config_args)
    if unused and rank == 0:
        logger.warning(
            "Ignoring %d config settings that POTATR doesn't use: %s",
            len(unused),
            ", ".join(unused),
        )
    # A command-line value overrides the config only when it's given, so options that can also
    # be set in the config have no argparse defaults; their defaults are applied here, after
    # merging. Options with argparse defaults (--mode, --port, --table_words_dir and the
    # store_true flags) are command-line only and always take their command-line value.
    for key, value in cmd_args.items():
        if key not in config_args or value is not None:
            config_args[key] = value
    if config_args.get("checkpoint_freq") is None:
        config_args["checkpoint_freq"] = 1
    if config_args.get("aux_loss") is None:
        config_args["aux_loss"] = False
    if config_args["mode"] == "train" and not has_loss_settings(config_args):
        raise ValueError(
            "Training needs the loss settings, for example configs/loss/potatr_v1.json"
        )
    if config_args["mode"] == "test":
        # Training settings that building the model still reads. They have no effect on a model
        # in eval mode, so evaluation configs can leave them out.
        config_args.setdefault("dropout", 0.0)
        config_args.setdefault("lr_backbone", 0.0)

    args: Any = type("Args", (object,), config_args)
    logger.info("Config: %s", args.__dict__)
    logger.info("-" * 100)

    ddp_setup(rank, world_size, args.port)

    # fix the seed for reproducibility
    logger.info("Using random seed: %d", args.seed)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger.info("Loading model")
    device = torch.device(args.device)
    model, criterion, postprocessors = get_model(args, device)

    ddp_model = DDP(model, find_unused_parameters=True)

    if args.mode == "train":
        # main checked above that the loss settings are given, so the criterion was built.
        assert criterion is not None
        train(args, ddp_model, criterion, postprocessors, device)
    elif args.mode == "test":
        rank = dist.get_rank()
        data_loaders_test, datasets_test = get_eval_data(args)

        # Metrics go to --model_save_dir if given, otherwise to a new output/<timestamp>
        # directory, never next to the checkpoint, where they could overwrite earlier results.
        output_dir = args.model_save_dir or default_output_directory()
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Output directory: %s", output_dir)

        for data_loader_test, dataset_test in zip(data_loaders_test, datasets_test):
            if rank == 0:
                logger.info("Evaluating %s", dataset_test.root)

            stats, coco_evaluator, tsr_metrics = ev(
                args, ddp_model, criterion, postprocessors, data_loader_test, dataset_test, device
            )

            if rank == 0:
                PAD = 40
                logger.info("")
                logger.info("%s: %.4f", "AP".rjust(PAD), stats["coco_eval_bbox"][0])
                logger.info("%s: %.4f", "AP50".rjust(PAD), stats["coco_eval_bbox"][1])
                logger.info("%s: %.4f", "AP75".rjust(PAD), stats["coco_eval_bbox"][2])
                logger.info("%s: %.4f", "AR".rjust(PAD), stats["coco_eval_bbox"][8])

                dataset_name = dataset_test.name or os.path.basename(
                    os.path.normpath(dataset_test.root)
                )

                metric_names = [
                    "grits_top",
                    "grits_con",
                    "teds",
                    "teds_struct",
                    "edges@iou=0.5",
                    "edges@iou=0.8",
                    "edges-table-caption@iou=0.5",
                    "edges-table-caption@iou=0.8",
                    "edges-table-footer@iou=0.5",
                    "edges-table-footer@iou=0.8",
                    "edges-table-row@iou=0.5",
                    "edges-table-row@iou=0.8",
                    "edges-table-column@iou=0.5",
                    "edges-table-column@iou=0.8",
                ]
                dataset_metrics = {}
                for metric_name in metric_names:
                    total_tp = sum(m[metric_name + "_true_positive_score"] for m in tsr_metrics)
                    total_true = sum(m[metric_name + "_num_true"] for m in tsr_metrics)
                    total_positive = sum(m[metric_name + "_num_positive"] for m in tsr_metrics)
                    f1, precision, recall = compute_fscore(total_tp, total_true, total_positive)
                    dataset_metrics[metric_name + "_f1"] = f1
                    dataset_metrics[metric_name + "_precision"] = precision
                    dataset_metrics[metric_name + "_recall"] = recall

                # Caption and footer text F1, under each way of matching predicted tables to
                # true tables, as in the logged summary.
                summary = compute_metrics_summary(tsr_metrics)
                for text_type in ("caption", "footer"):
                    for matching in ("iou", "con", "top"):
                        for measure in ("f1", "precision", "recall"):
                            key = f"{text_type}_text_{measure}@{matching}"
                            if key in summary:
                                dataset_metrics[key] = summary[key]

                dataset_metrics["AP"] = stats["coco_eval_bbox"][0]
                dataset_metrics["AP50"] = stats["coco_eval_bbox"][1]
                dataset_metrics["AP75"] = stats["coco_eval_bbox"][2]
                dataset_metrics["AR"] = stats["coco_eval_bbox"][8]

                output = {
                    "model_load_path": args.model_load_path,
                    "dataset_metrics": dataset_metrics,
                }
                if criterion is not None:
                    # Test-time losses, averaged over all images, as logged during evaluation.
                    output["losses"] = {
                        k: v
                        for k, v in stats.items()
                        if k == "loss"
                        or k.startswith("loss_")
                        or k.endswith("_unscaled")
                        or k == "class_error"
                    }
                output["sample_metrics"] = tsr_metrics
                if "inference_ms_per_image" in stats:
                    output["inference_timing"] = {
                        "batch_size": stats["inference_batch_size"],
                        "num_images": stats["inference_num_images"],
                        "forward_ms_per_image": stats["inference_ms_per_image"],
                        "forward_total_seconds": stats["inference_total_seconds"],
                    }
                    if "postprocess_ms_per_image" in stats:
                        output["inference_timing"]["postprocess_ms_per_image"] = stats[
                            "postprocess_ms_per_image"
                        ]
                        output["inference_timing"]["postprocess_total_seconds"] = stats[
                            "postprocess_total_seconds"
                        ]
                        output["inference_timing"]["combined_ms_per_image"] = (
                            stats["inference_ms_per_image"] + stats["postprocess_ms_per_image"]
                        )
                metrics_path = os.path.join(output_dir, f"metrics_{dataset_name}.json")
                with open(metrics_path, "w") as f:
                    json.dump(
                        output,
                        f,
                        default=lambda o: int(o) if isinstance(o, np.integer) else float(o),
                    )
                logger.info("Metrics saved to %s", metrics_path)

    destroy_process_group()


def cli_main() -> None:
    """CLI entry point for training."""
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise SystemExit("POTATR training and evaluation need at least one GPU; none found")
    mp.spawn(main, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    cli_main()

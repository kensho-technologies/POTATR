# Copyright 2025-present Kensho Technologies, LLC.
"""Config files: the settings POTATR reads, which file each belongs in, and merging."""

import json
from typing import Any


# Settings by the config file they belong in. Training uses model + loss + train configs, and
# evaluation uses model + eval configs, plus the loss config to also compute test-time losses.
MODEL_KEYS = {
    "backbone",
    "dilation",
    "masks",
    "position_embedding",
    "hidden_dim",
    "enc_layers",
    "dec_layers",
    "nheads",
    "dim_feedforward",
    "pre_norm",
    "num_queries",
    "classes",
    "num_node_classes",
    "num_edge_classes",
    "node_embed_hidden_layers",
    "node_embed_hidden_dim",
    "edge_embed_hidden_layers",
    "edge_embed_hidden_dim",
    "bbox_embed_hidden_layers",
    "bbox_embed_hidden_dim",
    "pre_adj_dim",
}
LOSS_KEYS = {
    "aux_loss",
    "ce_loss_coef",
    "bbox_loss_coef",
    "giou_loss_coef",
    "adj_loss_coef",
    "eos_coef",
    "adj_class_weights",
    "set_cost_class",
    "set_cost_bbox",
    "set_cost_giou",
}
# Settings used both for training and for evaluation, so they appear in the train and eval configs.
RUN_KEYS = {"device", "seed", "batch_size", "num_workers", "datasets"}
TRAIN_KEYS = RUN_KEYS | {
    "lr",
    "lr_backbone",
    "weight_decay",
    "dropout",
    "epochs",
    "start_epoch",
    "lr_drop",
    "lr_gamma",
    "clip_max_norm",
    "max_batches_per_epoch",
    "checkpoint_freq",
}
EVAL_KEYS = RUN_KEYS
# Settings of each entry in "datasets".
DATASET_KEYS = {
    "data_root_dir",
    "name",
    "img_extension",
    "train_transform",
    "val_transform",
    "train_max_size",
    "val_max_size",
    "test_max_size",
    "train_page_list",
    "val_page_list",
    "test_page_list",
}
KNOWN_KEYS = MODEL_KEYS | LOSS_KEYS | TRAIN_KEYS | EVAL_KEYS


def load_config_files(paths: list[str]) -> dict[str, Any]:
    """Load and merge JSON config files, in order.

    Each setting may appear in only one file, so that combining files (for example a model,
    loss and training config) can't silently override a value.
    """
    config: dict[str, Any] = {}
    source: dict[str, str] = {}
    for path in paths:
        with open(path, "rb") as config_file:
            file_config = json.load(config_file)
        repeated = sorted(key for key in file_config if key in config)
        if repeated:
            details = ", ".join(f"{key} (also in {source[key]})" for key in repeated)
            raise ValueError(f"Config file {path} repeats settings from an earlier file: {details}")
        config.update(file_config)
        source.update({key: path for key in file_config})
    return config


def has_loss_settings(config: dict[str, Any]) -> bool:
    """Return whether a config has the loss settings, so losses can be computed.

    Training needs them. For evaluation they're optional: with them, test-time losses are
    computed too. A partial set is an error, so losses can't be switched on or off by accident.
    aux_loss is optional and defaults to False. A setting whose value is None counts as absent,
    since every command-line option is merged in, with None when it wasn't given.
    """
    required = LOSS_KEYS - {"aux_loss"}
    present = {key for key in required if config.get(key) is not None}
    if not present:
        return False
    if present != required:
        missing = ", ".join(sorted(required - present))
        raise ValueError(
            f"Incomplete loss settings, missing: {missing}. Give all of them (for example with "
            "configs/loss/potatr_v1.json) or none."
        )
    return True


def unused_config_keys(config: dict[str, Any]) -> list[str]:
    """Return the settings in a config that POTATR doesn't read, e.g. from older versions.

    Settings of dataset entries are reported as "datasets[i].key".
    """
    unused = sorted(key for key in config if key not in KNOWN_KEYS)
    for i, dataset in enumerate(config.get("datasets", [])):
        unused += sorted(f"datasets[{i}].{key}" for key in dataset if key not in DATASET_KEYS)
    return unused

# Copyright 2025-present Kensho Technologies, LLC.
"""Tests for loading and merging config files."""

import json
from pathlib import Path

import pytest

from potatr.config import (
    EVAL_KEYS,
    LOSS_KEYS,
    MODEL_KEYS,
    RUN_KEYS,
    TRAIN_KEYS,
    has_loss_settings,
    load_config_files,
    unused_config_keys,
)


def write_json(path, data):
    path.write_text(json.dumps(data))
    return str(path)


def test_single_config_file(tmp_path):
    path = write_json(tmp_path / "config.json", {"lr": 5e-5, "datasets": [{"name": "a"}]})
    assert load_config_files([path]) == {"lr": 5e-5, "datasets": [{"name": "a"}]}


def test_config_files_are_merged_in_order(tmp_path):
    model = write_json(tmp_path / "model.json", {"hidden_dim": 256, "num_queries": 250})
    train = write_json(tmp_path / "train.json", {"lr": 5e-5})
    merged = load_config_files([model, train])
    assert merged == {"hidden_dim": 256, "num_queries": 250, "lr": 5e-5}
    assert list(merged) == ["hidden_dim", "num_queries", "lr"]


def test_repeated_setting_is_an_error(tmp_path):
    model = write_json(tmp_path / "model.json", {"hidden_dim": 256, "dropout": 0.1})
    train = write_json(tmp_path / "train.json", {"lr": 5e-5, "dropout": 0.04})
    with pytest.raises(
        ValueError,
        match=r"repeats settings from an earlier file: dropout \(also in .*model\.json\)",
    ):
        load_config_files([model, train])


def test_unused_config_keys():
    config = {
        "hidden_dim": 256,
        "lrr": 1e-4,  # typo
        "datasets": [
            {"data_root_dir": "data", "train_transform": "train", "img_extention": ".png"},
            {"data_root_dir": "more", "colour": "blue"},
        ],
    }
    assert unused_config_keys(config) == [
        "lrr",
        "datasets[0].img_extention",
        "datasets[1].colour",
    ]


def test_config_file_types_only_share_run_settings():
    assert not MODEL_KEYS & (LOSS_KEYS | TRAIN_KEYS | EVAL_KEYS)
    assert not LOSS_KEYS & (TRAIN_KEYS | EVAL_KEYS)
    assert TRAIN_KEYS & EVAL_KEYS == RUN_KEYS


def test_loss_settings_all_or_none():
    loss_settings = {key: 1 for key in LOSS_KEYS - {"aux_loss"}}
    assert has_loss_settings({"hidden_dim": 256}) is False
    assert has_loss_settings({"hidden_dim": 256, **loss_settings}) is True
    partial = dict(loss_settings)
    del partial["giou_loss_coef"]
    with pytest.raises(ValueError, match="Incomplete loss settings, missing: giou_loss_coef"):
        has_loss_settings(partial)


def test_loss_settings_given_as_none_count_as_absent():
    # Command-line options are merged in with None when they weren't given.
    assert has_loss_settings({"hidden_dim": 256, "adj_loss_coef": None}) is False


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
MODEL_CONFIG = str(CONFIGS / "model" / "potatr_v1.json")
LOSS_CONFIG = str(CONFIGS / "loss" / "potatr_v1.json")
STAGE1_CONFIG = str(CONFIGS / "train" / "pubtables_v2_stage1.json")
STAGE2_CONFIG = str(CONFIGS / "train" / "pubtables_v2_stage2.json")
EVAL_CONFIG = str(CONFIGS / "eval" / "pubtables_v2.json")


@pytest.mark.parametrize(
    "path, allowed_keys",
    [
        (MODEL_CONFIG, MODEL_KEYS),
        (LOSS_CONFIG, LOSS_KEYS),
        (STAGE1_CONFIG, TRAIN_KEYS),
        (STAGE2_CONFIG, TRAIN_KEYS),
        (EVAL_CONFIG, EVAL_KEYS),
    ],
)
def test_repo_config_contains_only_its_own_settings(path, allowed_keys):
    assert set(load_config_files([path])) <= allowed_keys


@pytest.mark.parametrize(
    "paths, needs_loss",
    [
        ([MODEL_CONFIG, LOSS_CONFIG, STAGE1_CONFIG], True),
        ([MODEL_CONFIG, LOSS_CONFIG, STAGE2_CONFIG], True),
        ([MODEL_CONFIG, EVAL_CONFIG], False),
        ([MODEL_CONFIG, LOSS_CONFIG, EVAL_CONFIG], True),
    ],
)
def test_repo_configs_combine_without_unused_settings(paths, needs_loss):
    config = load_config_files(paths)
    assert unused_config_keys(config) == []
    assert has_loss_settings(config) is needs_loss


def test_page_lists_in_the_repo_configs_exist():
    repo = CONFIGS.parent
    for path in (STAGE1_CONFIG, STAGE2_CONFIG):
        for dataset in load_config_files([path])["datasets"]:
            for key in ("train_page_list", "val_page_list", "test_page_list"):
                if key in dataset:
                    assert (repo / dataset[key]).is_file(), dataset[key]


def test_command_line_size_overrides_every_split_not_skipped():
    from potatr.train import get_max_size

    assert get_max_size({}, "val_max_size", None) is None
    assert get_max_size({"val_max_size": 250}, "val_max_size", None) == 250
    assert get_max_size({"val_max_size": 250}, "val_max_size", 16) == 16
    assert get_max_size({}, "val_max_size", 16) == 16
    # A config that skips the split keeps skipping it, e.g. a collection with no validation data.
    assert get_max_size({"val_max_size": 0}, "val_max_size", 16) == 0
    with pytest.raises(ValueError, match="non-negative"):
        get_max_size({}, "val_max_size", -1)

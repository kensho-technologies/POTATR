# Copyright 2025-present Kensho Technologies, LLC.
"""Tests for the table-graph dataset."""

from pathlib import Path

from PIL import Image
import pytest
import torch

from potatr import datasets
from potatr.config import load_config_files
from potatr.datasets import RandomCrop, TableGraphDataset


MODEL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model" / "potatr_v1.json"


def test_ap_is_computed_over_every_class(tmp_path):
    classes = load_config_files([str(MODEL_CONFIG)])["classes"]
    class_map = {name: idx for idx, name in enumerate(classes)}
    (tmp_path / "test" / "xml_annotations").mkdir(parents=True)
    (tmp_path / "test" / "images").mkdir()

    dataset = TableGraphDataset(str(tmp_path / "test"), class_map=class_map, make_coco=True)

    object_class_ids = sorted(idx for name, idx in class_map.items() if name != "no object")
    assert len(object_class_ids) == 16
    assert dataset.getCatIds() == object_class_ids
    assert [cat["id"] for cat in dataset.dataset["categories"]] == object_class_ids


def test_random_crop_drops_removed_objects_from_every_field(monkeypatch):
    # Always crop 5 pixels from each side, so the first box falls entirely outside the crop.
    monkeypatch.setattr(datasets.random, "randint", lambda low, high: high)
    image = Image.new("RGB", (100, 100))
    target = {
        "boxes": torch.tensor(
            [[0.0, 0.0, 3.0, 3.0], [10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]]
        ),
        "labels": torch.tensor([1, 2, 3]),
        "iscrowd": torch.zeros(3, dtype=torch.int64),
        "bbox_loss_mask": torch.tensor([1, 1, 0]),
        "adjacencies": torch.arange(9).reshape(3, 3),
    }

    cropped_image, cropped = RandomCrop(1, 5, 5, 5, 5)(image, target)

    assert cropped_image.size == (90, 90)
    assert cropped["boxes"].tolist() == [[5.0, 5.0, 45.0, 45.0], [55.0, 55.0, 85.0, 85.0]]
    assert cropped["labels"].tolist() == [2, 3]
    assert cropped["iscrowd"].tolist() == [0, 0]
    assert cropped["bbox_loss_mask"].tolist() == [1, 0]
    assert cropped["adjacencies"].tolist() == [[4, 5], [7, 8]]


def _write_split(root, page_ids):
    (root / "xml_annotations").mkdir(parents=True)
    (root / "images").mkdir()
    for page_id in page_ids:
        (root / "xml_annotations" / f"{page_id}.xml").write_text("<annotation/>")
        (root / "images" / f"{page_id}.jpg").write_bytes(b"")


def test_page_list_selects_pages(tmp_path):
    class_map = {"table": 0, "no object": 1}
    _write_split(tmp_path / "train", ["page_a", "page_b", "page_c"])
    page_list = tmp_path / "pages.txt"
    page_list.write_text("page_c\npage_a\n")

    dataset = TableGraphDataset(
        str(tmp_path / "train"), class_map=class_map, page_list=str(page_list)
    )
    assert dataset.page_ids == ["page_a", "page_c"]

    page_list.write_text("page_a\npage_d\n")
    with pytest.raises(ValueError, match="1 pages in .* have no annotation or image.*page_d"):
        TableGraphDataset(str(tmp_path / "train"), class_map=class_map, page_list=str(page_list))

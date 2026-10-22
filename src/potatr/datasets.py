# Copyright 2025-present Kensho Technologies, LLC.
# Portions copyright (C) 2021 Microsoft Corporation, from Table Transformer
# (https://github.com/microsoft/table-transformer), MIT License.
# Third-party portions have been modified. See THIRD_PARTY_NOTICES.md.
"""Dataset classes and transforms for POTATR table structure recognition training."""

from collections import defaultdict
import itertools
import json
import logging
import os
import random
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torchvision.transforms import functional as F

# Project imports
from potatr.detr.datasets import transforms as R


logger = logging.getLogger(__name__)


def _find_text(element: ET.Element, tag: str, xml_file: str) -> str:
    """Return the text of an XML element's child, with an error naming the file if it's missing."""
    child = element.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"{xml_file}: an object has no <{tag}>")
    return child.text


def read_pascal_voc(xml_file: str) -> tuple[list[list[float]], list[str]]:
    """Read the object boxes and labels from a PASCAL VOC XML annotation file.

    Returns (bboxes, labels): one [xmin, ymin, xmax, ymax] box and one class name per object.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    labels = []

    for object_ in root.iter("object"):
        label = _find_text(object_, "name", xml_file)

        box = object_.find("bndbox")
        if box is None:
            raise ValueError(f"{xml_file}: an object has no <bndbox>")
        ymin = float(_find_text(box, "ymin", xml_file))
        xmin = float(_find_text(box, "xmin", xml_file))
        ymax = float(_find_text(box, "ymax", xml_file))
        xmax = float(_find_text(box, "xmax", xml_file))

        bbox = [xmin, ymin, xmax, ymax]  # PASCAL VOC

        bboxes.append(bbox)
        labels.append(label)

    return bboxes, labels


class RandomCrop:
    def __init__(
        self, prob: float, left_pixels: int, top_pixels: int, right_pixels: int, bottom_pixels: int
    ) -> None:
        """With probability prob, crop a random 0 to N pixels from each side (N given per side)."""
        self.prob = prob
        self.left_pixels = left_pixels
        self.top_pixels = top_pixels
        self.right_pixels = right_pixels
        self.bottom_pixels = bottom_pixels

    def __call__(
        self, image: Image.Image, target: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Crop the image, clip the target boxes to it and drop the objects left empty.

        A dropped object is also removed from the target's other per-object fields: "labels",
        "iscrowd", "bbox_loss_mask" and the rows and columns of "adjacencies". If every object
        would be dropped, the image and target are returned uncropped.
        """
        if random.random() < self.prob:
            width, height = image.size
            left = random.randint(0, self.left_pixels)
            top = random.randint(0, self.top_pixels)
            right = width - random.randint(0, self.right_pixels)
            bottom = height - random.randint(0, self.bottom_pixels)
            cropped_image = image.crop((left, top, right, bottom))
            cropped_bboxes = []
            kept = []
            for i, bbox in enumerate(target["boxes"]):
                bbox = [
                    max(bbox[0], left) - left,
                    max(bbox[1], top) - top,
                    min(bbox[2], right) - left,
                    min(bbox[3], bottom) - top,
                ]
                if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    cropped_bboxes.append(bbox)
                    kept.append(i)

            if len(cropped_bboxes) > 0:
                target["boxes"] = torch.as_tensor(cropped_bboxes, dtype=torch.float32)
                kept_idx = torch.as_tensor(kept, dtype=torch.int64)
                for key in ("labels", "iscrowd", "bbox_loss_mask"):
                    if key in target:
                        target[key] = target[key][kept_idx]
                if "adjacencies" in target:
                    target["adjacencies"] = target["adjacencies"][kept_idx][:, kept_idx]
                return cropped_image, target

        return image, target


class RandomMaxResize:
    def __init__(self, min_max_size: int, max_max_size: int) -> None:
        """Resize the longest side to a random length from min_max_size to max_max_size pixels."""
        self.min_max_size = min_max_size
        self.max_max_size = max_max_size

    def __call__(
        self, image: Image.Image, target: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Resize the image and scale the target boxes to match."""
        width, height = image.size
        current_max_size = max(width, height)
        target_max_size = random.randint(self.min_max_size, self.max_max_size)
        scale = target_max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))
        resized_bboxes = []
        for bbox in target["boxes"]:
            bbox = [scale * elem for elem in bbox]
            resized_bboxes.append(bbox)

        if len(resized_bboxes) > 0:
            target["boxes"] = torch.as_tensor(resized_bboxes, dtype=torch.float32)
        else:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)

        return resized_image, target


class EnsureSizeRange:
    def __init__(self, min_max_size: int, max_max_size: int) -> None:
        """Resize only if the longest side is outside min_max_size to max_max_size pixels.

        The longest side is then resized to the nearer of the two limits.
        """
        self.min_max_size = min_max_size
        self.max_max_size = max_max_size

    def __call__(
        self, image: Image.Image, target: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Resize the image if needed and scale the target boxes to match."""
        width, height = image.size
        current_max_size = max(width, height)
        if current_max_size < self.min_max_size:
            target_max_size = self.min_max_size
        elif current_max_size > self.max_max_size:
            target_max_size = self.max_max_size
        else:
            target_max_size = current_max_size
        scale = target_max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))
        resized_bboxes = []
        for bbox in target["boxes"]:
            bbox = [scale * elem for elem in bbox]
            resized_bboxes.append(bbox)

        if len(resized_bboxes) > 0:
            target["boxes"] = torch.as_tensor(resized_bboxes, dtype=torch.float32)
        else:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)

        return resized_image, target


class RandomPad(torch.nn.Module):
    def __init__(
        self, left_pixels: int, top_pixels: int, right_pixels: int, bottom_pixels: int
    ) -> None:
        """Pad each side with a random 0 to N white pixels (N given per side)."""
        self.left_pixels = left_pixels
        self.top_pixels = top_pixels
        self.right_pixels = right_pixels
        self.bottom_pixels = bottom_pixels

        self.fill = 255
        self.padding_mode = "constant"

    def __call__(
        self, image: Image.Image, target: dict[str, Any]
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Pad the image and shift the target boxes to match."""
        width, height = image.size
        left_padding = random.randint(0, self.left_pixels)
        top_padding = random.randint(0, self.top_pixels)
        right_padding = random.randint(0, self.right_pixels)
        bottom_padding = random.randint(0, self.bottom_pixels)

        padded_bboxes = []
        for bbox in target["boxes"]:
            bbox = [
                bbox[0] + left_padding,
                bbox[1] + top_padding,
                bbox[2] + left_padding,
                bbox[3] + top_padding,
            ]
            padded_bboxes.append(bbox)

        if len(padded_bboxes) > 0:
            target["boxes"] = torch.as_tensor(padded_bboxes, dtype=torch.float32)

        padding = [left_padding, top_padding, right_padding, bottom_padding]
        padded_img = F.pad(image, padding, self.fill, self.padding_mode)

        return padded_img, target


normalize = R.Compose([R.ToTensor(), R.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def get_transform(transform_name: str) -> R.Compose:
    """Return the transform pipeline with the given name.

    "train" is the training augmentation: a small random crop or pad, then a random resize of
    the longest side to 900-1100 pixels (or, 10% of the time, only clamping it to 600-1200).
    "eval" resizes the longest side to 1000 pixels, for validation and testing. Both end by
    converting the image to a tensor and normalizing it with the ImageNet mean and std.
    """
    transform_list: list[Any]
    if transform_name == "train":
        transform_list = [
            R.RandomSelect(RandomCrop(1, 5, 5, 5, 5), RandomPad(5, 5, 5, 5), p=0.5),
            R.RandomSelect(RandomMaxResize(900, 1100), EnsureSizeRange(600, 1200), p=0.9),
        ]
    elif transform_name == "eval":
        transform_list = [RandomMaxResize(1000, 1000)]
    else:
        raise ValueError(f"Unknown transform {transform_name!r}; expected 'train' or 'eval'")

    return R.Compose(transform_list + [normalize])


def _isArrayLike(obj: Any) -> bool:
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class TableGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        transforms: R.Compose | None = None,
        max_size: int | None = None,
        make_coco: bool = False,
        include_eval: bool = False,
        xml_fileset: str = "filelist.txt",
        image_extension: str = ".jpg",
        class_map: dict[str, int] | None = None,
        page_list: str | None = None,
    ) -> None:
        """Load one split of a table-graph dataset, such as PubTables-v2.

        The pages are those with both an XML annotation and an image, or only the ones in page_list.

        Parameters:
            root: the split's directory, for example <data_root_dir>/train. Annotations are read
                  from root/xml_annotations (or root itself), table graphs from root/tables, and
                  images from root/images (or root/../images).
            transforms: transform applied to each (image, target) pair, from get_transform.
            max_size: None for every page, or N for a random subset of N pages (the same on every
                      DDP rank).
            make_coco: also build the COCO-style annotation index that pycocotools uses for AP.
            include_eval: add the image path to each target, for evaluation.
            xml_fileset: file in root's parent directory listing the annotation files. If it
                         can't be read, the annotation directory is listed instead.
            image_extension: the image file extension, for example ".jpg".
            class_map: class name to class index, including "no object". Objects of other
                       classes are ignored.
            page_list: a text file of page ids, one per line, to use only those pages. Every page
                       in it must have an annotation and an image.
        """
        self.root = root
        self.transforms = transforms
        self.make_coco = make_coco
        self.image_extension = image_extension
        self.include_eval = include_eval
        if class_map is None:
            raise ValueError("TableGraphDataset needs a class_map")
        self.class_map = class_map
        self.class_list = list(class_map)
        self.class_set = set(self.class_list)
        self.class_set.remove("no object")
        # The ids of the object classes, which AP and AR are computed over.
        self.category_ids = sorted(self.class_map[name] for name in self.class_set)
        self.table_graph_directory = os.path.join(root, "tables")
        # A name for the dataset in output file names, set by the caller if needed.
        self.name: str | None = None

        logger.debug("Class map: %s", self.class_map)

        # Accomodate both annotation directory formats
        annotations_directory = os.path.join(root, "xml_annotations")
        if not os.path.exists(annotations_directory):
            annotations_directory = root
        self.annotations_directory = annotations_directory
        logger.debug("Annotations directory: %s", self.annotations_directory)

        try:
            with open(os.path.join(root, "..", xml_fileset)) as file:
                lines = file.readlines()
                lines = [line.split("/")[-1] for line in lines]
        except (FileNotFoundError, OSError):
            lines = os.listdir(annotations_directory)
        xml_page_ids = set(
            [f.strip().replace(".xml", "") for f in lines if f.strip().endswith(".xml")]
        )

        # Accomodate both annotation directory formats
        image_directory = os.path.join(root, "images")
        if not os.path.exists(image_directory):
            image_directory = os.path.join(root, "..", "images")
        self.image_directory = image_directory

        try:
            with open(os.path.join(image_directory, "filelist.txt")) as file:
                lines = file.readlines()
        except (FileNotFoundError, OSError):
            lines = os.listdir(image_directory)
        png_page_ids = set(
            [
                f.strip().replace(self.image_extension, "")
                for f in lines
                if f.strip().endswith(self.image_extension)
            ]
        )

        self.page_ids = sorted(xml_page_ids.intersection(png_page_ids))
        if page_list is not None:
            with open(page_list) as file:
                listed = {line.strip() for line in file if line.strip()}
            missing = sorted(listed.difference(self.page_ids))
            if missing:
                raise ValueError(
                    f"{len(missing)} pages in {page_list} have no annotation or image in {root}, "
                    f"for example {missing[0]}"
                )
            self.page_ids = [page_id for page_id in self.page_ids if page_id in listed]
        if max_size is not None:
            random.shuffle(self.page_ids)
            self.page_ids = self.page_ids[:max_size]
            # Each DDP rank seeds `random` differently, so each rank's shuffle selects a different
            # subset. Use rank 0's selection on every rank, so DistributedSampler splits one shared
            # list. The shuffle above still runs on every rank, so each rank's random stream (used
            # later by training augmentation and sampling) is unchanged.
            if dist.is_available() and dist.is_initialized():
                shared_page_ids = [self.page_ids]
                dist.broadcast_object_list(shared_page_ids, src=0)
                self.page_ids = shared_page_ids[0]

        self.has_mask = False

        if self.make_coco:
            self.dataset: dict[str, list[dict[str, Any]]] = {}
            self.dataset["images"] = [{"id": idx} for idx, _ in enumerate(self.page_ids)]
            self.dataset["annotations"] = []
            ann_id = 0
            for image_id, page_id in enumerate(self.page_ids):
                annot_path = os.path.join(self.annotations_directory, page_id + ".xml")
                bboxes, classes = read_pascal_voc(annot_path)

                # Reduce class set
                keep_indices = [
                    idx for idx, class_label in enumerate(classes) if class_label in self.class_set
                ]
                bboxes = [bboxes[idx] for idx in keep_indices]
                labels = [self.class_map[classes[idx]] for idx in keep_indices]

                for bbox, label in zip(bboxes, labels):
                    ann = {
                        "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        "iscrowd": 0,
                        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                        "category_id": label,
                        "image_id": image_id,
                        "id": ann_id,
                        "ignore": 0,
                        "segmentation": [],
                    }
                    self.dataset["annotations"].append(ann)
                    ann_id += 1
            self.dataset["categories"] = [{"id": idx} for idx in self.category_ids]

            self.createIndex()

    def createIndex(self) -> None:
        """Build the COCO-style lookup tables that pycocotools reads, as COCO.createIndex does."""
        logger.debug("Creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        logger.debug("Index created")

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        # load images ad masks
        page_id = self.page_ids[idx]
        img_path = os.path.join(self.image_directory, page_id + self.image_extension)
        annot_path = os.path.join(self.annotations_directory, page_id + ".xml")
        table_graph_path = os.path.join(self.table_graph_directory, page_id + ".json")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        bboxes, labels = read_pascal_voc(annot_path)

        # Reduce class set
        keep_indices = [idx for idx, label in enumerate(labels) if label in self.class_set]
        bboxes = [bboxes[idx] for idx in keep_indices]
        labels = [labels[idx] for idx in keep_indices]

        num_objs = len(bboxes)
        adjacency_matrix = np.zeros((num_objs, num_objs), dtype=np.int64)

        if num_objs > 0:
            old_idx_2_new_idx = {v: k for k, v in enumerate(keep_indices)}

            with open(table_graph_path) as in_file:
                table_graph_data = json.load(in_file)

            for table in table_graph_data:
                object_nums = [table["page_object_num"]] + table["associated_page_object_nums"]
                object_idxs = [
                    old_idx_2_new_idx[elem] for elem in object_nums if elem in keep_indices
                ]

                for obj_idx1 in object_idxs:
                    for obj_idx2 in object_idxs:
                        if obj_idx1 == obj_idx2:
                            adjacency_matrix[obj_idx1, obj_idx2] = 0
                        elif (
                            labels[obj_idx1] == "table"
                            or labels[obj_idx1] == "table rotated"
                            or labels[obj_idx2] == "table"
                            or labels[obj_idx2] == "table rotated"
                        ):
                            adjacency_matrix[obj_idx1, obj_idx2] = 1
                        else:
                            adjacency_matrix[obj_idx1, obj_idx2] = 0

        label_ids = [self.class_map[class_label] for class_label in labels]

        # Convert to Torch Tensor
        if len(label_ids) > 0:
            boxes_tensor = torch.as_tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(label_ids, dtype=torch.int64)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)

        num_objs = boxes_tensor.shape[0]

        # Create target
        target: dict[str, Any] = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["adjacencies"] = torch.as_tensor(adjacency_matrix, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        target["bbox_loss_mask"] = torch.as_tensor([1 for ob in range(num_objs)], dtype=torch.int64)

        if self.include_eval:
            target["img_path"] = img_path

        if self.transforms is not None:
            img_tensor, target = self.transforms(img, target)

        del img

        return img_tensor, target

    def __len__(self) -> int:
        return len(self.page_ids)

    def getImgIds(self) -> range:
        """Return the ids of all images, for pycocotools."""
        return range(len(self.page_ids))

    def getCatIds(self) -> list[int]:
        """Return the ids of the classes that pycocotools computes AP and AR over: every class."""
        return self.category_ids

    def loadAnns(self, ids: Any = None) -> list[dict[str, Any]] | None:
        """Return the annotations with the given ids (a list of ids, or a single id)."""
        if ids is None:
            ids = []
        if _isArrayLike(ids):
            return [self.anns[ann_id] for ann_id in ids]
        elif isinstance(ids, int):
            return [self.anns[ids]]
        return None

    def getAnnIds(
        self, imgIds: Any = None, catIds: Any = None, areaRng: list[float] | None = None
    ) -> list[int]:
        """Return the ids of the annotations that pass every given filter.

        imgIds and catIds are lists of ids or single ids, and areaRng is [min, max]. An empty
        filter (the default) passes every annotation.
        """
        if imgIds is None:
            imgIds = []
        if catIds is None:
            catIds = []
        if areaRng is None:
            areaRng = []
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns if len(catIds) == 0 else [ann for ann in anns if ann["category_id"] in catIds]
            )
            anns = (
                anns
                if len(areaRng) == 0
                else [ann for ann in anns if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]]
            )

            ids = [ann["id"] for ann in anns]
        return ids

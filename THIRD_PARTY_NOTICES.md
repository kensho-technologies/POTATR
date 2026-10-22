# Third-Party Notices

This project includes code from the following third-party sources. Full license texts are in the
[`licenses/`](licenses/) directory.

## DETR (DEtection TRansformer)

**Location:** `src/potatr/detr/` (vendored). Code derived from DETR also appears in
`src/potatr/model.py`, `src/potatr/matcher.py` and `src/potatr/evaluate.py`.

**Source:** https://github.com/facebookresearch/detr

**License:** Apache License 2.0 (full text: [`licenses/detr-LICENSE`](licenses/detr-LICENSE))

**Copyright:** Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

A subset of the DETR codebase is vendored in this project to provide backbone,
transformer, and training utilities used by the POTATR model.

**Modifications:** The vendored files have been modified from the original. Unused modules were
removed, import paths were changed for this package, and parts of the backbone, transformer,
training loop, evaluation, and utilities were adapted for POTATR. Each modified file is marked
with a notice below its original copyright line. The DETR-derived code in `model.py`
(set criterion, post-processing and MLP), `matcher.py` (Hungarian matcher) and `evaluate.py`
(evaluation loop) has also been modified, and those files note it in their headers.

---

## Table Transformer

**Location:** Portions of `src/potatr/datasets.py`, `src/potatr/postprocess.py`,
`src/potatr/metrics.py`, `src/potatr/evaluate.py` and `src/potatr/train.py`.

**Source:** https://github.com/microsoft/table-transformer

**License:** MIT License (full text: [`licenses/table-transformer-LICENSE`](licenses/table-transformer-LICENSE))

**Copyright:** Copyright (C) 2021 Microsoft Corporation.

POTATR's dataset loading and augmentation, table post-processing, evaluation metrics and
training entry point are adapted from Table Transformer. This code has been modified for POTATR,
and each file notes it in its header.

---

## LOBSTR (Linked Objects Transformer)

**Location:** Portions of `src/potatr/model.py` and `src/potatr/matcher.py`.

**Source:** https://github.com/microsoft/LOBSTR

**License:** MIT License (full text: [`licenses/LOBSTR-LICENSE`](licenses/LOBSTR-LICENSE))

**Copyright:** Copyright (C) 2023 Microsoft Corporation.

POTATR's graph model (node and edge prediction on top of DETR, with an adjacency loss) and its
Hungarian matcher are extended from an early public version of LOBSTR. This code has been
modified for POTATR, and both files note it in their headers.

---

## TEDS (Tree Edit Distance-based Similarity)

**Location:** `src/potatr/teds.py`

**Source:** https://github.com/ibm-aur-nlp/PubTabNet (`src/metric.py`)

**License:** Apache License 2.0 (full text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt))

**Copyright:** Copyright 2020 IBM. Author: peter.zhong@au1.ibm.com

The TEDS metric implementation is vendored for table structure evaluation.

**Modifications:** The file has been modified from the original. The `distance` package was
replaced with `rapidfuzz` for string edit distance, the parallel `batch_evaluate` helper was
removed, and a `hungarian_teds` function was added for matching multiple predicted and
ground-truth tables. It is marked with a notice below its original copyright lines.

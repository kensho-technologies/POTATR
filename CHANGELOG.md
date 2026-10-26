# Changelog

## 0.2.0

### Fixed
- A bug in evaluation for caption and footer text.

## 0.1.0

First public release: training, evaluation and inference code for POTATR v1.1 on PubTables-v2.

### Added
- The POTATR model (DETR with an edge head linking each table to its objects), its loss and matcher.
- Two-stage training of POTATR v1.1 on PubTables-v2 (`potatr-train`), with configs and page lists.
- Evaluation with GriTS, TEDS, edge F1 and caption/footer text F1 (`potatr-train --mode test`).
- Inference on page images and their words (`potatr-infer`).
- `potatr-init`, which makes the starting checkpoint from the public TATR-v1.1-Pub weights.
- Loading model weights from `.safetensors` files as well as `.pth` checkpoints.

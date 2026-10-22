# Page-Object Table Transformer (POTATR)

POTATR is a transformer-based model for recognizing tables on document pages. From a page image it
detects each table, the objects that make up its structure (rows, columns, column headers,
projected row headers, spanning cells), as well as its surrounding context (captions and footers).
It predicts a graph that links each table to its objects. With the page's words, the predicted
structure is turned into table cells with their text.

POTATR extends DETR (DEtection TRansformer) with an edge prediction head that classifies every
ordered pair of object queries. This repository contains the code to train POTATR and to
evaluate it on PubTables-v2.

## Installation

POTATR needs Python 3.11 or later. Training and evaluation run on GPUs, with one process per GPU.

With [uv](https://docs.astral.sh/uv/), install the exact versions in `uv.lock`:

```bash
uv sync
source .venv/bin/activate
```

Or install with pip:

```bash
pip install -e .
```

The PyTorch wheels in `uv.lock` (torch 2.14) are built for CUDA 13, which needs NVIDIA driver 580
or later and a GPU of compute capability 7.5 or higher (Turing, such as the T4, or newer). This is
the setup POTATR was tested with. Other recent PyTorch builds, installed with pip, should also
work, though they're untested: for example a CUDA 12 build for older NVIDIA GPUs, or a ROCm build
for AMD GPUs. Results may differ slightly with other versions and hardware.

## Data

POTATR is trained and evaluated on the Single Pages collection of
[PubTables-v2](https://huggingface.co/datasets/kensho/PubTables-v2) (CDLA-Permissive-2.0 license):
pages with every table annotated, split into 408,647 training, 51,214 validation and 7,680 test
pages.

Download the Single Pages files into `data/PubTables-v2` and extract them there. The archives
already contain their `Single Pages/<split>/<part>/` paths, so extract each one in place:

```bash
mkdir -p data/PubTables-v2 && cd data/PubTables-v2
for split in train val test; do
  for part in images tables words xml-annotations; do
    curl -L -C - -O "https://huggingface.co/datasets/kensho/PubTables-v2/resolve/main/PubTables-v2_Single-Pages_${split}_${part}.tar.gz"
  done
done
for f in PubTables-v2_Single-Pages_*.tar.gz; do tar -xzf "$f"; done
```

The 12 files are 171 GB, 139 GB of it the training images. Training reads the images, tables and
annotations of the train and val splits; evaluation reads the test split, including the page
words, so evaluating alone needs only the 4 test files (2.7 GB). The result is:

```
data/PubTables-v2/Single Pages/
├── train/
│   ├── images/            # page images, <page_id>.jpg
│   ├── xml_annotations/   # object boxes and classes, PASCAL VOC format, <page_id>.xml
│   ├── tables/            # each table's graph, HTML and grids, <page_id>.json
│   └── words/             # the page's words with their boxes, <page_id>.json
├── val/
└── test/
```

The second stage of training also uses the pages without tables from PubTables-v2's Full
Documents collection. For that, download its training images and annotations (27 GB) the same
way:

```bash
cd data/PubTables-v2
for part in images xml-annotations; do
  curl -L -C - -O "https://huggingface.co/datasets/kensho/PubTables-v2/resolve/main/PubTables-v2_Full-Documents_train_${part}.tar.gz"
done
for f in PubTables-v2_Full-Documents_train_*.tar.gz; do tar -xzf "$f"; done
```

This gives `data/PubTables-v2/Full Documents/train/` with `images/` and `xml_annotations/`.

The configs' `data_root_dir` settings point at these folders; change them in the configs in
`configs/train/` and `configs/eval/` if your copy is elsewhere. Evaluation reads each page's words
from `words/`; use `--table_words_dir` to read them from another directory in each split.

## Pretrained weights

The trained POTATR v1.1 weights are at
[kensho/POTATR-v1.1-Pub](https://huggingface.co/kensho/POTATR-v1.1-Pub).
`configs/model/potatr_v1.json` is the model configuration that goes with them:

```bash
wget https://huggingface.co/kensho/POTATR-v1.1-Pub/resolve/main/POTATR-v1.1-Pub.safetensors
```

Training starts from Table Transformer v1.1 trained on PubTables-1M,
[TATR-v1.1-Pub](https://huggingface.co/bsmock/TATR-v1.1-Pub) (MIT license). `potatr-init` makes
the starting checkpoint from it, copying the backbone, transformer and box head, and adding
POTATR's extra object queries with random values from a fixed seed:

```bash
wget https://huggingface.co/bsmock/TATR-v1.1-Pub/resolve/main/TATR-v1.1-Pub-msft.pth
potatr-init \
    --config_file configs/model/potatr_v1.json \
    --tatr_weights TATR-v1.1-Pub-msft.pth \
    --output potatr_init.pth
```

POTATR v1.1 was trained from a checkpoint made the same way, but whose extra queries came from an
unrecorded random seed, so a training run from `potatr_init.pth` matches it in everything except
those starting values.

## Configuration

Settings are split across JSON config files, which `--config_file` merges in order. Each setting
may appear in only one of them:

- `configs/model/potatr_v1.json`: the model architecture (backbone, transformer layers, hidden
  sizes, number of queries, classes). It goes with the model weights.
- `configs/loss/potatr_v1.json`: loss weights and matching costs.
- `configs/train/pubtables_v2_stage1.json`, `configs/train/pubtables_v2_stage2.json`: training
  settings (learning rate, schedule, batch size, epochs) and the training and validation data, for
  the two stages of training.
- `configs/eval/pubtables_v2.json`: evaluation settings and the test data.

Training uses the model, loss and training configs. Evaluation uses the model and evaluation
configs; add the loss config to also compute test-time losses. A dataset entry can use only some
of a collection's pages, listed in a file named by `train_page_list`, `val_page_list` or
`test_page_list`; the page lists that stage 2 uses are in `configs/data/pubtables_v2/`. Settings
given on the command line override the configs, and the run warns about any config setting that
POTATR doesn't use. `--train_max_size`, `--val_max_size` and `--test_max_size` apply to every
dataset entry, except entries that skip that split with a size of 0, such as stage 2's Full
Documents entry, which has no validation data.

## Training

POTATR v1.1 is trained in two stages, both on 8 GPUs:

1. **Stage 1** fine-tunes `potatr_init.pth` for 97 epochs on the Single Pages training split
   (408,647 pages), starting at a learning rate of 5e-5 and multiplying it by 0.9 every 2 epochs.
   This is the training of POTATR v1.0.
2. **Stage 2** fine-tunes stage 1's epoch-97 weights for 35 more epochs, starting again at 5e-5
   and multiplying the learning rate by 0.9 every epoch. Each epoch trains on three sets of pages:
   - all 408,647 pages of the Single Pages training split, as in stage 1;
   - the 33,678 of those pages that have long tables, a second time;
   - 2,000 random pages without tables from the Full Documents training split.

   POTATR v1.1 is the model after its last epoch.

```bash
# Stage 1
python -m potatr.train \
    --config_file configs/model/potatr_v1.json configs/loss/potatr_v1.json configs/train/pubtables_v2_stage1.json \
    --model_load_path potatr_init.pth \
    --load_weights_only \
    --train_all_weights \
    --model_save_dir output/stage1

# Stage 2
python -m potatr.train \
    --config_file configs/model/potatr_v1.json configs/loss/potatr_v1.json configs/train/pubtables_v2_stage2.json \
    --model_load_path output/stage1/model_97.pth \
    --load_weights_only \
    --train_all_weights \
    --model_save_dir output/stage2
```

`scripts/train.sh <train config> <checkpoint> <output directory>` runs the same command for either
stage. Leave out `--model_load_path` and `--load_weights_only` to train from an
ImageNet-pretrained backbone instead.

Training uses every available GPU, with a batch size of 2 per GPU. POTATR v1.1 was trained on 8
NVIDIA T4 GPUs (an effective batch size of 16), where an epoch takes about 2.6 hours in stage 1 and
2.9 hours in stage 2: about 10.5 and 4 days. The number of GPUs sets the effective batch size and
the number of steps per epoch, so use 8 GPUs to reproduce it.

Each epoch, the output directory gets `model.pth` (the model, optimizer and scheduler state, for
resuming), `model_<epoch>.pth` (the model weights; see `checkpoint_freq`) and
`model.safetensors` (the latest model weights). `--model_load_path` takes either format. To
resume, pass the `model.pth` of an earlier run as `--model_load_path`, without
`--load_weights_only`.

## Evaluation

```bash
python -m potatr.train \
    --mode test \
    --config_file configs/model/potatr_v1.json configs/eval/pubtables_v2.json \
    --model_load_path POTATR-v1.1-Pub.safetensors \
    --model_save_dir output/eval
```

By default this evaluates every page of the test split, once each, across all available GPUs.
Set `test_max_size` (in the config, or with `--test_max_size N`) to evaluate a random subset of
N pages. Results are logged and saved to `metrics_<dataset name>.json` in the output directory,
with the dataset-level metrics, the metrics of each page and the inference timing.

Evaluate with a batch size of 1, as the eval config does: with larger batches, padding changes
some predictions.

The metrics are:

- **GriTS_Top, GriTS_Con:** grid table similarity of each table's cell topology and cell text.
- **TEDS, TEDS_Struct:** tree edit distance similarity of each table's HTML, with and without
  the cell text.
- **Edge F1 at IoU 0.5 and 0.8:** F1 of the predicted links between tables and their objects.
- **Caption and footer text F1:** how well the text of each table's caption and footer is found,
  from the objects linked to the table.

## Inference

To find the tables on your own pages, run `potatr-infer` on a directory of page images and a
directory of the pages' words:

```bash
potatr-infer \
    --config_file configs/model/potatr_v1.json \
    --model_load_path POTATR-v1.1-Pub.safetensors \
    --images_dir pages/images \
    --words_dir pages/words \
    --output_dir output/tables
```

For each image `<name>.jpg` (or `.jpeg`, `.png`), the words are read from `<name>.json` in the
words directory: a list of words, each with its `text`, its `bbox` (`[x0, y0, x1, y1]` in image
pixels) and its `block_num`, `line_num` and `span_num`, which order the words when their text is
joined. The words files of PubTables-v2 have this format; for other pages, they can come from the
PDF or from OCR.

The tables of each page are written to `<name>.json` in the output directory. Each table has its
HTML, its cells (rows, columns, box, text and whether it's a header cell), its caption and footer
text, and its box and score. `--device cpu` runs without a GPU.

## Project structure

```
src/potatr/
├── train.py          # Training and evaluation entry point (DDP)
├── config.py         # Config file settings, merging and checks
├── model.py          # POTATR model, loss (SetCriterion) and output postprocessing
├── matcher.py        # Hungarian matching of predictions to targets
├── datasets.py       # TableGraphDataset and augmentation transforms
├── evaluate.py       # Evaluation loop
├── inference.py      # Tables, captions and footers for page images
├── initialize.py     # The starting checkpoint, from Table Transformer weights
├── postprocess.py    # Model outputs to table structures and cells
├── metrics.py        # GriTS, TEDS and text metrics
├── graph_metrics.py  # Edge prediction metrics
├── geometry.py       # Bounding box utilities
├── teds.py           # Tree Edit Distance Similarity (vendored, from PubTabNet)
└── detr/             # Vendored subset of DETR
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{smock2026potatr,
  title={{POTATR}: A Lightweight Image-to-Graph Model for Page-Level Table Extraction},
  author={Smock, Brandon and Liang, Libin and Sokolov, Maxim and Ramesh, Amrit and Faucon-Morin, Valerie and Khanam, Tayyibah and Courtland, Maury},
  booktitle={International Workshop on Document Analysis Systems},
  pages={3--17},
  year={2026},
  organization={Springer}
}
```

If you use the PubTables-v2 dataset, please also cite:

```bibtex
@article{smock2025pubtables,
  title={Pub{T}ables-v2: A new large-scale dataset for full-page and multi-page table extraction},
  author={Smock, Brandon and Faucon-Morin, Valerie and Sokolov, Max and Liang, Libin and Khanam, Tayyibah and Ramesh, Amrit and Courtland, Maury},
  journal={arXiv preprint arXiv:2512.10888},
  year={2025}
}
```

## License

Licensed under the MIT License. Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Copyright 2025-present Kensho Technologies, LLC. The present date is determined by the timestamp of the most recent commit in the repository.

This project includes code from third parties under their respective licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

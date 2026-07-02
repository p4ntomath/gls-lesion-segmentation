# GLS Lesion Quantification — U-Net vs Attention U-Net

COS700 Research Project: grey leaf spot (GLS) lesion segmentation and
leaf-aware coverage estimation in maize leaf images.

This repository includes the working data preprocessing, training, evaluation,
and statistical comparison scripts for GLS lesion segmentation experiments.

## Layout

```
gls-lesion-segmentation/
├── configs/
│   ├── base.yaml               # paths, data, training defaults — shared by every run
│   ├── unet.yaml                # U-Net architecture params
│   └── attention_unet.yaml       # Attention U-Net architecture params
│
├── data/
│   ├── raw/
│   │   ├── images/              # downloaded RGB leaf images (untouched)
│   │   ├── masks/               # downloaded raw 32-bit instance PNGs, pre-decode
│   │   └── annotations/
│   │       └── dataset.json      # the Segments.ai export — never modified
│   │
│   ├── cache/                     # scratch space for transient intermediates
│   │   └── (optional)             # only used by later experiments if needed
│   │
│   ├── processed/
│   │   ├── manifest.csv           # single source of truth: sample_id, image_url,
│   │   │                           # mask_url, label_status, num_instances (+ coverage,
│   │   │                           # leaf_area etc. appended by later steps)
│   │   ├── images/                 # resized/normalised images
│   │   ├── lesion_masks/            # decoded binary GLS masks
│   │   └── leaf_masks/               # cached output of pipelines/generate_leaf_masks.py
│   │
│   └── splits/
│       ├── train.txt              # one sample_id per line — no paths, the loader
│       ├── val.txt                 # builds processed/images/<id>.jpg etc. from the id
│       └── test.txt
│
├── src/
│   ├── data/            # §4.1.1, §4.2.3–4.2.6: parse_dataset, generate_masks,
│   │                     #   split_data, dataset.py (PyTorch Dataset), augmentations
│   ├── models/            # §4.2.7: unet.py, attention_unet.py
│   ├── training/            # §4.2.8: losses.py, metrics.py, trainer.py (Trainer class)
│   ├── evaluation/           # §4.2.9–4.2.10: coverage.py, evaluate.py, stats.py (Wilcoxon)
│   └── utils/                 # seed.py, viz.py
│
├── pipelines/
│   └── generate_leaf_masks.py   # your existing leaf-segmentation code goes here.
│                                  # Run once, independently of training — writes to
│                                  # data/processed/leaf_masks/. Nothing under src/training
│                                  # or src/evaluation calls this directly.
│
├── scripts/                      # thin CLI entry points — the things you actually run
│   ├── preprocess.py               # parse -> generate_masks -> split_data
│   ├── train.py                     # loads experiments/<exp>/config.yaml, runs Trainer
│   └── evaluate.py                   # runs src.evaluation.evaluate for one experiment
│
├── experiments/
│   ├── exp01_unet_noaug/
│   │   ├── config.yaml          # resolved snapshot (extends base+model, one override)
│   │   └── notes.md              # run log: dates, best epoch, final metrics, observations
│   ├── exp02_unet_aug/
│   ├── exp03_attnunet_noaug/
│   └── exp04_attnunet_aug/
│
├── outputs/
│   ├── checkpoints/    # best model weights per experiment
│   ├── logs/            # per-epoch training curves
│   └── figures/          # overlays, plots for the report
│
├── reports/                       # thesis-facing assets, separate from outputs/
│   ├── figures/                     # final polished figures for the report
│   ├── tables/                       # exported results tables (from experiments/*/results.json)
│   └── drafts/                        # methodology notes, draft sections
│
├── tests/
├── requirements.txt
└── .gitignore
```

## Full setup instructions

Follow these steps from a fresh clone to a first successful run.

### 1. Clone the repository

```bash
git clone https://github.com/p4ntomath/gls-lesion-segmentation.git
cd gls-lesion-segmentation
```

### 2. Create and activate a Python environment

On Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place your dataset file at:

```text
data/raw/annotations/dataset.json
```

This is the only dataset input file you need to provide manually. The preprocessing pipeline will create the rest of the data structure for you.

### 5. Run preprocessing

```bash
python scripts/preprocess.py --config configs/base.yaml
```

This will generate the processed images, lesion masks, split files, and related outputs under the data folders.

### 6. Train a model

```bash
python scripts/train.py --experiment exp01_unet_noaug
```

### 7. Evaluate a trained model

```bash
python scripts/evaluate.py --experiment exp01_unet_noaug --config configs/base.yaml
```

### 8. Run tests

```bash
py -m pytest -q tests/test_dataset.py
```

## Planned build order

1.  **`src/data/parse_dataset.py`** → `data/processed/manifest.csv`
   (implemented & tested against the real `dataset.json` — 438 labeled
   samples parsed, no duplicate ids)
2. **`src/data/generate_masks.py`** → download images + raw instance PNGs to
   `data/raw/`, decode to binary lesion masks in `data/processed/`
3. **`src/data/split_data.py`** → stratified `train/val/test.txt` (ids only)
4. **`scripts/preprocess.py`** — wires 1–3 together, one command
5. **`src/data/dataset.py`** + **`src/data/augmentations.py`**
6. **`src/models/unet.py`**
7. **`src/training/{losses,metrics,trainer}.py`** + **`scripts/train.py`**
8. Run **Experiment 1 (U-Net, no augmentation)** end-to-end
9. **`pipelines/generate_leaf_masks.py`** — drop in your existing leaf-extraction
   code, run once to populate `data/processed/leaf_masks/`
10. **`src/evaluation/coverage.py`**, `evaluate.py`, `stats.py` — plug the leaf
    masks in for §4.2.9 coverage estimation, then Attention U-Net, augmentation,
    and the Wilcoxon comparisons (§4.2.12)



## Note on data access

`data/raw/annotations/dataset.json` stores images/masks as URLs hosted on
Segments.ai's S3 bucket (`segmentsai-prod.s3.eu-west-2.amazonaws.com`).
`src/data/generate_masks.py` needs network access to that host to download
them.

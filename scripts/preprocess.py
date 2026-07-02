"""CLI entry point: full data preparation, start to finish.

Usage:
    python scripts/preprocess.py --config configs/base.yaml

Calls, in order:
    src.data.parse_dataset.main()      -> data/processed/manifest.csv
    src.data.generate_masks.main()     -> data/processed/images/, lesion_masks/
    src.data.split_data.main()         -> data/splits/{train,val,test}.txt

Does NOT run leaf extraction — that's pipelines/generate_leaf_masks.py,
run separately since it's independent of the train/val/test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.generate_masks import main as generate_masks_main
from src.data.parse_dataset import main as parse_dataset_main
from src.data.split_data import main as split_data_main


def main(config_path: str = "configs/base.yaml") -> None:
    parse_dataset_main(config_path)
    generate_masks_main(config_path)
    split_data_main(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full GLS data preprocessing pipeline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path")
    args = parser.parse_args()
    main(args.config)

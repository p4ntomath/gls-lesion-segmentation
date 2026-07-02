"""CLI entry point for evaluating a trained experiment on the test set.

Usage:
    python scripts/evaluate.py --experiment exp01_unet_noaug

Calls src.evaluation.evaluate.main(experiment), which writes
experiments/<experiment>/results.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.evaluate import main


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained GLS segmentation experiment")
    parser.add_argument("--experiment", required=True, help="Experiment folder name under experiments/")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path")
    args = parser.parse_args()
    main(args.experiment, config_path=args.config)


if __name__ == "__main__":
    _cli()

"""CLI: generate qualitative prediction visualizations for an experiment.

Usage:
    python scripts/visualize.py --experiment exp01_unet_noaug --n 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.evaluate import load_experiment_config, build_model, _load_checkpoint, build_test_loader, run_inference
from src.utils.viz import plot_prediction_grid
import yaml


def main(experiment: str, n: int = 6, out_dir: str = "outputs/figures/predictions") -> None:
    config = load_experiment_config(experiment)

    model = build_model(config)
    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
    _load_checkpoint(model, Path(config["paths"]["checkpoints_dir"]) / f"{experiment}.pt", device)
    model.to(device)

    test_loader = build_test_loader(config)
    predictions, ground_truth = run_inference(model, test_loader, device)

    # Choose sample ids to visualize: first `n` sorted
    sample_ids = sorted(ground_truth)[:n]

    processed_images_dir = Path(config["paths"]["processed_images_dir"])
    lesion_masks_dir = Path(config["paths"]["lesion_masks_dir"])
    image_size = int(config.get("data", {}).get("image_size", 256))

    images = []
    gt_masks = []
    pred_masks = []
    from PIL import Image

    for sample_id in sample_ids:
        img_path = processed_images_dir / f"{sample_id}.jpg"
        gt_path = lesion_masks_dir / f"{sample_id}.png"
        if not img_path.exists() or not gt_path.exists() or sample_id not in predictions:
            # skip missing files/predictions
            continue
        img = Image.open(img_path).convert("RGB")
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
        img_arr = np.asarray(img)

        gt = Image.open(gt_path).convert("L")
        if gt.size != (image_size, image_size):
            gt = gt.resize((image_size, image_size), resample=Image.NEAREST)
        gt_arr = (np.asarray(gt) > 0).astype(np.uint8)

        pred_arr = predictions[sample_id]
        # predictions are numpy arrays (H,W); resize if needed to image_size
        if pred_arr.shape != (image_size, image_size):
            pred_img = Image.fromarray((pred_arr * 255).astype(np.uint8), mode="L").resize((image_size, image_size), resample=Image.NEAREST)
            pred_arr = (np.asarray(pred_img) > 0).astype(np.uint8)

        images.append(img_arr)
        gt_masks.append(gt_arr)
        pred_masks.append(pred_arr)

    out_dir = Path(out_dir) / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions_grid.png"

    if len(images) == 0:
        print("No images found to visualize.")
        return

    fig = plot_prediction_grid(images, gt_masks, pred_masks, n=len(images), save_path=out_path)
    print(f"Saved visualization grid to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate qualitative prediction images for an experiment")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--out", default="outputs/figures/predictions")
    args = parser.parse_args()
    main(args.experiment, n=args.n, out_dir=args.out)

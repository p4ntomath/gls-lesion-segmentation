"""CLI: generate qualitative review figures for a trained experiment.

Produces, per selected test sample: Image | Leaf region | Ground truth
lesions | Predicted lesions | Prediction confidence heatmap -- addressing
the supervisor review request to look at leaf identification, lesion masks,
high/low-lesion examples, success/failure cases, and a confidence heatmap
(not just a binary mask).

Usage:
    python scripts/qualitative_report.py --experiment exp01_unet_noaug --n-per-category 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import albumentations as A

from src.data.augmentations import get_eval_transforms
from src.data.dataset import GLSDataset
from src.evaluation.evaluate import load_experiment_config, build_model, _load_checkpoint
from src.utils.viz import plot_qualitative_panel


def select_samples(per_image: list[dict], n_per_category: int) -> dict[str, list[dict]]:
    """
    Pick samples for each category the supervisor asked for. A sample can
    appear in more than one category (e.g. a high-coverage sample might
    also be a failure case) -- that overlap is informative, not a bug, so
    it's reported rather than deduplicated away.

    Categories:
        high_coverage: highest reference_coverage (lots of lesions)
        low_coverage:  lowest NONZERO reference_coverage (very few lesions)
        success:       highest dice
        failure:       lowest dice
    """
    with_coverage = [r for r in per_image if "reference_coverage" in r]
    if len(with_coverage) < len(per_image):
        missing = len(per_image) - len(with_coverage)
        print(f"NOTE: {missing} samples have no leaf mask / coverage yet -- "
              f"excluded from high/low-coverage selection (still eligible for success/failure).")

    by_coverage_desc = sorted(with_coverage, key=lambda r: r["reference_coverage"], reverse=True)
    nonzero = [r for r in with_coverage if r["reference_coverage"] > 0]
    by_coverage_asc_nonzero = sorted(nonzero, key=lambda r: r["reference_coverage"])

    by_dice_desc = sorted(per_image, key=lambda r: r["dice"], reverse=True)
    by_dice_asc = sorted(per_image, key=lambda r: r["dice"])

    return {
        "high_coverage": by_coverage_desc[:n_per_category],
        "low_coverage": by_coverage_asc_nonzero[:n_per_category],
        "success": by_dice_desc[:n_per_category],
        "failure": by_dice_asc[:n_per_category],
    }


def main(experiment: str, n_per_category: int = 2, out_dir: str = "outputs/figures/qualitative") -> None:
    config = load_experiment_config(experiment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = config["data"]["image_size"]
    threshold = config["training"].get("threshold", 0.5)

    model = build_model(config)
    _load_checkpoint(model, Path(config["paths"]["checkpoints_dir"]) / f"{experiment}.pt", device)
    model.to(device).eval()

    results_path = Path("experiments") / experiment / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found -- run scripts/evaluate.py for this experiment first")
    results = json.load(open(results_path))
    per_image = results["per_image"]

    selections = select_samples(per_image, n_per_category)

    images_dir = Path(config["paths"]["processed_images_dir"])
    masks_dir = Path(config["paths"]["lesion_masks_dir"])
    leaf_masks_dir = Path(config["paths"]["leaf_masks_dir"])

    model_transform = get_eval_transforms(image_size)
    display_resize = A.Compose([A.Resize(image_size, image_size)])

    out_dir_path = Path(out_dir) / experiment
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Track which categories each sample_id belongs to, for the summary + filename
    sample_categories: dict[str, list[str]] = {}
    sample_rows: dict[str, dict] = {}
    for category, rows in selections.items():
        for row in rows:
            sample_categories.setdefault(row["sample_id"], []).append(category)
            sample_rows[row["sample_id"]] = row

    print(f"Generating qualitative panels for {len(sample_categories)} unique samples "
          f"({sum(len(v) for v in selections.values())} category slots, some overlap expected)...")

    for sample_id, categories in sample_categories.items():
        row = sample_rows[sample_id]

        raw_image = GLSDataset._load_rgb(images_dir / f"{sample_id}.jpg")
        raw_gt_mask = GLSDataset._load_mask(masks_dir / f"{sample_id}.png")

        leaf_path = leaf_masks_dir / f"{sample_id}.png"
        if leaf_path.exists():
            from PIL import Image
            raw_leaf_mask = (np.array(Image.open(leaf_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
        else:
            raw_leaf_mask = np.zeros(raw_image.shape[:2], dtype=np.uint8)
            print(f"  WARNING: no leaf mask for {sample_id}, showing blank leaf panel")

        model_input = model_transform(image=raw_image, mask=raw_gt_mask)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(model_input)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask = (probs >= threshold).astype(np.uint8)

        display = display_resize(image=raw_image, mask=raw_gt_mask)
        display_image, display_gt = display["image"], display["mask"]
        display_leaf = display_resize(image=raw_image, mask=raw_leaf_mask)["mask"]

        cov = row.get("reference_coverage")
        cov_str = f", coverage={cov:.2f}%" if cov is not None else ""
        title = f"{sample_id}  [{', '.join(categories)}]  dice={row['dice']:.3f}{cov_str}"

        safe_categories = "-".join(categories)
        save_path = out_dir_path / f"{safe_categories}_{sample_id}.png"
        plot_qualitative_panel(display_image, display_leaf, display_gt, pred_mask, probs,
                                title=title, save_path=save_path)
        print(f"  saved {save_path}")

    print(f"\nDone -- {len(sample_categories)} panels in {out_dir_path}")
    print("\nSelection summary:")
    for category, rows in selections.items():
        print(f"  {category}: " + ", ".join(
            f"{r['sample_id']} (dice={r['dice']:.3f}"
            + (f", cov={r['reference_coverage']:.2f}%)" if "reference_coverage" in r else ")")
            for r in rows
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate qualitative review figures for an experiment")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--n-per-category", type=int, default=2)
    parser.add_argument("--out", default="outputs/figures/qualitative")
    args = parser.parse_args()
    main(args.experiment, n_per_category=args.n_per_category, out_dir=args.out)
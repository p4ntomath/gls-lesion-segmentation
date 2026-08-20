"""Compare the legacy 256x256 full-image pipeline with the 512x512 leaf-aware pipeline.

Usage:
    python scripts/compare_pipelines.py \
        --old-checkpoints-dir /path/to/old/checkpoints \
        --new-checkpoints-dir /path/to/new/checkpoints \
        --n 4

Outputs are written to ``outputs/comparison``:
    comparison_table.csv
    comparison_table.md
    <experiment>_comparison.png

Both checkpoints must use the same model architecture and the same test split.
The old checkpoint is evaluated with 256x256 full-image preprocessing regardless
of the current experiment configuration. The new checkpoint uses the current
experiment configuration, including leaf masking.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from src.data.augmentations import get_eval_transforms
from src.data.dataset import GLSDataset
from src.evaluation.evaluate import (
    _load_checkpoint,
    build_model,
    build_test_loader,
    load_experiment_config,
)
from src.training.metrics import (
    confusion_counts,
    dice_coefficient,
    iou_score,
    precision_score,
    recall_score,
)

EXPERIMENTS = [
    "exp01_unet_noaug",
    "exp02_unet_aug",
    "exp03_attnunet_noaug",
    "exp04_attnunet_aug",
]

EXP_LABELS = {
    "exp01_unet_noaug": "U-Net, No Aug",
    "exp02_unet_aug": "U-Net, Aug",
    "exp03_attnunet_noaug": "AttnU-Net, No Aug",
    "exp04_attnunet_aug": "AttnU-Net, Aug",
}


def aggregate_metrics(predictions: dict, ground_truth: dict) -> dict:
    if set(predictions) != set(ground_truth):
        raise ValueError("Prediction and ground-truth sample IDs do not match")

    tp = fp = fn = tn = 0
    for sample_id in predictions:
        counts = confusion_counts(predictions[sample_id], ground_truth[sample_id])
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
        tn += counts[3]

    return {
        "dice": dice_coefficient(tp, fp, fn),
        "iou": iou_score(tp, fp, fn),
        "precision": precision_score(tp, fp),
        "recall": recall_score(tp, fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def load_pipeline(exp_name: str, checkpoint_path: Path, device: torch.device, *, old: bool):
    config = load_experiment_config(exp_name)
    model = build_model(config)
    _load_checkpoint(model, checkpoint_path, device)
    model.to(device).eval()

    if not old:
        return model, build_test_loader(config), config

    old_config = copy.deepcopy(config)
    old_config["data"] = dict(old_config["data"])
    old_config["training"] = dict(old_config["training"])
    old_config["data"]["image_size"] = 256
    old_config["training"]["use_leaf_masking"] = False

    paths = old_config["paths"]
    transform = get_eval_transforms(256, with_leaf=False)
    dataset = GLSDataset(
        Path(paths["split_dir"]) / "test.txt",
        paths["processed_images_dir"],
        paths["lesion_masks_dir"],
        256,
        transform=transform,
        return_id=True,
        leaf_masks_dir=None,
        return_leaf=False,
        apply_leaf_masking=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(old_config["training"].get("batch_size", 8)),
        shuffle=False,
        num_workers=int(old_config["training"].get("num_workers", 0)),
    )
    return model, loader, old_config


def run_inference_with_probs(model, loader, device):
    predictions, ground_truth, probabilities = {}, {}, {}
    threshold = 0.5
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                sample_ids, images, masks = batch
                leaf = None
            elif len(batch) == 4:
                sample_ids, images, masks, leaf = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            images = images.to(device)
            masks = masks.to(device)
            if leaf is not None:
                leaf = leaf.to(device)
                if leaf.dim() == 3:
                    leaf = leaf.unsqueeze(1)

            probs = torch.sigmoid(model(images))
            binary = probs >= threshold
            target = masks >= 0.5
            if leaf is not None:
                leaf_binary = leaf >= 0.5
                binary = binary & leaf_binary
                target = target & leaf_binary

            for index, sample_id in enumerate(sample_ids):
                probabilities[sample_id] = probs[index, 0].cpu().numpy()
                predictions[sample_id] = binary[index, 0].cpu().numpy().astype(np.uint8)
                ground_truth[sample_id] = target[index, 0].cpu().numpy().astype(np.uint8)

    return predictions, ground_truth, probabilities


def load_display_image(sample_id: str, images_dir: Path, size: int) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(images_dir / f"{sample_id}.jpg")).convert("RGB")
    return np.array(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def load_display_mask(sample_id: str, masks_dir: Path, size: int) -> np.ndarray:
    mask = Image.open(masks_dir / f"{sample_id}.png").convert("L")
    mask = mask.resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray, color=(220, 0, 0), alpha=0.45) -> np.ndarray:
    result = image.astype(np.float32).copy()
    color_array = np.asarray(color, dtype=np.float32)
    result[mask > 0] = (1 - alpha) * result[mask > 0] + alpha * color_array
    return np.clip(result, 0, 255).astype(np.uint8)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (size, size), Image.Resampling.NEAREST
    )
    return np.asarray(resized) > 127


def resize_probability(probability: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray((np.clip(probability, 0, 1) * 255).astype(np.uint8))
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR)) / 255.0


def make_comparison_grid(
    exp_name: str,
    old_preds: dict,
    old_probs: dict,
    new_preds: dict,
    new_probs: dict,
    old_ground_truth: dict,
    new_ground_truth: dict,
    config: dict,
    sample_ids: list[str],
    output_path: Path,
) -> None:
    display_size = 256
    images_dir = Path(config["paths"]["processed_images_dir"])
    masks_dir = Path(config["paths"]["lesion_masks_dir"])
    titles = [
        "Image",
        "Ground truth",
        "Old prediction\n(full-image, 256²)",
        "New prediction\n(leaf-masked, 512²)",
        "Old confidence",
        "New confidence",
    ]

    figure = plt.figure(figsize=(len(titles) * 3.2, len(sample_ids) * 3.2 + 0.8))
    grid = gridspec.GridSpec(
        len(sample_ids) + 1,
        len(titles),
        figure=figure,
        hspace=0.08,
        wspace=0.04,
        height_ratios=[0.18] + [1] * len(sample_ids),
    )

    for column, title in enumerate(titles):
        axis = figure.add_subplot(grid[0, column])
        axis.text(0.5, 0.5, title, ha="center", va="center", fontsize=9, fontweight="bold")
        axis.axis("off")

    for row_index, sample_id in enumerate(sample_ids, start=1):
        image = load_display_image(sample_id, images_dir, display_size)
        target = load_display_mask(sample_id, masks_dir, display_size)
        old_prediction = resize_mask(old_preds[sample_id], display_size)
        new_prediction = resize_mask(new_preds[sample_id], display_size)
        old_probability = resize_probability(old_probs[sample_id], display_size)
        new_probability = resize_probability(new_probs[sample_id], display_size)

        old_counts = confusion_counts(old_preds[sample_id], old_ground_truth[sample_id])
        new_counts = confusion_counts(new_preds[sample_id], new_ground_truth[sample_id])
        old_dice = dice_coefficient(*old_counts[:3])
        new_dice = dice_coefficient(*new_counts[:3])

        panels = [
            image,
            overlay(image, target, color=(0, 200, 0)),
            overlay(image, old_prediction),
            overlay(image, new_prediction),
        ]
        for column in range(len(titles)):
            axis = figure.add_subplot(grid[row_index, column])
            axis.axis("off")
            if column == 4:
                axis.imshow(image)
                axis.imshow(old_probability, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
            elif column == 5:
                axis.imshow(image)
                heatmap = axis.imshow(new_probability, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
                if row_index == 1:
                    colorbar = figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
                    colorbar.set_label("confidence", fontsize=7)
                    colorbar.ax.tick_params(labelsize=6)
            else:
                axis.imshow(panels[column])

            if column == 0:
                axis.set_ylabel(sample_id, fontsize=7.5, rotation=0, labelpad=60, va="center", ha="right")
            if column == 2:
                axis.set_title(f"Dice={old_dice:.3f}", fontsize=7.5, color="#dc2626", pad=2)
            if column == 3:
                delta = new_dice - old_dice
                color = "#16a34a" if delta >= 0 else "#dc2626"
                axis.set_title(f"Dice={new_dice:.3f} ({delta:+.3f})", fontsize=7.5, color=color, pad=2)

    figure.suptitle(f"{EXP_LABELS[exp_name]} - Old vs New Pipeline", fontsize=12, fontweight="bold", y=1.002)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    print(f"  saved {output_path}")


def main(old_checkpoints_dir: str, new_checkpoints_dir: str, n: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for experiment in EXPERIMENTS:
        old_checkpoint = Path(old_checkpoints_dir) / f"{experiment}.pt"
        new_checkpoint = Path(new_checkpoints_dir) / f"{experiment}.pt"
        if not old_checkpoint.exists() or not new_checkpoint.exists():
            print(f"[SKIP] missing checkpoint for {experiment}")
            continue

        print(f"\n{experiment}")
        old_model, old_loader, old_config = load_pipeline(experiment, old_checkpoint, device, old=True)
        new_model, new_loader, new_config = load_pipeline(experiment, new_checkpoint, device, old=False)
        old_preds, old_gt, old_probs = run_inference_with_probs(old_model, old_loader, device)
        new_preds, new_gt, new_probs = run_inference_with_probs(new_model, new_loader, device)

        sample_ids = sorted(set(old_gt) & set(new_gt))
        if not sample_ids:
            raise ValueError(f"No common test samples found for {experiment}")
        old_metrics = aggregate_metrics(old_preds, old_gt)
        new_metrics = aggregate_metrics(new_preds, new_gt)

        for pipeline, metrics in [("old (full-img, 256x256)", old_metrics), ("new (leaf-masked, 512x512)", new_metrics)]:
            rows.append({
                "experiment": EXP_LABELS[experiment],
                "pipeline": pipeline,
                **{key: round(metrics[key], 4) for key in ("dice", "iou", "precision", "recall")},
                **{key: metrics[key] for key in ("tp", "fp", "fn", "tn")},
            })

        print(f"  old dice={old_metrics['dice']:.4f}, new dice={new_metrics['dice']:.4f}, delta={new_metrics['dice'] - old_metrics['dice']:+.4f}")
        deltas = []
        for sample_id in sample_ids:
            old_dice = dice_coefficient(*confusion_counts(old_preds[sample_id], old_gt[sample_id])[:3])
            new_dice = dice_coefficient(*confusion_counts(new_preds[sample_id], new_gt[sample_id])[:3])
            deltas.append((abs(new_dice - old_dice), sample_id))
        selected = [sample_id for _, sample_id in sorted(deltas, reverse=True)[:n]]
        make_comparison_grid(
            experiment,
            old_preds,
            old_probs,
            new_preds,
            new_probs,
            old_gt,
            new_gt,
            new_config,
            selected,
            output_dir / f"{experiment}_comparison.png",
        )

    if not rows:
        print("No experiments completed - check checkpoint paths.")
        return

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_dir / "comparison_table.csv", index=False)
    markdown_columns = ["experiment", "pipeline", "dice", "iou", "precision", "recall"]
    markdown_lines = [
        "| " + " | ".join(markdown_columns) + " |",
        "|" + "|".join("---" for _ in markdown_columns) + "|",
    ]
    for _, row in dataframe[markdown_columns].iterrows():
        markdown_lines.append("| " + " | ".join(str(row[column]) for column in markdown_columns) + " |")
    (output_dir / "comparison_table.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"Saved comparison outputs to {output_dir}")
    print(dataframe[markdown_columns].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare old and new pipeline checkpoints.")
    parser.add_argument("--old-checkpoints-dir", required=True)
    parser.add_argument("--new-checkpoints-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    arguments = parser.parse_args()
    main(arguments.old_checkpoints_dir, arguments.new_checkpoints_dir, arguments.n)

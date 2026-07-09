"""Visual inspection helpers (leaf-mask QA, qualitative prediction overlays for
the report, §4.2.13).

Four functions:
    overlay_mask          -- blend a single binary mask onto an RGB image
    plot_prediction_grid  -- grid of Image / GT / Prediction / Error-map rows,
                              for qualitative inspection of model output
    plot_training_curves  -- loss + Dice/IoU over epochs from a trainer.py log
    plot_coverage_scatter -- predicted vs reference leaf-area coverage,
                              straight from evaluate.py's results.json

All functions take/return numpy arrays or matplotlib Figures directly rather
than reading files themselves (except the two convenience file-path
wrappers), so they compose with whatever's already in memory during
evaluation without a round-trip through disk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a binary mask onto an RGB image as a translucent color wash.

    Args:
        image: (H, W, 3) uint8 RGB image.
        mask: (H, W) binary mask, any of {0,1} or {0,255} -- anything > 0
            is treated as foreground.
        color: RGB color for the overlay, e.g. (255, 0, 0) for red.
        alpha: opacity of the overlay in the masked region, in [0, 1].
            0 = mask invisible, 1 = mask fully opaque (image not visible
            underneath it).

    Returns:
        (H, W, 3) uint8 RGB image, same shape as input.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must be (H, W, 3), got {image.shape}")
    mask_bin = np.asarray(mask) > 0
    if mask_bin.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask_bin.shape} doesn't match image shape {image.shape[:2]}")

    image = image.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)

    out = image.copy()
    out[mask_bin] = (1 - alpha) * image[mask_bin] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _error_map(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB visualisation: TP=green, FP=red, FN=yellow, TN=black.
    Lets you see at a glance whether a model's mistakes are false alarms
    (red) or missed lesions (yellow), not just an aggregate Dice number."""
    gt = np.asarray(gt_mask) > 0
    pred = np.asarray(pred_mask) > 0

    h, w = gt.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[gt & pred] = (0, 200, 0)       # TP -- green
    out[~gt & pred] = (220, 0, 0)      # FP -- red
    out[gt & ~pred] = (230, 200, 0)    # FN -- yellow
    # TN stays black
    return out


def plot_prediction_grid(
    images: list[np.ndarray],
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    n: int = 6,
    save_path: str | Path | None = None,
    titles: list[str] | None = None,
):
    """
    Grid for qualitative inspection: one row per sample, four columns --
    RGB image | ground truth overlay | prediction overlay | error map.

    Args:
        images: list of (H, W, 3) uint8 RGB arrays.
        gt_masks, pred_masks: lists of (H, W) binary masks, same length/order
            as images.
        n: number of samples to show (first n, in the order given -- pass
            already-selected/shuffled lists in if you want a specific
            subset, e.g. the worst-Dice samples from results.json).
        save_path: if given, saves the figure there (any matplotlib-
            supported extension: .png, .pdf, ...). Always returns the
            Figure regardless, so it can also be displayed inline.
        titles: optional per-row label (e.g. sample_id + dice score),
            same length as images. Defaults to "Sample {i}".

    Returns:
        matplotlib.figure.Figure
    """
    if not (len(images) == len(gt_masks) == len(pred_masks)):
        raise ValueError(
            f"images ({len(images)}), gt_masks ({len(gt_masks)}), and "
            f"pred_masks ({len(pred_masks)}) must be the same length"
        )
    n = min(n, len(images))
    if n == 0:
        raise ValueError("nothing to plot -- images list is empty")

    col_titles = ["Image", "Ground Truth", "Prediction", "Error Map\n(green=TP, red=FP, yellow=FN)"]
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # keep 2D indexing consistent for n=1

    for row in range(n):
        img, gt, pred = images[row], gt_masks[row], pred_masks[row]
        row_label = titles[row] if titles else f"Sample {row}"

        axes[row, 0].imshow(img)
        axes[row, 1].imshow(overlay_mask(img, gt, color=(0, 200, 0)))
        axes[row, 2].imshow(overlay_mask(img, pred, color=(220, 0, 0)))
        axes[row, 3].imshow(_error_map(gt, pred))

        axes[row, 0].set_ylabel(row_label, fontsize=10)
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=11)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(log_csv_path: str | Path, save_path: str | Path | None = None):
    """
    Plot train/val loss and val Dice/IoU over epochs from a trainer.py log
    (outputs/logs/<experiment>.csv -- columns: epoch, train_loss, val_loss,
    tp, fp, fn, tn, dice, iou, precision, recall).

    Returns:
        matplotlib.figure.Figure
    """
    import pandas as pd

    df = pd.read_csv(log_csv_path)
    required = {"epoch", "train_loss", "val_loss", "dice", "iou"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"log CSV is missing expected columns: {missing}")

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_loss.plot(df["epoch"], df["train_loss"], label="train loss", marker="o", markersize=3)
    ax_loss.plot(df["epoch"], df["val_loss"], label="val loss", marker="o", markersize=3)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("BCE+Dice loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_metric.plot(df["epoch"], df["dice"], label="val Dice", marker="o", markersize=3)
    ax_metric.plot(df["epoch"], df["iou"], label="val IoU", marker="o", markersize=3)
    ax_metric.set_xlabel("epoch")
    ax_metric.set_ylabel("score")
    ax_metric.set_ylim(0, 1)
    ax_metric.set_title("Validation segmentation metrics")
    ax_metric.legend()
    ax_metric.grid(alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_multiple_predictions(
    images_dir: str | Path,
    masks_dir: str | Path,
    predictions: dict[str, np.ndarray],
    sample_ids: list[str],
    out_dir: str | Path,
    image_size: int = 256,
) -> None:
    """
    Save one qualitative prediction figure per sample to out_dir/<sample_id>.png.

    Designed to plug directly into src.evaluation.evaluate.run_inference()'s
    output: `predictions` is a dict[sample_id -> (image_size, image_size)
    binary array], already thresholded, at the model's OUTPUT resolution.

    Important: src/data/generate_masks.py does NOT resize images when
    writing to images_dir/masks_dir -- they're saved at native/original
    resolution (resizing only happens inside get_eval_transforms /
    get_train_transforms, at Dataset.__getitem__ time). So this function
    resizes the raw image/mask to image_size itself before overlaying,
    using the same nearest-neighbor mask interpolation as
    src/data/augmentations.py (Albumentations' default), so mask edges
    stay exactly binary rather than picking up interpolation artifacts.
    Without this step, the raw image and the (image_size, image_size)
    prediction array would have mismatched shapes.

    Args:
        images_dir, masks_dir: directories of RAW (non-resized) processed
            images/masks, e.g. config["paths"]["processed_images_dir"] /
            ["lesion_masks_dir"].
        predictions: sample_id -> binary prediction array, as returned by
            src.evaluation.evaluate.run_inference().
        sample_ids: which samples to save (must all be keys of `predictions`).
        out_dir: directory to write <sample_id>.png into (created if needed).
        image_size: must match whatever image_size the model/loader that
            produced `predictions` used -- mismatches raise a clear error
            below rather than silently misaligning image and prediction.
    """
    import albumentations as A
    from PIL import Image, ImageOps

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resize = A.Compose([A.Resize(image_size, image_size)])  # mask_interpolation defaults to NEAREST

    for sample_id in sample_ids:
        if sample_id not in predictions:
            raise KeyError(
                f"No prediction for sample_id={sample_id!r} -- check it's actually in the "
                f"split/loader that `predictions` was built from"
            )

        image_path = Path(images_dir) / f"{sample_id}.jpg"
        mask_path = Path(masks_dir) / f"{sample_id}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        raw_image = np.array(ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"), dtype=np.uint8)
        raw_mask = (np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0).astype(np.uint8)

        resized = resize(image=raw_image, mask=raw_mask)
        image, gt_mask = resized["image"], resized["mask"]
        pred_mask = np.asarray(predictions[sample_id])

        if pred_mask.shape != (image_size, image_size):
            raise ValueError(
                f"prediction for {sample_id!r} has shape {pred_mask.shape}, expected "
                f"({image_size}, {image_size}) -- `image_size` passed to this function must match "
                f"whatever the model/loader that produced `predictions` actually used"
            )

        fig = plot_prediction_grid([image], [gt_mask], [pred_mask], n=1, titles=[sample_id])
        fig.savefig(out_dir / f"{sample_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_confidence_heatmap(
    image: np.ndarray,
    probs: np.ndarray,
    save_path: str | Path | None = None,
    title: str | None = None,
    alpha: float = 0.6,
):
    """
    Overlay the model's raw per-pixel confidence (post-sigmoid probability,
    NOT thresholded) as a heatmap, with a colorbar. Requested by supervisor
    review: shows where the model is confident vs uncertain, rather than
    collapsing everything to a binary yes/no lesion call.

    Args:
        image: (H, W, 3) uint8 RGB image.
        probs: (H, W) float array in [0, 1] -- raw sigmoid output, same
            spatial size as image. NOT the binarized prediction.
        alpha: opacity of the heatmap overlay, in [0, 1].

    Returns:
        matplotlib.figure.Figure
    """
    probs = np.asarray(probs)
    if probs.shape != image.shape[:2]:
        raise ValueError(f"probs shape {probs.shape} doesn't match image shape {image.shape[:2]}")
    if probs.min() < 0 or probs.max() > 1:
        raise ValueError(
            f"probs must be in [0, 1] (raw sigmoid output) -- got range "
            f"[{probs.min():.3f}, {probs.max():.3f}]. Did you pass a binarized "
            f"mask by mistake?"
        )

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(image)
    im = ax.imshow(probs, cmap="inferno", alpha=alpha, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("predicted lesion probability")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_qualitative_panel(
    image: np.ndarray,
    leaf_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    pred_probs: np.ndarray,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """
    Five-panel qualitative figure for one sample: Image | Leaf region |
    Ground truth lesions | Predicted lesions (binary) | Prediction
    confidence (heatmap). Matches supervisor review request: show the leaf
    segmentation, the lesion segmentation, and a confidence heatmap
    side by side for the same sample.

    Args:
        image: (H, W, 3) uint8 RGB image.
        leaf_mask: (H, W) binary mask of the SAM/YOLO-extracted leaf region.
        gt_mask, pred_mask: (H, W) binary lesion masks.
        pred_probs: (H, W) float array in [0, 1] -- raw sigmoid output
            (same as plot_confidence_heatmap's `probs`), NOT the binarized
            pred_mask.
        title: optional figure-level title (e.g. sample_id + dice score).

    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))

    axes[0].imshow(image)
    axes[0].set_title("Image", fontsize=11)

    axes[1].imshow(overlay_mask(image, leaf_mask, color=(0, 150, 255)))
    axes[1].set_title("Leaf region", fontsize=11)

    axes[2].imshow(overlay_mask(image, gt_mask, color=(0, 200, 0)))
    axes[2].set_title("Ground truth lesions", fontsize=11)

    axes[3].imshow(overlay_mask(image, pred_mask, color=(220, 0, 0)))
    axes[3].set_title("Predicted lesions", fontsize=11)

    axes[4].imshow(image)
    im = axes[4].imshow(pred_probs, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
    axes[4].set_title("Prediction confidence", fontsize=11)
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_coverage_scatter(per_image_results: list[dict], save_path: str | Path | None = None):
    """
    Scatter of predicted vs reference leaf-area GLS coverage, one point per
    test sample -- visual companion to evaluate.py's coverage_mae/rmse/
    pearson_r summary numbers (§4.2.10). Points on the y=x line are
    perfect predictions.

    Args:
        per_image_results: the "per_image" list from a results.json written
            by src/evaluation/evaluate.py. Samples without a
            "predicted_coverage"/"reference_coverage" key (i.e. no leaf mask
            available yet -- see evaluate.py's graceful-degradation
            behaviour) are silently skipped rather than erroring.

    Returns:
        matplotlib.figure.Figure
    """
    pred = [r["predicted_coverage"] for r in per_image_results if "predicted_coverage" in r]
    ref = [r["reference_coverage"] for r in per_image_results if "reference_coverage" in r]
    if len(pred) == 0:
        raise ValueError(
            "no samples with coverage data in per_image_results -- "
            "leaf masks may not be available yet for this experiment"
        )

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(ref, pred, alpha=0.7, edgecolors="k", linewidths=0.5)

    lim_max = max(max(pred, default=0), max(ref, default=0)) * 1.05 + 1
    ax.plot([0, lim_max], [0, lim_max], "--", color="gray", linewidth=1, label="perfect prediction (y=x)")

    ax.set_xlabel("reference coverage (%, from ground-truth mask)")
    ax.set_ylabel("predicted coverage (%, from model mask)")
    ax.set_title(f"Leaf-area GLS coverage (n={len(pred)})")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
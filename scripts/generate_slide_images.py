"""Generate every image needed for the supervisor presentation slides.

Maps directly to every INSERT placeholder in GLS_Supervisor_Visual_Template.pptx.
Run this once after training and evaluation are complete, then drop each output
image into the corresponding slide placeholder.

Usage:
    python scripts/generate_slide_images.py \
        --experiment exp01_unet_noaug \
        --old-checkpoints-dir /path/to/old/checkpoints \
        --new-checkpoints-dir /path/to/new/checkpoints

All outputs go to outputs/slide_images/. A manifest (slide_image_manifest.txt)
is printed and saved there listing which file goes on which slide.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps
import albumentations as A

from src.data.augmentations import get_eval_transforms
from src.data.dataset import GLSDataset
from src.evaluation.evaluate import (
    load_experiment_config,
    build_model,
    _load_checkpoint,
    build_test_loader,
)
from src.training.metrics import confusion_counts, dice_coefficient

OUT_DIR = Path("outputs/slide_images")

EXPERIMENTS = [
    "exp01_unet_noaug",
    "exp02_unet_aug",
    "exp03_attnunet_noaug",
    "exp04_attnunet_aug",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_rgb(path: Path, size: int = 512) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.array(img.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def load_mask(path: Path, size: int = 512) -> np.ndarray:
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.array(m) > 0).astype(np.uint8)


def overlay(img, mask, color=(220, 0, 0), alpha=0.45):
    out = img.astype(np.float32).copy()
    out[mask > 0] = (1 - alpha) * out[mask > 0] + alpha * np.array(color)
    return np.clip(out, 0, 255).astype(np.uint8)


def save(fig, name: str) -> Path:
    p = OUT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def run_inference_with_probs(model, loader, device):
    model.eval()
    preds, gts, probs = {}, {}, {}
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                ids, imgs, masks = batch
                leaf = None
            else:
                ids, imgs, masks, leaf = batch
            imgs = imgs.to(device)
            logits = model(imgs)
            prob = torch.sigmoid(logits)
            if leaf is not None:
                leaf = leaf.to(device)
                if leaf.dim() == 3:
                    leaf = leaf.unsqueeze(1)
            for i, sid in enumerate(ids):
                p = prob[i, 0].cpu().numpy()
                b = (p >= 0.5).astype(np.uint8)
                if leaf is not None:
                    lf = leaf[i, 0].cpu().numpy()
                    b = b & (lf >= 0.5)
                    gt = masks[i, 0].cpu().numpy().astype(np.uint8) & (lf >= 0.5).astype(np.uint8)
                else:
                    gt = masks[i, 0].cpu().numpy().astype(np.uint8)
                probs[sid] = p
                preds[sid] = b
                gts[sid] = gt
    return preds, gts, probs


def pick_samples(preds, gts, n_best=2, n_worst=2, n_high_cov=2):
    """Pick interesting samples: best dice, worst dice, highest GT coverage."""
    scored = []
    for sid in preds:
        tp, fp, fn, tn = confusion_counts(preds[sid], gts[sid])
        d = dice_coefficient(tp, fp, fn)
        cov = gts[sid].mean() * 100
        scored.append((sid, d, cov))
    scored.sort(key=lambda x: x[1])
    worst = [s[0] for s in scored[:n_worst]]
    best  = [s[0] for s in scored[-n_best:]]
    scored.sort(key=lambda x: x[2], reverse=True)
    high  = [s[0] for s in scored[:n_high_cov] if s[0] not in best + worst]
    return list(dict.fromkeys(worst + best + high))  # dedup, preserve order


def normalize_sample_id(sid: str | None) -> str | None:
    if not sid:
        return None
    s = str(sid).strip()
    if s.lower().endswith(".jpg") or s.lower().endswith(".png"):
        s = Path(s).stem
    return s


def get_prediction_for_sample(
    sid: str,
    preds: dict[str, np.ndarray],
    probs: dict[str, np.ndarray],
    gts: dict[str, np.ndarray],
    config: dict,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve existing prediction or run on-the-fly inference for any sample ID."""
    sid = normalize_sample_id(sid)
    if sid in preds:
        return preds[sid], gts[sid], probs.get(sid, np.zeros_like(preds[sid], dtype=np.float32))

    imgs_dir = Path(config["paths"]["processed_images_dir"])
    masks_dir = Path(config["paths"]["lesion_masks_dir"])
    leaf_dir = Path(config["paths"]["leaf_masks_dir"]) if config["paths"].get("leaf_masks_dir") else None
    size = int(config["data"]["image_size"])
    use_leaf_masking = bool(config["training"].get("use_leaf_masking", False))

    img_path = imgs_dir / f"{sid}.jpg"
    mask_path = masks_dir / f"{sid}.png"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found for sample '{sid}': {img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Lesion mask not found for sample '{sid}': {mask_path}")

    img_rgb = load_rgb(img_path, size)
    gt_mask = load_mask(mask_path, size)
    lmask = (
        load_mask(leaf_dir / f"{sid}.png", size)
        if (leaf_dir and (leaf_dir / f"{sid}.png").exists())
        else np.ones((size, size), dtype=np.uint8)
    )

    if model is not None and device is not None:
        inp = img_rgb.astype(np.float32) / 255.0
        if use_leaf_masking:
            inp = inp * lmask[:, :, None]

        tensor_in = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        model.eval()
        with torch.no_grad():
            logits = model(tensor_in)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob >= 0.5).astype(np.uint8)
            if use_leaf_masking:
                pred = pred & (lmask > 0)
                gt_mask = gt_mask & (lmask > 0)
    else:
        prob = np.zeros((size, size), dtype=np.float32)
        pred = np.zeros((size, size), dtype=np.uint8)

    return pred, gt_mask, prob


# ──────────────────────────────────────────────────────────────────────────────
# Image generators — one function per INSERT placeholder group
# ──────────────────────────────────────────────────────────────────────────────

def gen_slide1_images(
    preds, probs, gts, config, manifest, sample_id: str | None = None, model=None, device=None
):
    """Slide 1: Best qualitative result + best comparison result."""
    if sample_id:
        sid = normalize_sample_id(sample_id)
        pred, gt, prob = get_prediction_for_sample(sid, preds, probs, gts, config, model, device)
    else:
        ids = pick_samples(preds, gts, n_best=1, n_worst=0, n_high_cov=0)
        sid = ids[0] if ids else list(preds.keys())[0]
        pred, gt, prob = preds[sid], gts[sid], probs.get(sid, np.zeros_like(preds[sid], dtype=np.float32))

    imgs_dir = Path(config["paths"]["processed_images_dir"])
    size = int(config["data"]["image_size"])

    img = load_rgb(imgs_dir / f"{sid}.jpg", size)
    dice = dice_coefficient(*confusion_counts(pred, gt)[:3])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img); axes[0].set_title("Image", fontsize=11, fontweight="bold")
    axes[1].imshow(overlay(img, gt, (0, 200, 0))); axes[1].set_title("Ground truth", fontsize=11, fontweight="bold")
    axes[2].imshow(overlay(img, pred, (220, 0, 0))); axes[2].set_title(f"Prediction  dice={dice:.3f}", fontsize=11)
    axes[3].imshow(img); im = axes[3].imshow(prob, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
    axes[3].set_title("Confidence heatmap", fontsize=11); fig.colorbar(im, ax=axes[3], fraction=0.046)
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    p = save(fig, "slide01_best_qualitative.png")
    manifest.append(f"SLIDE 1 | INSERT: Best qualitative result | {p}")
    print(f"  saved {p.name} (sample: {sid}, dice: {dice:.3f})")


def gen_slide2_research_image(config, manifest, sample_id: str | None = None):
    """Slide 2: Example maize leaf showing GLS lesions."""
    imgs_dir = Path(config["paths"]["processed_images_dir"])
    masks_dir = Path(config["paths"]["lesion_masks_dir"])
    size = int(config["data"]["image_size"])

    if sample_id:
        best_sid = normalize_sample_id(sample_id)
        gt = load_mask(masks_dir / f"{best_sid}.png", size)
        best_cov = gt.mean() * 100
    else:
        split_file = Path(config["paths"]["split_dir"]) / "test.txt"
        ids = split_file.read_text().strip().splitlines()

        # pick highest coverage sample as the most visually striking
        best_sid, best_cov = ids[0], 0
        for sid in ids:
            m = load_mask(masks_dir / f"{sid}.png", size)
            cov = m.mean() * 100
            if cov > best_cov:
                best_cov, best_sid = cov, sid
        gt = load_mask(masks_dir / f"{best_sid}.png", size)

    img = load_rgb(imgs_dir / f"{best_sid}.jpg", size)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(img); axes[0].set_title("Maize leaf — GLS visible", fontweight="bold")
    axes[1].imshow(overlay(img, gt, (0, 200, 0))); axes[1].set_title(f"Lesion mask overlay  coverage={best_cov:.1f}%", fontweight="bold")
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    p = save(fig, "slide02_research_image.png")
    manifest.append(f"SLIDE 2 | INSERT: Research image | {p}")
    print(f"  saved {p.name}")


def gen_slide3_dataset_triplet(config, manifest, sample_id: str | None = None):
    """Slide 3: Original image | Leaf mask | Lesion mask."""
    imgs_dir  = Path(config["paths"]["processed_images_dir"])
    masks_dir = Path(config["paths"]["lesion_masks_dir"])
    leaf_dir  = Path(config["paths"]["leaf_masks_dir"])
    size = int(config["data"]["image_size"])

    if sample_id:
        sid = normalize_sample_id(sample_id)
    else:
        split_file = Path(config["paths"]["split_dir"]) / "test.txt"
        ids = split_file.read_text().strip().splitlines()
        sid = ids[0]
        for s in ids:
            m = load_mask(masks_dir / f"{s}.png", size)
            if m.mean() * 100 > 5:
                sid = s
                break

    img  = load_rgb(imgs_dir / f"{sid}.jpg", size)
    lmask = load_mask(leaf_dir / f"{sid}.png", size) if (leaf_dir / f"{sid}.png").exists() else np.ones((size, size), dtype=np.uint8)
    gmask = load_mask(masks_dir / f"{sid}.png", size)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles = ["Original RGB", "Leaf mask", "Lesion mask (GT)"]
    images = [img, overlay(img, lmask, (0, 150, 255)), overlay(img, gmask, (0, 200, 0))]
    for ax, im_, t in zip(axes, images, titles):
        ax.imshow(im_); ax.set_title(t, fontweight="bold"); ax.axis("off")
    fig.tight_layout()
    p = save(fig, "slide03_dataset_triplet.png")
    manifest.append(f"SLIDE 3 | INSERT: ORIGINAL IMAGE + LEAF MASK + LESION MASK | {p}")
    print(f"  saved {p.name}")


def gen_slide4_masking(config, manifest, sample_id: str | None = None):
    """Slide 4: Before vs after leaf masking + mask quality check."""
    imgs_dir  = Path(config["paths"]["processed_images_dir"])
    leaf_dir  = Path(config["paths"]["leaf_masks_dir"])
    size = int(config["data"]["image_size"])
    if sample_id:
        sid = normalize_sample_id(sample_id)
    else:
        split_file = Path(config["paths"]["split_dir"]) / "test.txt"
        sid = split_file.read_text().strip().splitlines()[0]

    img   = load_rgb(imgs_dir / f"{sid}.jpg", size)
    lmask = load_mask(leaf_dir / f"{sid}.png", size) if (leaf_dir / f"{sid}.png").exists() else np.ones((size, size), dtype=np.uint8)
    masked = img * lmask[:, :, None]

    # Before vs after
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(img); axes[0].set_title("Before masking (full image)", fontweight="bold")
    axes[1].imshow(masked); axes[1].set_title("After masking (leaf only)", fontweight="bold")
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    p = save(fig, "slide04_before_after_masking.png")
    manifest.append(f"SLIDE 4 | INSERT: Before vs after masking | {p}")
    print(f"  saved {p.name}")

    # Mask quality overlay
    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    ax2.imshow(overlay(img, lmask, (0, 150, 255)))
    ax2.set_title("Leaf mask overlay (blue = leaf)", fontweight="bold")
    ax2.axis("off")
    fig2.tight_layout()
    p2 = save(fig2, "slide04_mask_quality.png")
    manifest.append(f"SLIDE 4 | INSERT: Mask quality check | {p2}")
    print(f"  saved {p2.name}")


def gen_slide5_unet_diagram(manifest):
    """Slide 5: U-Net architecture diagram."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    colors = {"enc": "2D6A4F", "bot": "1A1A2E", "dec": "40916C", "skip": "74C69D"}
    def hexc(h): return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

    enc = [(0.3, 1.4, 1.6, 2.2), (0.3, 0.7, 1.6, 1.0),
           (0.3, 0.1, 1.6, 0.4)]
    dec = [(11.1, 1.4, 1.6, 2.2), (11.1, 0.7, 1.6, 1.0),
           (11.1, 0.1, 1.6, 0.4)]
    bot = [(5.7, 0.1, 1.6, 0.7)]

    for x, y, w, h in enc:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=hexc("2D6A4F"), ec="white", lw=1.2, zorder=3))
    for x, y, w, h in dec:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=hexc("40916C"), ec="white", lw=1.2, zorder=3))
    for x, y, w, h in bot:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=hexc("1A1A2E"), ec="white", lw=1.2, zorder=3))

    labels = [("Encoder", 1.1, 2.5), ("Encoder", 1.1, 1.2), ("Encoder", 1.1, 0.35),
              ("Bottleneck", 6.5, 0.45),
              ("Decoder", 11.9, 2.5), ("Decoder", 11.9, 1.2), ("Decoder", 11.9, 0.35)]
    for lbl, lx, ly in labels:
        ax.text(lx, ly, lbl, ha="center", va="center", fontsize=8.5, color="white", fontweight="bold", zorder=4)

    # Skip connections
    for y1, y2 in [(2.5, 2.5), (1.2, 1.2), (0.35, 0.35)]:
        ax.annotate("", xy=(11.1, y1), xytext=(1.9, y2),
                    arrowprops=dict(arrowstyle="->", color=hexc("74C69D"), lw=1.5,
                                    connectionstyle="arc3,rad=-0.25"), zorder=2)
    ax.text(6.5, 3.5, "Skip connections", ha="center", fontsize=9, color=hexc("74C69D"), fontweight="bold")

    # Arrows down/up
    for x, y in [(1.1, 1.35), (1.1, 0.65)]: ax.annotate("", xy=(x, y-0.2), xytext=(x, y), arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    for x, y in [(11.9, 1.35), (11.9, 0.65)]: ax.annotate("", xy=(x, y+0.2), xytext=(x, y), arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    # Input / Output labels
    ax.text(0.15, 2.5, "Input\n3×512×512", ha="center", fontsize=8, va="center")
    ax.text(12.85, 2.5, "Output\n1×512×512\n(logits)", ha="center", fontsize=8, va="center")
    ax.set_title("U-Net Architecture (depth=4, base_filters=64)", fontsize=12, fontweight="bold", pad=8)
    fig.tight_layout()
    p = save(fig, "slide05_unet_diagram.png")
    manifest.append(f"SLIDE 5 | INSERT: U-Net architecture diagram | {p}")
    print(f"  saved {p.name}")


def gen_slide6_training_curves(experiment: str, manifest):
    """Slide 6: Training curve (loss + Dice by epoch)."""
    log_path = Path("outputs/logs") / f"{experiment}.csv"
    if not log_path.exists():
        print(f"  [SKIP] no log found: {log_path}")
        return
    import pandas as pd
    df = pd.read_csv(log_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(df["epoch"], df["train_loss"], label="train loss", marker="o", ms=3)
    ax1.plot(df["epoch"], df["val_loss"],   label="val loss",   marker="o", ms=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE+Dice loss")
    ax1.set_title("Loss", fontweight="bold"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(df["epoch"], df["dice"], label="val Dice", marker="o", ms=3, color="green")
    ax2.plot(df["epoch"], df["iou"],  label="val IoU",  marker="o", ms=3, color="teal")
    ax2.set_ylim(0, 1); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.set_title("Validation metrics", fontweight="bold"); ax2.legend(); ax2.grid(alpha=0.3)
    best_epoch = int(df.loc[df["dice"].idxmax(), "epoch"])
    best_dice  = df["dice"].max()
    ax2.axvline(best_epoch, color="red", ls="--", lw=1, label=f"best epoch {best_epoch}")
    ax2.legend()
    fig.suptitle(f"Training curves — {experiment}", fontweight="bold", fontsize=12)
    fig.tight_layout()
    p = save(fig, f"slide06_training_curve_{experiment}.png")
    manifest.append(f"SLIDE 6 | INSERT: Training curve ({experiment}) | {p}")
    print(f"  saved {p.name}")


def gen_slide7_baseline_grid(preds, probs, gts, config, manifest):
    """Slide 7: Baseline qualitative grid (best/worst/high-coverage samples)."""
    sample_ids = pick_samples(preds, gts)
    imgs_dir  = Path(config["paths"]["processed_images_dir"])
    masks_dir = Path(config["paths"]["lesion_masks_dir"])
    size = int(config["data"]["image_size"])
    n = len(sample_ids)

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1: axes = axes[np.newaxis, :]
    col_titles = ["Image", "Ground truth", "Prediction", "Confidence"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontweight="bold", fontsize=10)

    for ri, sid in enumerate(sample_ids):
        img  = load_rgb(imgs_dir / f"{sid}.jpg", size)
        gt   = load_mask(masks_dir / f"{sid}.png", size)
        pred = preds[sid]
        prob = probs[sid]
        d = dice_coefficient(*confusion_counts(pred, gt)[:3])
        panels = [img, overlay(img, gt, (0, 200, 0)), overlay(img, pred, (220, 0, 0)), None]
        for ci in range(4):
            axes[ri, ci].axis("off")
            if ci < 3:
                axes[ri, ci].imshow(panels[ci])
            else:
                axes[ri, ci].imshow(img)
                im = axes[ri, ci].imshow(prob, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
            if ci == 0:
                axes[ri, ci].set_ylabel(f"{sid}\ndice={d:.3f}", fontsize=8, rotation=0, labelpad=55, va="center")
    fig.tight_layout()
    p = save(fig, "slide07_baseline_grid.png")
    manifest.append(f"SLIDE 7 | INSERT: Baseline qualitative grid | {p}")
    print(f"  saved {p.name}")


def gen_slide8_leafmasked_triple(
    preds, probs, gts, config, manifest, sample_id: str | None = None, model=None, device=None
):
    """Slide 8: Leaf-masked input | prediction | ground truth."""
    if sample_id:
        sid = normalize_sample_id(sample_id)
        pred, gt, prob = get_prediction_for_sample(sid, preds, probs, gts, config, model, device)
    else:
        sample_ids = pick_samples(preds, gts, n_best=1, n_worst=0, n_high_cov=0)
        sid = sample_ids[0] if sample_ids else list(preds.keys())[0]
        pred, gt, prob = preds[sid], gts[sid], probs.get(sid, np.zeros_like(preds[sid], dtype=np.float32))

    imgs_dir  = Path(config["paths"]["processed_images_dir"])
    leaf_dir  = Path(config["paths"]["leaf_masks_dir"])
    size = int(config["data"]["image_size"])

    img   = load_rgb(imgs_dir / f"{sid}.jpg", size)
    lmask = load_mask(leaf_dir / f"{sid}.png", size) if (leaf_dir / f"{sid}.png").exists() else np.ones((size, size), dtype=np.uint8)
    masked_img = img * lmask[:, :, None]

    dice = dice_coefficient(*confusion_counts(pred, gt)[:3])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles = ["Leaf-masked input\n(what model sees)", f"Prediction\ndice={dice:.3f}", "Ground truth"]
    images = [masked_img, overlay(img, pred, (220, 0, 0)), overlay(img, gt, (0, 200, 0))]
    for ax, im_, t in zip(axes, images, titles):
        ax.imshow(im_); ax.set_title(t, fontweight="bold"); ax.axis("off")
    fig.tight_layout()
    p = save(fig, "slide08_leafmasked_triple.png")
    manifest.append(f"SLIDE 8 | INSERT: Leaf-masked input + Prediction + Ground truth | {p}")
    print(f"  saved {p.name} (sample: {sid}, dice: {dice:.3f})")


def gen_slide10_qualitative_comparison(
    old_preds, old_probs, new_preds, new_probs, gts,
    config, manifest, n=4
):
    """Slide 10: Old vs new pipeline qualitative comparison grid."""
    # pick samples with biggest dice change
    deltas = []
    for sid in old_preds:
        od = dice_coefficient(*confusion_counts(old_preds[sid], gts[sid])[:3])
        nd = dice_coefficient(*confusion_counts(new_preds[sid], gts[sid])[:3])
        deltas.append((abs(nd - od), sid, od, nd))
    deltas.sort(reverse=True)
    selected = [d[1] for d in deltas[:n]]

    imgs_dir = Path(config["paths"]["processed_images_dir"])
    masks_dir = Path(config["paths"]["lesion_masks_dir"])
    size = int(config["data"]["image_size"])

    col_titles = ["Image", "Ground truth", "Old prediction\n(256², full-img)",
                  "New prediction\n(512², leaf-masked)", "Old confidence", "New confidence"]
    ncols = len(col_titles)
    nr = len(selected)

    fig, axes = plt.subplots(nr, ncols, figsize=(ncols * 3, nr * 3))
    if nr == 1: axes = axes[np.newaxis, :]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=8.5, fontweight="bold")

    def rsz_mask(m): return np.array(Image.fromarray((m*255).astype(np.uint8)).resize((size,size), Image.NEAREST)) > 127
    def rsz_prob(p): return np.array(Image.fromarray((np.clip(p,0,1)*255).astype(np.uint8)).resize((size,size), Image.BILINEAR)) / 255.

    for ri, sid in enumerate(selected):
        img  = load_rgb(imgs_dir / f"{sid}.jpg", size)
        gt   = load_mask(masks_dir / f"{sid}.png", size)
        op   = rsz_mask(old_preds[sid]); np_  = rsz_mask(new_preds[sid])
        oh   = rsz_prob(old_probs[sid]); nh   = rsz_prob(new_probs[sid])
        od   = dice_coefficient(*confusion_counts(op, gt)[:3])
        nd   = dice_coefficient(*confusion_counts(np_, gt)[:3])
        delta = nd - od

        panels = [img, overlay(img, gt, (0,200,0)), overlay(img, op, (220,0,0)), overlay(img, np_, (220,0,0)), None, None]
        for ci in range(ncols):
            axes[ri, ci].axis("off")
            if ci < 4:
                axes[ri, ci].imshow(panels[ci])
            elif ci == 4:
                axes[ri, ci].imshow(img)
                axes[ri, ci].imshow(oh, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
            else:
                axes[ri, ci].imshow(img)
                im = axes[ri, ci].imshow(nh, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
            if ci == 0:
                axes[ri, ci].set_ylabel(sid, fontsize=7.5, rotation=0, labelpad=50, va="center")
            if ci == 2:
                axes[ri, ci].set_title(f"dice={od:.3f}", fontsize=7.5, color="#dc2626", pad=2)
            if ci == 3:
                col = "#16a34a" if delta >= 0 else "#dc2626"
                axes[ri, ci].set_title(f"dice={nd:.3f} ({delta:+.3f})", fontsize=7.5, color=col, pad=2)

    fig.tight_layout()
    p = save(fig, "slide10_qualitative_comparison.png")
    manifest.append(f"SLIDE 10 | INSERT: Old vs new qualitative comparison | {p}")
    print(f"  saved {p.name}")


def gen_slide11_coverage(
    preds, probs, gts, config, manifest, sample_id: str | None = None, model=None, device=None
):
    """Slide 11: Leaf mask | lesion prediction | coverage overlay."""
    if sample_id:
        sid = normalize_sample_id(sample_id)
        pred, gt, prob = get_prediction_for_sample(sid, preds, probs, gts, config, model, device)
    else:
        sample_ids = pick_samples(preds, gts, n_best=1, n_worst=0, n_high_cov=1)
        sid = sample_ids[-1] if sample_ids else list(preds.keys())[0]
        pred, gt, prob = preds[sid], gts[sid], probs.get(sid, np.zeros_like(preds[sid], dtype=np.float32))

    imgs_dir  = Path(config["paths"]["processed_images_dir"])
    leaf_dir  = Path(config["paths"]["leaf_masks_dir"])
    size = int(config["data"]["image_size"])

    img   = load_rgb(imgs_dir / f"{sid}.jpg", size)
    lmask = load_mask(leaf_dir / f"{sid}.png", size) if (leaf_dir / f"{sid}.png").exists() else np.ones((size, size), dtype=np.uint8)

    leaf_px   = int(lmask.sum())
    lesion_px = int((pred & lmask).sum())
    coverage  = float(lesion_px / leaf_px * 100.0) if leaf_px > 0 else 0.0

    combined = overlay(overlay(img, lmask, (0,150,255), 0.2), pred & lmask, (220,0,0), 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(overlay(img, lmask, (0,150,255))); axes[0].set_title("Leaf mask (blue)", fontweight="bold")
    axes[1].imshow(overlay(img, pred, (220,0,0))); axes[1].set_title("Lesion prediction (red)", fontweight="bold")
    axes[2].imshow(combined)
    axes[2].set_title(f"Coverage = {lesion_px:,} ÷ {leaf_px:,} × 100 = {coverage:.1f}%", fontweight="bold", fontsize=9)
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    p = save(fig, "slide11_coverage.png")
    manifest.append(f"SLIDE 11 | INSERT: Leaf mask + Lesion prediction + Coverage visual | {p}")
    print(f"  saved {p.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(
    experiment: str,
    old_ckpt_dir: str | None,
    new_ckpt_dir: str | None,
    n: int,
    slide1_sample_id: str | None = None,
    slide2_sample_id: str | None = None,
    slide3_sample_id: str | None = None,
    slide4_sample_id: str | None = None,
    slide8_sample_id: str | None = None,
    slide11_sample_id: str | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    # ── Load new (current) checkpoint ──────────────────────────────────────────
    config = load_experiment_config(experiment)
    new_ckpt = Path(config["paths"]["checkpoints_dir"]) / f"{experiment}.pt"
    if not new_ckpt.exists() and new_ckpt_dir:
        new_ckpt = Path(new_ckpt_dir) / f"{experiment}.pt"

    model = None
    if not new_ckpt.exists():
        print(f"No checkpoint found for {experiment} — skipping inference-dependent slides.")
        new_preds = new_probs = new_gts = {}
    else:
        model = build_model(config)
        _load_checkpoint(model, new_ckpt, device)
        model.to(device).eval()
        loader = build_test_loader(config)
        new_preds, new_gts, new_probs = run_inference_with_probs(model, loader, device)
        print(f"New inference done: {len(new_preds)} test samples")

    # ── Load old checkpoint (optional) ────────────────────────────────────────
    old_preds = old_probs = old_gts = {}
    if old_ckpt_dir:
        old_ckpt = Path(old_ckpt_dir) / f"{experiment}.pt"
        if old_ckpt.exists():
            # temporarily swap leaf masking off for old checkpoint evaluation
            import copy
            old_config = copy.deepcopy(config)
            old_config["training"]["use_leaf_masking"] = False
            old_config["data"]["image_size"] = 256
            old_model = build_model(old_config)
            _load_checkpoint(old_model, old_ckpt, device)
            old_model.to(device).eval()

            # build a plain (non-leaf-masked) test loader for the old checkpoint
            from torch.utils.data import DataLoader
            from src.data.augmentations import get_eval_transforms as gft
            from pathlib import Path as P
            old_transform = gft(256)
            old_ds = GLSDataset(
                P(old_config["paths"]["split_dir"]) / "test.txt",
                P(old_config["paths"]["processed_images_dir"]),
                P(old_config["paths"]["lesion_masks_dir"]),
                256, transform=old_transform, return_id=True,
            )
            old_loader = DataLoader(old_ds, batch_size=8, shuffle=False)
            old_preds, old_gts, old_probs = run_inference_with_probs(old_model, old_loader, device)
            print(f"Old inference done: {len(old_preds)} test samples")

    # ── Generate all slides ────────────────────────────────────────────────────
    print("\nGenerating slide images...")

    if new_preds or (model is not None and slide1_sample_id):
        print("Slide 1: best qualitative result")
        gen_slide1_images(
            new_preds,
            new_probs,
            new_gts,
            config,
            manifest,
            sample_id=slide1_sample_id,
            model=model,
            device=device,
        )

    print("Slide 2: research image")
    gen_slide2_research_image(config, manifest, sample_id=slide2_sample_id)

    print("Slide 3: dataset triplet")
    gen_slide3_dataset_triplet(config, manifest, sample_id=slide3_sample_id)

    print("Slide 4: before/after masking")
    gen_slide4_masking(config, manifest, sample_id=slide4_sample_id)

    print("Slide 5: U-Net diagram")
    gen_slide5_unet_diagram(manifest)

    print("Slide 6: training curves")
    for exp in EXPERIMENTS:
        gen_slide6_training_curves(exp, manifest)

    if new_preds:
        print("Slide 7: baseline qualitative grid")
        gen_slide7_baseline_grid(new_preds, new_probs, new_gts, config, manifest)

    if new_preds or (model is not None and slide8_sample_id):
        print("Slide 8: leaf-masked triple")
        gen_slide8_leafmasked_triple(
            new_preds,
            new_probs,
            new_gts,
            config,
            manifest,
            sample_id=slide8_sample_id,
            model=model,
            device=device,
        )

    if old_preds and new_preds:
        print("Slide 10: qualitative comparison old vs new")
        gen_slide10_qualitative_comparison(old_preds, old_probs, new_preds, new_probs, new_gts, config, manifest, n=n)

    if new_preds or (model is not None and slide11_sample_id):
        print("Slide 11: coverage")
        gen_slide11_coverage(
            new_preds,
            new_probs,
            new_gts,
            config,
            manifest,
            sample_id=slide11_sample_id,
            model=model,
            device=device,
        )

    # ── Save manifest ──────────────────────────────────────────────────────────
    manifest_path = OUT_DIR / "slide_image_manifest.txt"
    manifest_path.write_text("\n".join(manifest))
    print(f"\n{'='*60}")
    print("SLIDE IMAGE MANIFEST")
    print(f"{'='*60}")
    for line in manifest:
        print(line)
    print(f"\nAll images saved to: {OUT_DIR.resolve()}")
    print(f"Manifest saved to:   {manifest_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate every image needed for the supervisor presentation slides.")
    parser.add_argument("--experiment", default="exp01_unet_noaug",
                        help="Primary experiment to use for new-pipeline slides (default: exp01_unet_noaug)")
    parser.add_argument("--old-checkpoints-dir", default=None,
                        help="Directory of OLD (full-image 256x256) .pt files — enables slide 10 comparison")
    parser.add_argument("--new-checkpoints-dir", default=None,
                        help="Directory of NEW (leaf-masked 512x512) .pt files — overrides config path if given")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of samples in comparison grid (default: 4)")
    parser.add_argument("--slide1-sample-id", default=None,
                        help="Specific sample ID to use for Slide 1 (default: best test prediction)")
    parser.add_argument("--slide2-sample-id", default=None,
                        help="Specific sample ID to use for Slide 2 (default: highest coverage test sample)")
    parser.add_argument("--slide3-sample-id", default=None,
                        help="Specific sample ID to use for Slide 3 dataset triplet (default: first test sample >5% cov)")
    parser.add_argument("--slide4-sample-id", default=None,
                        help="Specific sample ID to use for Slide 4 before/after leaf masking (default: first test sample)")
    parser.add_argument("--slide8-sample-id", default=None,
                        help="Specific sample ID to use for Slide 8 leaf-masked triple (default: best test prediction)")
    parser.add_argument("--slide11-sample-id", default=None,
                        help="Specific sample ID to use for Slide 11 coverage visualization (default: high coverage test prediction)")
    args = parser.parse_args()
    main(
        args.experiment,
        args.old_checkpoints_dir,
        args.new_checkpoints_dir,
        args.n,
        slide1_sample_id=args.slide1_sample_id,
        slide2_sample_id=args.slide2_sample_id,
        slide3_sample_id=args.slide3_sample_id,
        slide4_sample_id=args.slide4_sample_id,
        slide8_sample_id=args.slide8_sample_id,
        slide11_sample_id=args.slide11_sample_id,
    )


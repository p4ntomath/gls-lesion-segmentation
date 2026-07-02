
"""One-shot pipeline: run leaf-region extraction over every raw image and cache
the results. Run once, independently of training/evaluation.

Outputs are saved as binary images (0/255) under data/processed/leaf_masks/, and
visual quality assurance overlays are saved under outputs/figures/leaf_mask_qa/.
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
import torch
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download # pyright: ignore[reportMissingImports]
from ultralytics import YOLO # type: ignore
from segment_anything import sam_model_registry, SamPredictor # type: ignore


YOLO_MODEL = None
SAM_PREDICTOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _initialize_models():
    """Initializes and caches the YOLO and SAM models lazily."""
    global YOLO_MODEL, SAM_PREDICTOR

    if YOLO_MODEL is not None and SAM_PREDICTOR is not None:
        return

    yolo_model_path = hf_hub_download(
        repo_id="pedromiguelsanchez/yolo-plant-leaf-detection",
        filename="yolo11x_leaf.pt",
    )
    YOLO_MODEL = YOLO(yolo_model_path)

    sam_dir = Path(os.path.expanduser("~/.cache/segment_anything"))
    sam_dir.mkdir(parents=True, exist_ok=True)
    sam_checkpoint_path = sam_dir / "sam_vit_b_01ec64.pth"

    if not sam_checkpoint_path.exists():
        import urllib.request

        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        urllib.request.urlretrieve(url, sam_checkpoint_path)

    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint_path))
    sam.to(device=DEVICE)
    SAM_PREDICTOR = SamPredictor(sam)


def load_image_rgb(path: Path) -> np.ndarray:
    """Load an image and normalize any EXIF orientation."""
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.array(image, dtype=np.uint8)


def choose_best_yolo_box(boxes_xyxy, image_shape, centre_penalty=0.8):
    """Choose the most likely target leaf bounding box based on size and centrality."""
    height, width = image_shape[:2]
    image_center = np.array([width / 2, height / 2], dtype=np.float32)
    diag = math.sqrt(width**2 + height**2)

    best_score = -1e9
    best_box = None

    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        area_frac = (bw * bh) / (width * height)

        box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
        dist = np.linalg.norm(box_center - image_center) / diag

        score = area_frac - centre_penalty * dist

        if score > best_score:
            best_score = score
            best_box = box

    return best_box


def pad_box(box, image_shape, pad=30):
    """Pads a bounding box while keeping it within image boundaries."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box.astype(int)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def propose_leaf_bbox(image: np.ndarray):
    """Detect the main target leaf using YOLO and return a padded box."""
    _initialize_models()
    height, width = image.shape[:2]

    results = YOLO_MODEL.predict(source=image, conf=0.20, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return np.array([0, 0, width, height], dtype=np.float32)

    boxes = result.boxes.xyxy.cpu().numpy()
    best_box = choose_best_yolo_box(boxes, image.shape)

    if best_box is None:
        return np.array([0, 0, width, height], dtype=np.float32)

    return pad_box(best_box, image.shape, pad=30)


def segment_leaf(image: np.ndarray, bbox):
    """Generate a high-quality binary mask inside the bounding box using SAM."""
    _initialize_models()
    SAM_PREDICTOR.set_image(image)

    masks, scores, _ = SAM_PREDICTOR.predict(box=bbox, multimask_output=True)
    best_idx = int(np.argmax(scores))
    return masks[best_idx]


def postprocess_leaf_mask(mask: np.ndarray) -> np.ndarray:
    """Clean the mask by keeping the largest component and filling holes."""
    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_uint8 = (labels == largest_label).astype(np.uint8)

    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    return closed > 0


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Save a binary mask as a visible 0/255 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_qa_overlay(image_rgb: np.ndarray, box, mask: np.ndarray, save_path: Path) -> None:
    """Save a 4-panel QA figure for inspection."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = box.astype(int)
    boxed_img = image_rgb.copy()
    cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (255, 0, 0), 6)

    mask_rgb = cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)

    overlay = image_rgb.copy()
    red_color = np.array([255, 0, 0], dtype=np.uint8)
    alpha = 0.45
    overlay[mask] = ((1 - alpha) * overlay[mask] + alpha * red_color).astype(np.uint8)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("a) RGB Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(boxed_img)
    axes[0, 1].set_title("b) Bounding Box", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(mask_rgb)
    axes[1, 0].set_title("c) SAM Mask", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("d) Overlay", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main(config_path: str = "configs/base.yaml") -> None:
    """Run leaf mask preprocessing across the dataset."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    raw_images_dir = Path(paths.get("raw_images_dir", "data/raw/images"))
    processed_masks_dir = Path(paths.get("leaf_masks_dir", "data/processed/leaf_masks"))
    qa_dir = Path(paths.get("figures_dir", "outputs/figures")) / "leaf_mask_qa"

    if not raw_images_dir.exists():
        raise FileNotFoundError(f"Required input directory is missing: {raw_images_dir}")

    processed_masks_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(p for p in raw_images_dir.iterdir() if p.suffix.lower() in valid_extensions)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {raw_images_dir}")

    print(f"Found {len(image_paths)} images to process for leaf mask extraction.")

    qa_counter = 0
    max_qa_images = 10

    for idx, img_path in enumerate(image_paths, start=1):
        sample_id = img_path.stem
        output_mask_path = processed_masks_dir / f"{sample_id}.png"

        if output_mask_path.exists():
            print(f"[{idx}/{len(image_paths)}] Skipping existing mask: {sample_id}")
            continue

        print(f"[{idx}/{len(image_paths)}] Processing: {img_path.name}")
        image_rgb = load_image_rgb(img_path)
        bbox = propose_leaf_bbox(image_rgb)
        raw_mask = segment_leaf(image_rgb, bbox)
        cleaned_mask = postprocess_leaf_mask(raw_mask)

        save_mask(cleaned_mask, output_mask_path)

        if qa_counter < max_qa_images:
            qa_output_path = qa_dir / f"{sample_id}_qa.png"
            save_qa_overlay(image_rgb, bbox, cleaned_mask, qa_output_path)
            qa_counter += 1

    print("Leaf mask preprocessing generation complete.")


if __name__ == "__main__":
    main()

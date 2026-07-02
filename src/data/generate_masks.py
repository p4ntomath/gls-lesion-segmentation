"""Download raw images and raw instance bitmaps, then decode them.

Reads:   data/processed/manifest.csv (from parse_dataset.py)
Writes:  data/raw/images/<id>.jpg, data/raw/masks/<id>.png
Writes:  data/processed/images/<id>.jpg, data/processed/lesion_masks/<id>.png
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd
import requests
import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

from src.utils.progress import ProgressTracker, create_progress_bar


def download_file(url: str, dest_path: str) -> None:
    """Download a file if it is not already present."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def load_instance_bitmap(path: str) -> np.ndarray:
    """Load a Segments.ai RGBA instance bitmap as a uint32 id map."""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    # Segments.ai stores the instance id in the RGB channels and fixes alpha at 255.
    # Mask off the alpha byte so the values match the annotation ids in the manifest.
    return arr.view(np.uint32).squeeze(-1) & np.uint32(0x00FFFFFF)


def _parse_annotations(annotations: object) -> list[dict]:
    if isinstance(annotations, list):
        return annotations
    if annotations is None:
        return []
    if isinstance(annotations, float) and np.isnan(annotations):
        return []
    if isinstance(annotations, str) and annotations.strip():
        return json.loads(annotations)
    return []


def instance_ids_to_binary(instance_ids: np.ndarray, annotations: list[dict]) -> np.ndarray:
    """Map instance ids to a binary lesion mask using annotation ids."""
    lesion_ids = {
        int(annotation["id"])
        for annotation in annotations
        if int(annotation.get("category_id", -1)) == 0
    }
    if not lesion_ids:
        return np.zeros_like(instance_ids, dtype=np.uint8)
    return np.isin(instance_ids, list(lesion_ids)).astype(np.uint8)


def compute_lesion_coverage(binary_mask: np.ndarray) -> float:
    """Compute lesion coverage as the percentage of image pixels that are lesion."""
    if binary_mask.size == 0:
        return 0.0
    return float(binary_mask.astype(np.float32).mean() * 100.0)


def _save_binary_mask(binary_mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Store as 0/255 so the mask is still binary but visible in normal viewers.
    Image.fromarray((binary_mask.astype(np.uint8) * 255), mode="L").save(path)


def _save_oriented_image(src_path: Path, dest_path: Path) -> None:
    """Copy an RGB image while applying any EXIF orientation tag first."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src_path)
    image = ImageOps.exif_transpose(image)
    image.save(dest_path)


def process_manifest_rows(
    manifest: pd.DataFrame,
    *,
    row_processor: Callable[[pd.Series], dict[str, Any]] | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Process manifest rows, optionally in parallel.

    The default processor downloads the image and mask, generates the binary
    lesion mask, and returns the lesion coverage for the sample. When workers > 1,
    rows are processed concurrently using threads, which is beneficial for I/O-bound
    download and image handling work.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")

    if row_processor is None:
        row_processor = _process_manifest_row

    if workers == 1:
        tracker = ProgressTracker()
        results = []
        with create_progress_bar(total=len(manifest), desc="Generating masks") as bar:
            for _, row in manifest.iterrows():
                try:
                    result = row_processor(row)
                    results.append(result)
                except Exception:
                    tracker.update(bar, failed=tracker.failed + 1)
                    continue
                tracker.update(bar)
        return results

    rows = [row for _, row in manifest.iterrows()]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(row_processor, rows),
                total=len(rows),
                desc="Generating masks",
                dynamic_ncols=True,
            )
        )
    return results


def _process_manifest_row(row: pd.Series) -> dict[str, Any]:
    sample_id = str(row.sample_id)
    image_raw_path = Path(row.image_raw_path)
    mask_raw_path = Path(row.mask_raw_path)
    image_output_path = Path(row.image_output_path)
    mask_output_path = Path(row.mask_output_path)
    annotations = _parse_annotations(row.annotations)

    download_file(row.image_url, str(image_raw_path))
    download_file(row.mask_url, str(mask_raw_path))

    if not image_output_path.exists():
        _save_oriented_image(image_raw_path, image_output_path)

    instance_ids = load_instance_bitmap(str(mask_raw_path))
    binary_mask = instance_ids_to_binary(instance_ids, annotations)
    _save_binary_mask(binary_mask, mask_output_path)

    return {
        "sample_id": sample_id,
        "lesion_coverage": compute_lesion_coverage(binary_mask),
    }


def main(config_path: str = "configs/base.yaml") -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    manifest_path = Path(config["paths"]["manifest_csv"])
    raw_images_dir = Path(config["paths"]["raw_images_dir"])
    raw_masks_dir = Path(config["paths"]["raw_masks_dir"])
    processed_images_dir = Path(config["paths"]["processed_images_dir"])
    lesion_masks_dir = Path(config["paths"]["lesion_masks_dir"])

    manifest = pd.read_csv(manifest_path)
    if "annotations" not in manifest.columns:
        raise ValueError(
            "manifest.csv is missing the annotations column. Re-run src/data/parse_dataset.py first."
        )

    manifest = manifest.copy()
    manifest["image_raw_path"] = manifest["sample_id"].apply(lambda s: str(raw_images_dir / f"{s}.jpg"))
    manifest["mask_raw_path"] = manifest["sample_id"].apply(lambda s: str(raw_masks_dir / f"{s}.png"))
    manifest["image_output_path"] = manifest["sample_id"].apply(lambda s: str(processed_images_dir / f"{s}.jpg"))
    manifest["mask_output_path"] = manifest["sample_id"].apply(lambda s: str(lesion_masks_dir / f"{s}.png"))

    workers = int(config.get("processing", {}).get("workers", 1))
    results = process_manifest_rows(
        manifest,
        workers=workers,
    )

    manifest["lesion_coverage"] = [item["lesion_coverage"] for item in results]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)


if __name__ == "__main__":
    main()

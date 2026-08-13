"""PyTorch Dataset for (RGB image, binary GLS lesion mask) pairs.

Proposal ref: §4.2.5 (Preprocessing)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


class GLSDataset(torch.utils.data.Dataset):
    """Dataset returning aligned RGB images and binary lesion masks."""

    def __init__(
        self,
        split_txt: str | Path,
        images_dir: str | Path,
        masks_dir: str | Path,
        image_size: int,
        transform=None,
        return_id: bool = False,
        leaf_masks_dir: str | Path | None = None,
        return_leaf: bool = False,
        apply_leaf_masking: bool = False,
    ) -> None:
        self.split_txt = Path(split_txt)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = int(image_size)
        self.transform = transform
        self.return_id = return_id
        self.leaf_masks_dir = Path(leaf_masks_dir) if leaf_masks_dir is not None else None
        self.return_leaf = bool(return_leaf)
        self.apply_leaf_masking = bool(apply_leaf_masking)

        if not self.split_txt.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_txt}")

        self.sample_ids = [line.strip() for line in self.split_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.sample_ids:
            raise ValueError(f"Split file is empty: {self.split_txt}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.array(image, dtype=np.uint8)

    @staticmethod
    def _load_mask(path: Path) -> np.ndarray:
        mask = Image.open(path).convert("L")
        return (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        image_path = self.images_dir / f"{sample_id}.jpg"
        mask_path = self.masks_dir / f"{sample_id}.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for sample {sample_id}: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for sample {sample_id}: {mask_path}")

        image = self._load_rgb(image_path)
        mask = self._load_mask(mask_path)

        leaf_mask = None
        if self.leaf_masks_dir is not None:
            leaf_path = self.leaf_masks_dir / f"{sample_id}.png"
            if leaf_path.exists():
                leaf = Image.open(leaf_path).convert("L")
                leaf_mask = (np.array(leaf, dtype=np.uint8) > 0).astype(np.uint8)
            else:
                leaf_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            if leaf_mask is not None:
                augmented = self.transform(image=image, mask=mask, leaf=leaf_mask)
                image = augmented["image"]
                mask = augmented["mask"]
                leaf_mask = augmented["leaf"]
            else:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0)
            mask = torch.from_numpy(mask[None, ...]).float()
            if leaf_mask is not None:
                leaf_mask = torch.from_numpy(leaf_mask[None, ...]).float()

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.ndim == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        image = image.float()
        if image.max() > 1.0:
            image = image.div(255.0)
        mask = mask.float()
        if mask.max() > 1.0:
            mask = (mask > 0).float()

        if leaf_mask is not None:
            if isinstance(leaf_mask, np.ndarray):
                leaf_mask = torch.from_numpy(leaf_mask)
            if leaf_mask.dim() == 2:
                leaf_mask = leaf_mask.unsqueeze(0)
            leaf_mask = leaf_mask.float()
            if leaf_mask.max() > 1.0:
                leaf_mask = (leaf_mask > 0).float()

            if self.apply_leaf_masking:
                image = image * leaf_mask
                mask = mask * leaf_mask

        if self.return_id and self.return_leaf:
            return sample_id, image, mask, leaf_mask
        if self.return_id:
            return sample_id, image, mask
        if self.return_leaf:
            return image, mask, leaf_mask
        return image, mask
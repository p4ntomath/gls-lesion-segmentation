"""Albumentations pipelines for training-only augmentation.

Proposal ref: §4.2.6 — flips, small rotations, limited scaling, random crops,
moderate brightness/contrast. Spatial transforms applied to image+mask;
intensity transforms to image only. No augmentation for val/test.

Both functions accept an optional with_leaf flag. When True, the returned
Compose declares additional_targets at construction time, which is required
for Albumentations to correctly transform a leaf mask passed as a keyword
argument alongside the image and lesion mask. Passing additional_targets at
call time is silently ignored in recent Albumentations versions.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int, with_leaf: bool = False) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                border_mode=0,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ],
        additional_targets={"leaf": "mask"} if with_leaf else {},
    )


def get_eval_transforms(image_size: int, with_leaf: bool = False) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ],
        additional_targets={"leaf": "mask"} if with_leaf else {},
    )
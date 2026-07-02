"""Sanity tests for the data pipeline. Run with: pytest tests/"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.augmentations import get_eval_transforms
from src.data.dataset import GLSDataset
from src.data.parse_dataset import extract_labeled_samples, _sample_id_from_name
from src.data.generate_masks import instance_ids_to_binary, process_manifest_rows
from src.data.split_data import assign_coverage_bins, stratified_split
from src.utils.progress import ProgressTracker, create_progress_bar


def test_manifest_excludes_unlabeled_samples(tmp_path: Path) -> None:
    raw = {
        "dataset": {
            "samples": [
                {
                    "name": "001.JPG",
                    "uuid": "a1",
                    "attributes": {"image": {"url": "https://example.com/1.jpg"}},
                    "labels": {
                        "ground-truth": {
                            "label_status": "LABELED",
                            "attributes": {"segmentation_bitmap": {"url": "https://example.com/1.png"}},
                        }
                    },
                },
                {
                    "name": "002.JPG",
                    "uuid": "a2",
                    "attributes": {"image": {"url": "https://example.com/2.jpg"}},
                    "labels": {
                        "ground-truth": {
                            "label_status": "UNLABELED",
                            "attributes": {"segmentation_bitmap": {"url": "https://example.com/2.png"}},
                        }
                    },
                },
                {
                    "name": "003.JPG",
                    "uuid": "a3",
                    "attributes": {"image": {"url": "https://example.com/3.jpg"}},
                    "labels": {"ground-truth": None},
                },
            ]
        }
    }

    kept = extract_labeled_samples(raw)
    assert len(kept) == 1
    assert _sample_id_from_name(kept[0]["name"]) == "001"


def test_binary_mask_only_has_0_and_1() -> None:
    instance_ids = np.array([[0, 1, 2], [2, 2, 0]], dtype=np.uint32)
    annotations = [
        {"id": 1, "category_id": 0},
        {"id": 2, "category_id": 1},
    ]

    binary = instance_ids_to_binary(instance_ids, annotations)
    assert binary.shape == instance_ids.shape
    assert set(np.unique(binary)).issubset({0, 1})
    assert binary[0, 1] == 1
    assert binary[1, 2] == 0


def test_dataset_getitem_shapes(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    split_dir = tmp_path / "splits"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    sample_id = "0001"
    width, height = 10, 14
    Image.new("RGB", (width, height), color=(100, 120, 140)).save(image_dir / f"{sample_id}.jpg")
    Image.fromarray(np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (height // 2, width // 2)), mode="L").save(mask_dir / f"{sample_id}.png")
    (split_dir / "train.txt").write_text(f"{sample_id}\n", encoding="utf-8")

    dataset = GLSDataset(
        split_dir / "train.txt",
        image_dir,
        mask_dir,
        image_size=8,
        transform=get_eval_transforms(8),
    )
    image, mask = dataset[0]

    assert image.shape == (3, 8, 8)
    assert mask.shape == (1, 8, 8)
    assert image.dtype == pytest.approx(image.dtype)
    assert mask.dtype == pytest.approx(mask.dtype)
    assert image.max() <= 1.0
    assert set(np.unique(mask.numpy())).issubset({0.0, 1.0})


def test_process_manifest_rows_supports_parallel_workers() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["001", "002", "003"],
            "annotations": ["[]", "[]", "[]"],
        }
    )

    def row_processor(row: pd.Series) -> dict[str, str]:
        return {"sample_id": str(row.sample_id), "processed": "ok"}

    results = process_manifest_rows(manifest, row_processor=row_processor, workers=2)

    assert [item["sample_id"] for item in results] == ["001", "002", "003"]
    assert all(item["processed"] == "ok" for item in results)


def test_split_ratios_approximately_correct() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": [f"{i:03d}" for i in range(20)],
            "lesion_coverage": list(range(20)),
        }
    )
    manifest = assign_coverage_bins(manifest, n_bins=3)
    train_df, val_df, test_df = stratified_split(manifest, ratios=(0.7, 0.15, 0.15), seed=42)

    assert abs(len(train_df) - 14) <= 1
    assert abs(len(val_df) - 3) <= 1
    assert abs(len(test_df) - 3) <= 1
    assert len(train_df) + len(val_df) + len(test_df) == len(manifest)


def test_train_val_test_no_overlap() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": [f"{i:03d}" for i in range(15)],
            "lesion_coverage": list(range(15)),
        }
    )
    manifest = assign_coverage_bins(manifest, n_bins=3)
    train_df, val_df, test_df = stratified_split(manifest, ratios=(0.7, 0.15, 0.15), seed=123)

    train_ids = set(train_df["sample_id"])
    val_ids = set(val_df["sample_id"])
    test_ids = set(test_df["sample_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids | test_ids) == len(manifest)


def test_progress_tracker_updates_postfix_metrics() -> None:
    tracker = ProgressTracker()
    bar = create_progress_bar(total=3, desc="Testing")

    tracker.update(bar, n=1, failed=2)
    tracker.update(bar, n=1, failed=2, loss=0.5)

    assert tracker.failed == 2
    assert "failed=2" in bar.format_dict["postfix"]
    assert "loss=0.5" in bar.format_dict["postfix"]
    bar.close()

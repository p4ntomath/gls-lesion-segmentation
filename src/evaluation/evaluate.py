"""Full evaluation of a trained model on the test set.

Proposal ref: §4.2.10

Responsibilities:
    - Load checkpoint, run inference on test split.
    - Threshold predictions (0.5, or value selected on val set).
    - Compute Dice / IoU / precision / recall (src/training/metrics.py).
    - Compute leaf-area coverage MAE / RMSE / Pearson r (src/evaluation/coverage.py).
    - Save results to experiments/<experiment>/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from src.data.augmentations import get_eval_transforms
from src.data.dataset import GLSDataset
from src.evaluation.coverage import coverage_for_split
from src.models.attention_unet import AttentionUNet
from src.models.unet import UNet
from src.utils.progress import create_progress_bar
from src.training.metrics import (
    binarize,
    confusion_counts,
    dice_coefficient,
    iou_score,
    precision_score,
    recall_score,
    segmentation_metrics,
)


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_experiment_config(experiment: str, config_path: str = "configs/base.yaml") -> dict:
    exp_dir = Path("experiments") / experiment
    exp_config_path = exp_dir / "config.yaml"
    exp_config = _load_yaml(exp_config_path)

    merged = _load_yaml(Path(config_path))
    for extend_path in exp_config.get("extends", []):
        merged = _deep_merge(merged, _load_yaml(Path(extend_path)))

    merged = _deep_merge(merged, exp_config.get("overrides", {}))
    merged["experiment_name"] = experiment
    merged["config_path"] = config_path
    return merged


def build_model(config: dict):
    model_cfg = config["model"]
    name = model_cfg["name"].lower()
    model_kwargs = {
        "in_channels": model_cfg.get("in_channels", 3),
        "out_channels": model_cfg.get("out_channels", 1),
        "base_filters": model_cfg.get("base_filters", 64),
        "depth": model_cfg.get("depth", 4),
    }

    if name == "unet":
        return UNet(**model_kwargs)
    if name == "attention_unet":
        model_kwargs["attention_inter_channels"] = model_cfg.get("attention_inter_channels")
        return AttentionUNet(**model_kwargs)
    raise ValueError(f"Unknown model name: {model_cfg['name']}")


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def build_test_loader(config: dict) -> DataLoader:
    paths = config["paths"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    split_dir = Path(paths["split_dir"])
    processed_images_dir = Path(paths["processed_images_dir"])
    lesion_masks_dir = Path(paths["lesion_masks_dir"])
    image_size = int(data_cfg.get("image_size", 256))
    batch_size = int(training_cfg.get("batch_size", 8))
    num_workers = int(training_cfg.get("num_workers", 0))

    test_transform = get_eval_transforms(image_size)
    dataset = GLSDataset(
        split_dir / "test.txt",
        processed_images_dir,
        lesion_masks_dir,
        image_size,
        transform=test_transform,
        return_id=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model.eval()
    predictions: dict[str, np.ndarray] = {}
    ground_truth: dict[str, np.ndarray] = {}

    with torch.no_grad():
        with create_progress_bar(total=len(loader), desc="Running inference") as bar:
            for batch_idx, batch in enumerate(loader, start=1):
                sample_ids, images, masks = batch
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                probs = torch.sigmoid(logits)
                binary = binarize(probs)

                for idx, sample_id in enumerate(sample_ids):
                    predictions[sample_id] = binary[idx, 0].astype(np.uint8)
                    ground_truth[sample_id] = masks[idx, 0].detach().cpu().numpy().astype(np.uint8)

                bar.set_postfix(batch=batch_idx)
                bar.update(1)

    return predictions, ground_truth


def compute_segmentation_metrics(predictions: dict[str, np.ndarray], ground_truth: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    if set(predictions) != set(ground_truth):
        missing = sorted(set(predictions) ^ set(ground_truth))
        raise ValueError(f"Prediction/ground-truth mismatch for sample IDs: {missing}")

    summary_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_image = []

    for sample_id in sorted(ground_truth):
        pred = predictions[sample_id]
        gt = ground_truth[sample_id]
        metrics = segmentation_metrics(pred, gt)

        summary_counts["tp"] += metrics["tp"]
        summary_counts["fp"] += metrics["fp"]
        summary_counts["fn"] += metrics["fn"]
        summary_counts["tn"] += metrics["tn"]

        per_image.append(
            {
                "sample_id": sample_id,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

    return {
        "dice": dice_coefficient(summary_counts["tp"], summary_counts["fp"], summary_counts["fn"]),
        "iou": iou_score(summary_counts["tp"], summary_counts["fp"], summary_counts["fn"]),
        "precision": precision_score(summary_counts["tp"], summary_counts["fp"]),
        "recall": recall_score(summary_counts["tp"], summary_counts["fn"]),
        "tp": summary_counts["tp"],
        "fp": summary_counts["fp"],
        "fn": summary_counts["fn"],
        "tn": summary_counts["tn"],
    }, per_image


def compute_coverage_metrics(pred_coverage: np.ndarray, ref_coverage: np.ndarray) -> dict:
    pred = np.asarray(pred_coverage, dtype=float)
    ref = np.asarray(ref_coverage, dtype=float)

    mae = float(np.mean(np.abs(pred - ref)))
    rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))

    pearson_r = float("nan")
    if len(pred) > 1 and np.std(pred) > 0 and np.std(ref) > 0:
        pearson_r = float(pearsonr(pred, ref)[0])

    return {
        "coverage_mae": mae,
        "coverage_rmse": rmse,
        "coverage_pearson_r": pearson_r,
    }





def _load_leaf_masks(leaf_masks_dir: Path, sample_ids: list[str], image_size: int) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load whichever leaf masks are actually available on disk.

    Leaf masks come from the separate pipelines/generate_leaf_masks.py step
    (YOLO+SAM, QA-inspected), not from scripts/preprocess.py -- deliberately
    decoupled from training so training never depends on SAM (see README).
    That means it's entirely normal for this to be run before leaf
    extraction finishes, or for a handful of samples to fail QA and never
    get a leaf mask.

    Returns (leaf_masks, missing_ids) instead of raising, so evaluation can
    still report full segmentation metrics (Dice/IoU/precision/recall, which
    don't need leaf masks at all) even when leaf-area coverage isn't ready.
    """
    from PIL import Image

    leaf_masks: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for sample_id in sample_ids:
        leaf_path = leaf_masks_dir / f"{sample_id}.png"
        if not leaf_path.exists():
            missing.append(sample_id)
            continue
        image = Image.open(leaf_path).convert("L")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), resample=Image.NEAREST)
        leaf_masks[sample_id] = (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)
    return leaf_masks, missing


def main(experiment: str, config_path: str = "configs/base.yaml") -> None:
    config = load_experiment_config(experiment, config_path=config_path)
    set_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config)
    _load_checkpoint(model, Path(config["paths"]["checkpoints_dir"]) / f"{experiment}.pt", set_device)
    model.to(set_device)

    test_loader = build_test_loader(config)
    predictions, ground_truth = run_inference(model, test_loader, set_device)

    image_size = int(config.get("data", {}).get("image_size", 256))
    leaf_masks, missing_leaf_ids = _load_leaf_masks(Path(config["paths"]["leaf_masks_dir"]), sorted(ground_truth), image_size)

    segmentation_summary, per_image = compute_segmentation_metrics(predictions, ground_truth)

    coverage_summary: dict = {}
    combined_per_image = [dict(row) for row in per_image]

    if leaf_masks:
        # Only score coverage for samples that actually have a leaf mask --
        # compute_coverage_metrics/coverage_for_split would otherwise raise
        # on the missing ones.
        available_ids = set(leaf_masks)
        pred_subset = {k: v for k, v in predictions.items() if k in available_ids}
        gt_subset = {k: v for k, v in ground_truth.items() if k in available_ids}

        coverage_df = coverage_for_split(pred_subset, gt_subset, leaf_masks)
        coverage_summary = compute_coverage_metrics(
            coverage_df["predicted_coverage"].to_numpy(), coverage_df["reference_coverage"].to_numpy()
        )
        coverage_index = {row["sample_id"]: row for row in coverage_df.to_dict(orient="records")}
        for row in combined_per_image:
            sample_id = row["sample_id"]
            cov = coverage_index.get(sample_id)
            if cov is not None:
                row["predicted_coverage"] = cov["predicted_coverage"]
                row["reference_coverage"] = cov["reference_coverage"]

    if missing_leaf_ids:
        if leaf_masks:
            print(
                f"WARNING: {len(missing_leaf_ids)}/{len(ground_truth)} test samples have no leaf mask yet. "
                f"Segmentation metrics (Dice/IoU/precision/recall) below cover all {len(ground_truth)} "
                f"test samples. Leaf-area coverage metrics only cover the {len(leaf_masks)} samples that "
                f"have a leaf mask. Run pipelines/generate_leaf_masks.py to fill in the rest, then "
                f"re-run this script to get coverage metrics over the full test set."
            )
        else:
            print(
                f"WARNING: no leaf masks found for any of the {len(ground_truth)} test samples. "
                f"Segmentation metrics (Dice/IoU/precision/recall) below are still complete and valid. "
                f"Leaf-area coverage metrics were skipped entirely -- run "
                f"pipelines/generate_leaf_masks.py first, then re-run this script to get them."
            )

    results = {
        "experiment": experiment,
        "summary": {
            **segmentation_summary,
            "coverage": coverage_summary if coverage_summary else "skipped -- no leaf masks available yet",
            "num_samples": len(ground_truth),
            "num_samples_with_leaf_mask": len(leaf_masks),
            "num_samples_missing_leaf_mask": len(missing_leaf_ids),
        },
        "per_image": combined_per_image,
    }

    exp_dir = Path("experiments") / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_path = exp_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results to {results_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained GLS segmentation experiment")
    parser.add_argument("--experiment", required=True, help="Experiment folder name under experiments/")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config to resolve experiment settings")
    args = parser.parse_args()
    main(args.experiment, config_path=args.config)

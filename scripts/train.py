"""CLI entry point for training a single experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import yaml
from torch.utils.data import DataLoader

from src.data.augmentations import get_eval_transforms, get_train_transforms
from src.data.dataset import GLSDataset
from src.models.attention_unet import AttentionUNet
from src.models.unet import UNet
from src.training.trainer import Trainer
from src.utils.seed import set_seed


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


def load_experiment_config(experiment: str) -> dict:
    exp_dir = Path("experiments") / experiment
    exp_config_path = exp_dir / "config.yaml"
    exp_config = _load_yaml(exp_config_path)

    merged = {}
    for extend_path in exp_config.get("extends", []):
        merged = _deep_merge(merged, _load_yaml(Path(extend_path)))

    merged = _deep_merge(merged, exp_config.get("overrides", {}))
    merged["experiment_name"] = experiment
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


def build_loaders(config: dict):
    paths = config["paths"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    split_dir = Path(paths["split_dir"])
    processed_images_dir = Path(paths["processed_images_dir"])
    lesion_masks_dir = Path(paths["lesion_masks_dir"])
    image_size = int(data_cfg.get("image_size", 256))
    batch_size = int(training_cfg.get("batch_size", 8))
    num_workers = int(training_cfg.get("num_workers", 0))

    train_transform = get_train_transforms(image_size) if training_cfg.get("augmentation", False) else get_eval_transforms(image_size)
    eval_transform = get_eval_transforms(image_size)

    train_dataset = GLSDataset(split_dir / "train.txt", processed_images_dir, lesion_masks_dir, image_size, transform=train_transform)
    val_dataset = GLSDataset(split_dir / "val.txt", processed_images_dir, lesion_masks_dir, image_size, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def main(experiment: str, max_epochs: int | None = None, patience: int | None = None) -> None:
    config = load_experiment_config(experiment)
    set_seed(int(config["data"].get("seed", 42)))

    model = build_model(config)
    train_loader, val_loader = build_loaders(config)

    print(f"Starting training for experiment: {experiment}", flush=True)
    print(f"  device: {Trainer(model, train_loader, val_loader, config).device}", flush=True)
    print(f"  train samples: {len(train_loader.dataset)}", flush=True)
    print(f"  val samples: {len(val_loader.dataset)}", flush=True)
    print(f"  batch size: {train_loader.batch_size}", flush=True)
    print(f"  max_epochs: {max_epochs if max_epochs is not None else config['training'].get('max_epochs', 100)}", flush=True)
    print(f"  patience: {patience if patience is not None else config['training'].get('early_stopping_patience', 15)}", flush=True)

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit(
        max_epochs=max_epochs if max_epochs is not None else config["training"].get("max_epochs", 100),
        patience=patience if patience is not None else config["training"].get("early_stopping_patience", 15),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GLS segmentation experiment")
    parser.add_argument("--experiment", required=True, help="Experiment folder name under experiments/")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs for this training run")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience for this training run")
    args = parser.parse_args()
    main(args.experiment, max_epochs=args.max_epochs, patience=args.patience)

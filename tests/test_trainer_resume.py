from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.training.trainer import Trainer


class TinySegDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        image = torch.rand(3, 16, 16)
        mask = (torch.rand(1, 16, 16) > 0.7).float()
        return image, mask


def _build_config(tmp_path: Path, *, lr: float = 1e-3) -> dict:
    return {
        "experiment_name": "resume_test",
        "paths": {
            "checkpoints_dir": str(tmp_path / "checkpoints"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "training": {
            "learning_rate": lr,
            "threshold": 0.5,
            "bce_weight": 0.5,
        },
    }


def _build_loaders() -> tuple[DataLoader, DataLoader]:
    dataset = TinySegDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    return loader, loader


def _build_model() -> torch.nn.Module:
    return torch.nn.Conv2d(3, 1, kernel_size=1)


def test_resume_restores_state_and_continues_epochs(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    train_loader, val_loader = _build_loaders()

    trainer = Trainer(_build_model(), train_loader, val_loader, config)
    trainer.fit(max_epochs=1, patience=10)

    assert trainer.best_checkpoint_path.exists()
    assert trainer.latest_checkpoint_path.exists()

    latest_checkpoint = torch.load(trainer.latest_checkpoint_path, map_location="cpu", weights_only=False)
    assert int(latest_checkpoint["epoch"]) == 1
    assert "optimizer_state_dict" in latest_checkpoint

    resumed = Trainer(_build_model(), train_loader, val_loader, config)
    resume_state = resumed.resume_from_checkpoint()

    assert int(resume_state["start_epoch"]) == 2
    assert len(resumed.history) == 1

    resumed.fit(
        max_epochs=2,
        patience=10,
        start_epoch=resume_state["start_epoch"],
        epochs_without_improvement=resume_state["epochs_without_improvement"],
    )

    latest_after_resume = torch.load(resumed.latest_checkpoint_path, map_location="cpu", weights_only=False)
    assert int(latest_after_resume["epoch"]) == 2


def test_resume_rejects_mismatched_config_by_default(tmp_path: Path) -> None:
    base_config = _build_config(tmp_path, lr=1e-3)
    train_loader, val_loader = _build_loaders()

    trainer = Trainer(_build_model(), train_loader, val_loader, base_config)
    trainer.fit(max_epochs=1, patience=10)

    changed_config = _build_config(tmp_path, lr=2e-3)
    resumed = Trainer(_build_model(), train_loader, val_loader, changed_config)

    with pytest.raises(ValueError, match="config fingerprint"):
        resumed.resume_from_checkpoint()

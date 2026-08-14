"""Trainer class shared by all experiments.

Proposal ref: §4.2.8, §4.2.11
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.training.losses import BCEDiceLoss
from src.training.metrics import confusion_counts, dice_coefficient, iou_score, precision_score, recall_score
from src.utils.progress import create_progress_bar


class Trainer:
    """Minimal training loop with validation, checkpointing, and CSV logging."""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        training_cfg = config["training"]
        paths = config["paths"]

        self.lr = float(training_cfg.get("learning_rate", 1e-4))
        self.threshold = float(training_cfg.get("threshold", 0.5))
        self.bce_weight = float(training_cfg.get("bce_weight", 0.5))
        self.experiment_name = config.get("experiment_name", "run")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = BCEDiceLoss(bce_weight=self.bce_weight)

        self.checkpoint_dir = Path(paths["checkpoints_dir"])
        self.log_dir = Path(paths["logs_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.best_checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}.pt"
        self.latest_checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pt"
        self.checkpoint_path = self.best_checkpoint_path
        self.log_path = self.log_dir / f"{self.experiment_name}.csv"

        self.best_val_dice = float("-inf")
        self.best_epoch = 0
        self.history: list[dict] = []
        self.config_fingerprint = self._config_fingerprint(self.config)

    @staticmethod
    def _config_fingerprint(config: dict) -> str:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _get_rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _set_rng_state(state: dict[str, Any] | None) -> None:
        if not state:
            return
        python_state = state.get("python")
        if python_state is not None:
            random.setstate(python_state)

        numpy_state = state.get("numpy")
        if numpy_state is not None:
            np.random.set_state(numpy_state)

        torch_state = state.get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch_state)

        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        with create_progress_bar(total=len(self.train_loader), desc=f"Epoch {self.current_epoch}/{self.max_epochs} training", leave=False) as train_bar:
            for batch_idx, batch in enumerate(self.train_loader, start=1):
                if len(batch) == 2:
                    images, masks = batch
                    leaf = None
                elif len(batch) == 3:
                    images, masks, leaf = batch
                else:
                    raise ValueError("Unexpected batch structure from train_loader")

                images = images.to(self.device)
                masks = masks.to(self.device)
                if leaf is not None:
                    leaf = leaf.to(self.device)
                    if leaf.dim() == 3:
                        leaf = leaf.unsqueeze(1)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                loss.backward()
                self.optimizer.step()

                batch_size = images.size(0)
                running_loss += float(loss.item()) * batch_size
                total_samples += batch_size

                probs = torch.sigmoid(logits)
                pred_binary = probs >= self.threshold
                if leaf is not None:
                    pred_binary = pred_binary & (leaf >= 0.5)
                    masks_for_metrics = (masks >= 0.5) & (leaf >= 0.5)
                else:
                    masks_for_metrics = masks >= 0.5
                batch_tp, batch_fp, batch_fn, _ = confusion_counts(pred_binary, masks_for_metrics)
                batch_dice = dice_coefficient(batch_tp, batch_fp, batch_fn)
                batch_iou = iou_score(batch_tp, batch_fp, batch_fn)

                lr = self.optimizer.param_groups[0].get("lr")
                train_bar.set_postfix(
                    epoch=self.current_epoch,
                    batch=batch_idx,
                    loss=f"{loss.item():.4f}",
                    dice=f"{batch_dice:.4f}",
                    iou=f"{batch_iou:.4f}",
                    lr=f"{lr:.2e}",
                )
                train_bar.update(1)

        return running_loss / max(total_samples, 1)

    def validate(self) -> dict:
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        tp = fp = fn = tn = 0

        with torch.no_grad():
            with create_progress_bar(total=len(self.val_loader), desc="Validating", leave=False) as val_bar:
                for batch_idx, batch in enumerate(self.val_loader, start=1):
                    if len(batch) == 2:
                        images, masks = batch
                        leaf = None
                    elif len(batch) == 3:
                        images, masks, leaf = batch
                    else:
                        raise ValueError("Unexpected batch structure from val_loader")

                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    if leaf is not None:
                        leaf = leaf.to(self.device)
                        if leaf.dim() == 3:
                            leaf = leaf.unsqueeze(1)

                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                    probs = torch.sigmoid(logits)
                    pred_binary = probs >= self.threshold
                    if leaf is not None:
                        pred_binary = pred_binary & (leaf >= 0.5)
                        masks_for_metrics = (masks >= 0.5) & (leaf >= 0.5)
                    else:
                        masks_for_metrics = masks >= 0.5
                    batch_tp, batch_fp, batch_fn, batch_tn = confusion_counts(pred_binary, masks_for_metrics)

                    tp += batch_tp
                    fp += batch_fp
                    fn += batch_fn
                    tn += batch_tn

                    batch_size = images.size(0)
                    running_loss += float(loss.item()) * batch_size
                    total_samples += batch_size

                    batch_tp_display, batch_fp_display, batch_fn_display, _ = confusion_counts(pred_binary, masks_for_metrics)
                    batch_dice = dice_coefficient(batch_tp_display, batch_fp_display, batch_fn_display)
                    batch_iou = iou_score(batch_tp_display, batch_fp_display, batch_fn_display)

                    val_bar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        dice=f"{batch_dice:.4f}",
                        iou=f"{batch_iou:.4f}",
                    )
                    val_bar.update(1)

        val_loss = running_loss / max(total_samples, 1)
        metrics = {
            "val_loss": val_loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice": dice_coefficient(tp, fp, fn),
            "iou": iou_score(tp, fp, fn),
            "precision": precision_score(tp, fp),
            "recall": recall_score(tp, fn),
        }
        return metrics

    def _save_checkpoint(self, checkpoint_path: Path, epoch: int, metrics: dict, epochs_without_improvement: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
                "config_fingerprint": self.config_fingerprint,
                "best_val_dice": self.best_val_dice,
                "best_epoch": self.best_epoch,
                "epochs_without_improvement": int(epochs_without_improvement),
                "history": self.history,
                "rng_state": self._get_rng_state(),
            },
            checkpoint_path,
        )

    def resume_from_checkpoint(self, checkpoint_path: str | Path | None = None, *, strict_config: bool = True) -> dict[str, int]:
        if checkpoint_path is None:
            if self.latest_checkpoint_path.exists():
                ckpt_path = self.latest_checkpoint_path
            elif self.best_checkpoint_path.exists():
                ckpt_path = self.best_checkpoint_path
            else:
                raise FileNotFoundError(
                    "No checkpoint found to resume from. "
                    f"Checked {self.latest_checkpoint_path} and {self.best_checkpoint_path}."
                )
        else:
            ckpt_path = Path(checkpoint_path)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        checkpoint_fp = checkpoint.get("config_fingerprint")
        if strict_config and checkpoint_fp is not None and checkpoint_fp != self.config_fingerprint:
            raise ValueError(
                "Refusing to resume because checkpoint config fingerprint differs from current config. "
                "Use --allow-resume-config-mismatch only if you intentionally changed the config."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_val_dice = float(checkpoint.get("best_val_dice", checkpoint.get("metrics", {}).get("dice", float("-inf"))))
        self.best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))

        loaded_history = checkpoint.get("history")
        if isinstance(loaded_history, list):
            self.history = loaded_history

        self._set_rng_state(checkpoint.get("rng_state"))

        last_epoch = int(checkpoint.get("epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        return {
            "start_epoch": last_epoch + 1,
            "epochs_without_improvement": epochs_without_improvement,
        }

    def _write_log(self) -> None:
        if not self.history:
            return

        fieldnames = list({key for row in self.history for key in row.keys()})
        ordered_keys = [
            "epoch",
            "train_loss",
            "val_loss",
            "tp",
            "fp",
            "fn",
            "tn",
            "dice",
            "iou",
            "precision",
            "recall",
        ]
        fieldnames = [key for key in ordered_keys if key in fieldnames] + [key for key in fieldnames if key not in ordered_keys]

        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

    def fit(self, max_epochs, patience, *, start_epoch: int = 1, epochs_without_improvement: int = 0) -> None:
        patience = int(patience)

        self.max_epochs = int(max_epochs)
        start_epoch = int(start_epoch)

        if start_epoch < 1:
            raise ValueError(f"start_epoch must be >= 1, got {start_epoch}")
        if start_epoch > self.max_epochs:
            print(
                f"Checkpoint already at epoch {start_epoch - 1}, which is >= max_epochs ({self.max_epochs}). "
                "Nothing to train.",
                flush=True,
            )
            return

        for epoch in range(start_epoch, self.max_epochs + 1):
            self.current_epoch = epoch
            print(f"Starting epoch {epoch}/{self.max_epochs}", flush=True)
            train_loss = self.train_epoch()
            val_metrics = self.validate()

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics,
            }
            self.history.append(row)
            self._write_log()

            self._save_checkpoint(self.latest_checkpoint_path, epoch, val_metrics, epochs_without_improvement)

            if val_metrics["dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["dice"]
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(self.best_checkpoint_path, epoch, val_metrics, epochs_without_improvement)
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | dice={val_metrics['dice']:.4f} | "
                f"iou={val_metrics['iou']:.4f}",
                flush=True,
            )

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch: {self.best_epoch}).", flush=True)
                break

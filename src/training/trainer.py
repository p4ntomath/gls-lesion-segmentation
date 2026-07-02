"""Trainer class shared by all experiments.

Proposal ref: §4.2.8, §4.2.11
"""

from __future__ import annotations

import csv
from pathlib import Path

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

        self.checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}.pt"
        self.log_path = self.log_dir / f"{self.experiment_name}.csv"

        self.best_val_dice = float("-inf")
        self.best_epoch = 0
        self.history: list[dict] = []

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        with create_progress_bar(total=len(self.train_loader), desc=f"Epoch {self.current_epoch}/{self.max_epochs} training", leave=False) as train_bar:
            for batch_idx, (images, masks) in enumerate(self.train_loader, start=1):
                images = images.to(self.device)
                masks = masks.to(self.device)

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
                batch_tp, batch_fp, batch_fn, _ = confusion_counts(pred_binary, masks >= 0.5)
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
                for batch_idx, (images, masks) in enumerate(self.val_loader, start=1):
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                    probs = torch.sigmoid(logits)
                    pred_binary = probs >= self.threshold
                    batch_tp, batch_fp, batch_fn, batch_tn = confusion_counts(pred_binary, masks >= 0.5)

                    tp += batch_tp
                    fp += batch_fp
                    fn += batch_fn
                    tn += batch_tn

                    batch_size = images.size(0)
                    running_loss += float(loss.item()) * batch_size
                    total_samples += batch_size

                    batch_tp, batch_fp, batch_fn, _ = confusion_counts(pred_binary, masks >= 0.5)
                    batch_dice = dice_coefficient(batch_tp, batch_fp, batch_fn)
                    batch_iou = iou_score(batch_tp, batch_fp, batch_fn)

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

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            self.checkpoint_path,
        )

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

    def fit(self, max_epochs, patience) -> None:
        patience = int(patience)
        epochs_without_improvement = 0

        self.max_epochs = int(max_epochs)
        for epoch in range(1, self.max_epochs + 1):
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

            if val_metrics["dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["dice"]
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics)
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

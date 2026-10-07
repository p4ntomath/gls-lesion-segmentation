"""Run every GLS experiment across multiple random seeds.

Runs experiments sequentially so one TPU VM can safely use all XLA devices.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = [
    "exp01_unet_noaug",
    "exp02_unet_aug",
    "exp03_attnunet_noaug",
    "exp04_attnunet_aug",
]


def run_one(experiment: str, seed: int, *, device: str,
            max_epochs: int | None, patience: int | None) -> None:
    env = os.environ.copy()
    if device != "auto":
        env["GLS_DEVICE"] = device

    cmd = [
        sys.executable, "scripts/train.py",
        "--experiment", experiment,
        "--seed", str(seed),
        "--output-tag", f"seed{seed}",
    ]
    if max_epochs is not None:
        cmd += ["--max-epochs", str(max_epochs)]
    if patience is not None:
        cmd += ["--patience", str(patience)]

    print("\n" + "=" * 80, flush=True)
    print(
        f"RUN: {experiment} | seed={seed} | "
        f"device={env.get('GLS_DEVICE', 'auto')}",
        flush=True,
    )
    print("=" * 80, flush=True)
    subprocess.run(cmd, env=env, check=True)


def read_best_dice(log_path: Path) -> tuple[float | None, int | None]:
    if not log_path.exists():
        return None, None

    with log_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None

    dice_key = next(
        (k for k in ("val_dice", "validation_dice", "best_val_dice")
         if k in rows[0]),
        None,
    )
    epoch_key = next(
        (k for k in ("epoch", "current_epoch") if k in rows[0]),
        None,
    )
    if dice_key is None:
        return None, None

    values = []
    for row in rows:
        try:
            values.append((
                float(row[dice_key]),
                int(row[epoch_key]) if epoch_key else None,
            ))
        except (TypeError, ValueError):
            continue

    return max(values, key=lambda item: item[0]) if values else (None, None)


def write_summary() -> None:
    results_dir = Path("outputs/results")
    logs_dir = Path("outputs/logs")
    checkpoints_dir = Path("outputs/checkpoints")
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment in EXPERIMENTS:
        for checkpoint in sorted(
            checkpoints_dir.glob(f"{experiment}_seed*.pt")
        ):
            stem = checkpoint.stem
            prefix = f"{experiment}_"
            tag = stem[len(prefix):] if stem.startswith(prefix) else stem
            try:
                seed = int(tag.removeprefix("seed"))
            except ValueError:
                continue

            dice, best_epoch = read_best_dice(
                logs_dir / f"{stem}.csv"
            )
            rows.append({
                "experiment": experiment,
                "seed": seed,
                "best_val_dice": dice,
                "best_epoch": best_epoch,
                "checkpoint": str(checkpoint),
            })

    summary_path = results_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment", "seed", "best_val_dice",
                "best_epoch", "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda r: (r["experiment"], r["seed"]))
        )

    aggregate_rows = []
    for experiment in EXPERIMENTS:
        values = [
            float(r["best_val_dice"])
            for r in rows
            if r["experiment"] == experiment
            and r["best_val_dice"] is not None
        ]
        if not values:
            continue

        mean = sum(values) / len(values)
        variance = (
            sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            if len(values) > 1 else 0.0
        )
        aggregate_rows.append({
            "experiment": experiment,
            "n_runs": len(values),
            "mean_best_val_dice": mean,
            "std_best_val_dice": variance ** 0.5,
            "min_best_val_dice": min(values),
            "max_best_val_dice": max(values),
        })

    aggregate_path = results_dir / "aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment", "n_runs", "mean_best_val_dice",
                "std_best_val_dice", "min_best_val_dice",
                "max_best_val_dice",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"\nWrote {summary_path}", flush=True)
    print(f"Wrote {aggregate_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all GLS experiments across multiple seeds."
    )
    parser.add_argument(
        "--n-runs", type=int, default=10,
        help="Number of seeds per experiment.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "xla"], default="auto"
    )
    parser.add_argument(
        "--start-seed", type=int, default=42,
        help="First seed; subsequent seeds increment by 1.",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    if args.n_runs < 1:
        parser.error("--n-runs must be at least 1")

    print(f"Experiments: {len(EXPERIMENTS)}", flush=True)
    print(f"Runs per experiment: {args.n_runs}", flush=True)
    print(
        f"Total training runs: {len(EXPERIMENTS) * args.n_runs}",
        flush=True,
    )
    print(
        f"Seeds: {args.start_seed}.."
        f"{args.start_seed + args.n_runs - 1}",
        flush=True,
    )
    print(f"Device: {args.device}", flush=True)

    for experiment in EXPERIMENTS:
        for offset in range(args.n_runs):
            run_one(
                experiment,
                args.start_seed + offset,
                device=args.device,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )

    write_summary()


if __name__ == "__main__":
    main()

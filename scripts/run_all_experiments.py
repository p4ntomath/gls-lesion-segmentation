"""PJRT 4-Core Multi-Run Experiment Runner for TPU and GPU/CPU.

Runs experiments in parallel across all 4 TPU cores using:
    torch_xla.distributed.xla_multiprocessing.spawn(worker, nprocs=4)

Pipeline per run:
    1. TRAIN (with unique seed and output tag e.g. seed42)
    2. CLEANUP (_latest.pt checkpoint is deleted to save space)
    3. TEST EVALUATION (loads best checkpoint, computes segmentation + coverage metrics)
    4. RECORD RESULTS (outputs/results/<experiment>_seed<N>/results.json)
    5. COMPILE MASTER CSV (outputs/results/all_runs.csv)

Outputs
-------
outputs/
  checkpoints/
    <experiment>_seed<N>.pt          # best checkpoint per run
  logs/
    <experiment>_seed<N>.csv         # per-epoch training/val curves
  results/
    <experiment>_seed<N>/
      results.json                   # full test-set evaluation results
    all_runs.csv                     # master 12-column benchmark dataset across all runs
    aggregate.csv                    # mean/std/min/max per experiment condition
    summary.csv                      # compact summary of all runs

Usage
-----
# Smoke test (4 workers × 1 epoch, seeds 100-103 across the 4 conditions):
python scripts/run_all_experiments.py --smoke-test --device xla

# Full 40-run benchmark (seeds 42..51, skipping already-trained seed 42):
python scripts/run_all_experiments.py --device xla --start-seed 42 --n-runs 10

# Local test on CPU/GPU:
python scripts/run_all_experiments.py --device auto --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

EXPERIMENTS = [
    "exp01_unet_noaug",
    "exp02_unet_aug",
    "exp03_attnunet_noaug",
    "exp04_attnunet_aug",
]

SMOKE_TEST_JOBS = [
    {"experiment": "exp01_unet_noaug", "seed": 100, "tag": "seed100"},
    {"experiment": "exp02_unet_aug", "seed": 101, "tag": "seed101"},
    {"experiment": "exp03_attnunet_noaug", "seed": 102, "tag": "seed102"},
    {"experiment": "exp04_attnunet_aug", "seed": 103, "tag": "seed103"},
]

ALL_RUNS_COLUMNS = [
    "experiment",
    "seed",
    "best_val_dice",
    "best_epoch",
    "test_dice",
    "test_iou",
    "test_precision",
    "test_recall",
    "leaf_dice",
    "coverage_mae",
    "coverage_rmse",
    "coverage_pearson_r",
]


def is_tpu_available() -> bool:
    """Check if we are in a TPU environment (PJRT or Colab TPU)."""
    if os.environ.get("PJRT_DEVICE") == "TPU" or os.environ.get("COLAB_TPU_ADDR"):
        return True
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device_hw(xm.xla_device()) == "TPU"
    except Exception:
        return False


def read_best_dice_from_log(log_path: Path) -> tuple[float | None, int | None]:
    """Parse training log CSV to extract best validation Dice and its epoch."""
    if not log_path.exists():
        return None, None
    try:
        with log_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None, None

        dice_key = next((k for k in ("dice", "val_dice", "validation_dice", "best_val_dice") if k in rows[0]), None)
        epoch_key = next((k for k in ("epoch", "current_epoch") if k in rows[0]), None)
        if dice_key is None:
            return None, None

        parsed = []
        for row in rows:
            try:
                d_val = float(row[dice_key])
                e_val = int(row[epoch_key]) if epoch_key and row.get(epoch_key) else None
                parsed.append((d_val, e_val))
            except (TypeError, ValueError):
                continue
        return max(parsed, key=lambda x: x[0]) if parsed else (None, None)
    except Exception:
        return None, None


def execute_single_job(
    job: dict[str, Any],
    *,
    worker_id: int,
    max_epochs: int | None,
    patience: int | None,
    skip_existing: bool,
    config_path: str = "configs/base.yaml",
) -> None:
    """Run Training and Test Evaluation for a single job."""
    from scripts.train import main as train_main
    from scripts.evaluate import main as eval_main

    experiment = job["experiment"]
    seed = int(job["seed"])
    tag = job["tag"]

    checkpoints_dir = ROOT_DIR / "outputs" / "checkpoints"
    results_dir = ROOT_DIR / "outputs" / "results" / f"{experiment}_{tag}"
    checkpoint_path = checkpoints_dir / f"{experiment}_{tag}.pt"
    result_json_path = results_dir / "results.json"

    # ── Step 1: Training ──────────────────────────────────────────────────────
    if skip_existing and checkpoint_path.exists():
        print(
            f"[Worker {worker_id}] Checkpoint already exists for {experiment}/{tag}. "
            f"Skipping training.",
            flush=True,
        )
    else:
        print(f"\n[Worker {worker_id}] >>> START TRAINING: {experiment} | {tag} (seed={seed})", flush=True)
        train_main(
            experiment=experiment,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            output_tag=tag,
        )
        print(f"[Worker {worker_id}] <<< FINISHED TRAINING: {experiment} | {tag}", flush=True)

    # ── Step 2: Test Evaluation ───────────────────────────────────────────────
    if skip_existing and result_json_path.exists():
        print(
            f"[Worker {worker_id}] Evaluation already exists for {experiment}/{tag}. "
            f"Skipping evaluation.",
            flush=True,
        )
    else:
        print(f"[Worker {worker_id}] >>> EVALUATING TEST SET: {experiment} | {tag}...", flush=True)
        eval_main(
            experiment=experiment,
            config_path=config_path,
            output_tag=tag,
            device="cpu",  # evaluation on CPU is fast (~5s) and avoids device contention
        )
        print(f"[Worker {worker_id}] <<< EVALUATION COMPLETED: {experiment} | {tag}", flush=True)

    # Free memory
    gc.collect()


def _tpu_worker_entry(
    index: int,
    jobs: list[dict[str, Any]],
    max_epochs: int | None,
    patience: int | None,
    skip_existing: bool,
    config_path: str,
) -> None:
    """Worker entry point spawned by xmp.spawn().

    index: Local rank in {0, 1, 2, 3}.
    xm.xla_device() inside this process automatically selects xla:index.
    """
    os.environ["GLS_DEVICE"] = "xla"

    # Distribute jobs round-robin: worker i gets jobs i, i+4, i+8, ...
    worker_jobs = [job for i, job in enumerate(jobs) if i % 4 == index]
    print(
        f"[Worker {index}] Initialized on TPU core {index}. "
        f"Assigned {len(worker_jobs)} of {len(jobs)} total jobs.",
        flush=True,
    )

    for step, job in enumerate(worker_jobs, start=1):
        print(
            f"\n[Worker {index}] Progress: job {step}/{len(worker_jobs)} "
            f"({job['experiment']} | {job['tag']})",
            flush=True,
        )
        try:
            execute_single_job(
                job,
                worker_id=index,
                max_epochs=max_epochs,
                patience=patience,
                skip_existing=skip_existing,
                config_path=config_path,
            )
        except Exception as exc:
            print(f"[Worker {index}] ERROR on {job['experiment']}/{job['tag']}: {exc}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"\n[Worker {index}] ALL ASSIGNED JOBS COMPLETED!", flush=True)


def compile_master_results() -> list[dict[str, Any]]:
    """Scan outputs/results/*/results.json and logs to compile all_runs.csv."""
    results_base = ROOT_DIR / "outputs" / "results"
    logs_dir = ROOT_DIR / "outputs" / "logs"
    checkpoints_dir = ROOT_DIR / "outputs" / "checkpoints"
    results_base.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    # Find all results.json folders matching <experiment>_seed<N>
    for res_dir in sorted(results_base.glob("*_seed*")):
        if not res_dir.is_dir():
            continue
        res_file = res_dir / "results.json"
        if not res_file.exists():
            continue

        try:
            with res_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        experiment = data.get("experiment", "")
        output_tag = data.get("output_tag", "")
        summary = data.get("summary", {})

        try:
            seed = int(output_tag.removeprefix("seed"))
        except ValueError:
            continue

        # Extract best val dice and epoch from training log
        log_path = logs_dir / f"{experiment}_{output_tag}.csv"
        best_val_dice, best_epoch = read_best_dice_from_log(log_path)

        # Fallback to checkpoint if log was missing
        if best_val_dice is None:
            ckpt_path = checkpoints_dir / f"{experiment}_{output_tag}.pt"
            if ckpt_path.exists():
                try:
                    import torch
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    best_val_dice = float(ckpt.get("best_val_dice", "nan"))
                    best_epoch = int(ckpt.get("best_epoch", 0))
                except Exception:
                    pass

        # Extract test evaluation metrics
        test_dice = summary.get("dice")
        test_iou = summary.get("iou")
        test_prec = summary.get("precision")
        test_rec = summary.get("recall")

        # Masked metrics
        masked_sec = summary.get("segmentation_masked_by_leaf", {})
        leaf_dice = masked_sec.get("dice") if isinstance(masked_sec, dict) else None

        # Coverage metrics
        cov_sec = summary.get("coverage", {})
        cov_mae = cov_sec.get("coverage_mae") if isinstance(cov_sec, dict) else None
        cov_rmse = cov_sec.get("coverage_rmse") if isinstance(cov_sec, dict) else None
        cov_r = cov_sec.get("coverage_pearson_r") if isinstance(cov_sec, dict) else None

        record = {
            "experiment": experiment,
            "seed": seed,
            "best_val_dice": round(best_val_dice, 6) if isinstance(best_val_dice, (int, float)) else "N/A",
            "best_epoch": best_epoch if best_epoch is not None else "N/A",
            "test_dice": round(test_dice, 6) if isinstance(test_dice, (int, float)) else "N/A",
            "test_iou": round(test_iou, 6) if isinstance(test_iou, (int, float)) else "N/A",
            "test_precision": round(test_prec, 6) if isinstance(test_prec, (int, float)) else "N/A",
            "test_recall": round(test_rec, 6) if isinstance(test_rec, (int, float)) else "N/A",
            "leaf_dice": round(leaf_dice, 6) if isinstance(leaf_dice, (int, float)) else "N/A",
            "coverage_mae": round(cov_mae, 6) if isinstance(cov_mae, (int, float)) else "N/A",
            "coverage_rmse": round(cov_rmse, 6) if isinstance(cov_rmse, (int, float)) else "N/A",
            "coverage_pearson_r": round(cov_r, 6) if isinstance(cov_r, (int, float)) else "N/A",
        }
        records.append(record)

    # Sort deterministically
    records.sort(key=lambda r: (r["experiment"], int(r["seed"])))

    # Write all_runs.csv
    all_runs_path = results_base / "all_runs.csv"
    with all_runs_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_RUNS_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    # Write summary.csv (compatible with older scripts)
    summary_path = results_base / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "seed", "best_val_dice", "test_dice", "leaf_dice"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "experiment": r["experiment"],
                "seed": r["seed"],
                "best_val_dice": r["best_val_dice"],
                "test_dice": r["test_dice"],
                "leaf_dice": r["leaf_dice"],
            })

    # Write aggregate.csv with statistical metrics
    _write_aggregate_csv(records, results_base / "aggregate.csv")

    print(f"\n[Master CSV] Compiled {len(records)} runs -> {all_runs_path}", flush=True)
    return records


def _write_aggregate_csv(records: list[dict[str, Any]], aggregate_path: Path) -> None:
    """Compute mean/std/min/max across seeds for each condition."""
    import statistics
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in records:
        grouped[r["experiment"]].append(r)

    agg_rows = []
    for exp in EXPERIMENTS:
        items = grouped.get(exp, [])
        if not items:
            continue

        def _stats(key: str) -> tuple[float | str, float | str, float | str, float | str]:
            vals = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
            if not vals:
                return "N/A", "N/A", "N/A", "N/A"
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return round(mean, 6), round(std, 6), round(min(vals), 6), round(max(vals), 6)

        m_val, s_val, min_val, max_val = _stats("best_val_dice")
        m_test, s_test, min_test, max_test = _stats("test_dice")
        m_iou, s_iou, _, _ = _stats("test_iou")
        m_leaf, s_leaf, _, _ = _stats("leaf_dice")
        m_mae, s_mae, _, _ = _stats("coverage_mae")

        agg_rows.append({
            "experiment": exp,
            "n_runs": len(items),
            "mean_val_dice": m_val,
            "std_val_dice": s_val,
            "mean_test_dice": m_test,
            "std_test_dice": s_test,
            "min_test_dice": min_test,
            "max_test_dice": max_test,
            "mean_test_iou": m_iou,
            "std_test_iou": s_iou,
            "mean_leaf_dice": m_leaf,
            "std_leaf_dice": s_leaf,
            "mean_coverage_mae": m_mae,
            "std_coverage_mae": s_mae,
        })

    if agg_rows:
        with aggregate_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)
        print(f"[Aggregate CSV] Saved statistical summary -> {aggregate_path}", flush=True)


def build_job_list(
    experiments: list[str],
    seeds: list[int],
    smoke_test: bool = False,
) -> list[dict[str, Any]]:
    """Generate the full list of jobs to be executed."""
    if smoke_test:
        return list(SMOKE_TEST_JOBS)

    jobs = []
    # Interleave experiments so round-robin gives balanced workload across workers
    for seed in seeds:
        for exp in experiments:
            jobs.append({"experiment": exp, "seed": seed, "tag": f"seed{seed}"})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PJRT 4-Core Multi-Run Experiment Runner for TPU/GPU/CPU."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 4-worker smoke test: 1 epoch each on seeds 100-103 across the 4 conditions.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of random seeds per experiment (default: 10).",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
        help="First random seed (default: 42 -> seeds 42..51 for 10 runs).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit list of seeds to run (overrides --start-seed and --n-runs).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=EXPERIMENTS,
        help="Subset of experiments to run (default: all 4).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs (default: 1 for --smoke-test, or config default 100).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early stopping patience.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "xla", "cuda", "cpu"],
        help="Device to use. 'xla' enables 4-worker PJRT execution on TPU VM.",
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to base config file (default: configs/base.yaml).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Re-train all runs even if checkpoint/results already exist on disk.",
    )
    args = parser.parse_args()

    # Determine seeds
    if args.smoke_test:
        seeds = [100, 101, 102, 103]
        max_epochs = args.max_epochs if args.max_epochs is not None else 1
    elif args.seeds is not None:
        seeds = sorted(args.seeds)
        max_epochs = args.max_epochs
    else:
        seeds = list(range(args.start_seed, args.start_seed + args.n_runs))
        max_epochs = args.max_epochs

    jobs = build_job_list(args.experiments, seeds, smoke_test=args.smoke_test)
    skip_existing = not args.force_retrain

    # Resolve device mode
    use_xla = (args.device == "xla") or (args.device == "auto" and is_tpu_available())

    print("\n" + "=" * 70, flush=True)
    print("GLS MULTI-RUN EXPERIMENT ORCHESTRATOR", flush=True)
    print("=" * 70, flush=True)
    print(f"  Mode            : {'SMOKE TEST (4 cores × 1 job)' if args.smoke_test else 'FULL BENCHMARK'}", flush=True)
    print(f"  Total Jobs      : {len(jobs)}", flush=True)
    print(f"  Experiments     : {args.experiments}", flush=True)
    print(f"  Seeds           : {seeds}", flush=True)
    print(f"  Max Epochs      : {max_epochs if max_epochs is not None else 'config default'}", flush=True)
    print(f"  Device          : {'TPU (xla 4-core parallel)' if use_xla else 'CPU/GPU (sequential)'}", flush=True)
    print(f"  Skip Existing   : {skip_existing}", flush=True)
    print("=" * 70 + "\n", flush=True)

    if use_xla:
        try:
            import torch_xla.distributed.xla_multiprocessing as xmp
        except ImportError as exc:
            raise ImportError("TPU mode requested but torch_xla is not installed.") from exc

        print("Launching 4 TPU workers via xmp.spawn(nprocs=4)...\n", flush=True)
        xmp.spawn(
            _tpu_worker_entry,
            args=(jobs, max_epochs, args.patience, skip_existing, args.config),
            nprocs=4,
            start_method="spawn",
        )
    else:
        print("Running jobs sequentially (non-TPU mode)...\n", flush=True)
        for i, job in enumerate(jobs, start=1):
            print(f"\n[Job {i}/{len(jobs)}] {job['experiment']} | {job['tag']} (seed={job['seed']})", flush=True)
            execute_single_job(
                job,
                worker_id=0,
                max_epochs=max_epochs,
                patience=args.patience,
                skip_existing=skip_existing,
                config_path=args.config,
            )

    # Compile master results dataset across all runs
    compile_master_results()

    print("\n" + "=" * 70, flush=True)
    print("All tasks completed successfully!", flush=True)
    print("Master dataset saved to: outputs/results/all_runs.csv", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()

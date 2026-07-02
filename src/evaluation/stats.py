"""Paired statistical comparisons between experiments.

Proposal ref: §4.2.12 — Wilcoxon signed-rank test, p < 0.05, plus effect size.

Planned comparisons:
    - exp01_unet_noaug vs exp02_unet_aug
    - exp03_attnunet_noaug vs exp04_attnunet_aug
    - exp02_unet_aug vs exp04_attnunet_aug   (main comparison)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_compare(values_a: np.ndarray, values_b: np.ndarray) -> dict:
    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have the same length for a paired Wilcoxon test.")
    statistic, p_value = wilcoxon(values_a, values_b, zero_method="wilcox", correction=False)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def effect_size_r(statistic: float, n: int) -> float:
    if n <= 0:
        return 0.0

    expected = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance <= 0:
        return 0.0

    z = (statistic - expected) / math.sqrt(variance)
    return float(z / math.sqrt(n))


def _load_results(experiment: str, results_dir: Path) -> dict:
    path = results_dir / experiment / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file for {experiment}: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_values(experiment: str, results_dir: Path, metric: str) -> tuple[list[str], np.ndarray]:
    results = _load_results(experiment, results_dir)
    values = [row[metric] for row in results["per_image"]]
    sample_ids = [row["sample_id"] for row in results["per_image"]]
    return sample_ids, np.asarray(values, dtype=float)


def main(results_dir: str = "experiments", metric: str = "dice") -> None:
    results_path = Path(results_dir)
    comparisons = [
        ("exp01_unet_noaug", "exp02_unet_aug"),
        ("exp03_attnunet_noaug", "exp04_attnunet_aug"),
        ("exp02_unet_aug", "exp04_attnunet_aug"),
    ]

    for exp_a, exp_b in comparisons:
        ids_a, values_a = _metric_values(exp_a, results_path, metric)
        ids_b, values_b = _metric_values(exp_b, results_path, metric)
        if ids_a != ids_b:
            raise ValueError(
                f"Sample ordering mismatch between {exp_a} and {exp_b}. "
                "Ensure results.json per_image entries are aligned by sample_id."
            )

        compare = wilcoxon_compare(values_a, values_b)
        effect = effect_size_r(compare["statistic"], len(values_a))

        print(f"{exp_a} vs {exp_b} on '{metric}':")
        print(f"  n = {len(values_a)}")
        print(f"  statistic = {compare['statistic']:.2f}")
        print(f"  p-value = {compare['p_value']:.4g}")
        print(f"  r = {effect:.4f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paired Wilcoxon comparisons between experiment results.")
    parser.add_argument("--results-dir", default="experiments", help="Directory containing experiment folders")
    parser.add_argument("--metric", default="dice", help="Per-image metric to compare (dice, iou, precision, recall)")
    args = parser.parse_args()
    main(results_dir=args.results_dir, metric=args.metric)

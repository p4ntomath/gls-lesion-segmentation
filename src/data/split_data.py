"""Stratified train/val/test split by lesion coverage bin.

Proposal ref: §4.2.3 — 70/15/15, stratified into low/medium/high coverage
bins. Performed before augmentation, held fixed across all four experiments.

Writes: data/splits/train.txt, val.txt, test.txt (one sample id per line).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np # type: ignore
import pandas as pd # type: ignore
import yaml # type: ignore

from src.utils.progress import create_progress_bar


def assign_coverage_bins(df: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """Assign low/medium/high coverage bins using quantiles of lesion coverage.

    A rank-based qcut keeps the binning stable even when coverage values repeat.
    """
    if "lesion_coverage" not in df.columns:
        raise ValueError("manifest.csv is missing lesion_coverage. Run generate_masks.py first.")

    if df.empty:
        raise ValueError("manifest.csv is empty; cannot create train/val/test splits.")

    result = df.copy()
    ranked = result["lesion_coverage"].rank(method="first")

    try:
        bins = pd.qcut(ranked, q=n_bins, labels=False)
    except ValueError:
        # Fall back to equally spaced bins if the dataset is too small or too tied.
        bins = pd.cut(ranked, bins=n_bins, labels=False, include_lowest=True)

    labels = ["low", "medium", "high"][:n_bins]
    result["coverage_bin"] = pd.Categorical.from_codes(bins.astype(int), categories=labels[: len(pd.unique(bins.dropna()))], ordered=True)
    return result


def _allocate_counts(n_items: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    exact = np.array(ratios, dtype=float) * n_items
    base = np.floor(exact).astype(int)
    remainder = n_items - int(base.sum())
    if remainder:
        fractional = exact - base
        order = np.argsort(-fractional)
        for index in order[:remainder]:
            base[index] += 1
    return int(base[0]), int(base[1]), int(base[2])


def stratified_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into train/val/test sets while preserving coverage-bin balance."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}.")

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []

    with create_progress_bar(total=len(df.groupby("coverage_bin", sort=False)), desc="Splitting data") as bar:
        for _, group in df.groupby("coverage_bin", sort=False):
            shuffled = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**32 - 1)))
            n_train, n_val, n_test = _allocate_counts(len(shuffled), ratios)

            train_parts.append(shuffled.iloc[:n_train])
            val_parts.append(shuffled.iloc[n_train : n_train + n_val])
            test_parts.append(shuffled.iloc[n_train + n_val : n_train + n_val + n_test])
            bar.update(1)

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)

    return train_df, val_df, test_df


def _write_split(path: Path, sample_ids: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sample_ids.astype(str).tolist()) + "\n", encoding="utf-8")


def main(config_path: str = "configs/base.yaml") -> None:
    """Create and save fixed stratified splits for the dataset."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    data_config = config["data"]

    manifest_path = Path(paths["manifest_csv"])
    split_dir = Path(paths["split_dir"])
    ratios = tuple(data_config.get("split_ratios", [0.70, 0.15, 0.15]))
    seed = int(data_config.get("seed", 42))
    n_bins = int(data_config.get("stratify_bins", 3))

    manifest = pd.read_csv(manifest_path)
    manifest = assign_coverage_bins(manifest, n_bins=n_bins)

    train_df, val_df, test_df = stratified_split(manifest, ratios=ratios, seed=seed)

    _write_split(split_dir / "train.txt", train_df["sample_id"])
    _write_split(split_dir / "val.txt", val_df["sample_id"])
    _write_split(split_dir / "test.txt", test_df["sample_id"])

    print(f"Wrote splits to {split_dir}")
    print(f"  train: {len(train_df)}")
    print(f"  val:   {len(val_df)}")
    print(f"  test:  {len(test_df)}")
    print("  bins:", manifest["coverage_bin"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

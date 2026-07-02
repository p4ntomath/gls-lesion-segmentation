"""Leaf-area GLS coverage calculation.

Proposal ref: §4.2.9 (eq. 1-2)
Uses leaf masks produced once by pipelines/generate_leaf_masks.py
(data/processed/leaf_masks/) — this module does not run SAM itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def leaf_area_coverage(lesion_mask: np.ndarray, leaf_mask: np.ndarray) -> float:
    """Compute leaf-area GLS coverage as a percentage of leaf pixels."""
    lesion = np.asarray(lesion_mask) > 0
    leaf = np.asarray(leaf_mask) > 0

    leaf_area = int(np.sum(leaf))
    if leaf_area == 0:
        return 0.0

    intersection = int(np.sum(lesion & leaf))
    return float(intersection / leaf_area * 100.0)


def coverage_for_split(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    leaf_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Return predicted and reference coverage for each sample in the split."""
    rows = []
    missing = sorted(set(gt_masks) - set(leaf_masks))
    if missing:
        raise ValueError(f"Missing leaf masks for sample(s): {missing}")

    for sample_id in sorted(gt_masks):
        if sample_id not in pred_masks:
            raise ValueError(f"Missing prediction for sample {sample_id}")

        pred_cov = leaf_area_coverage(pred_masks[sample_id], leaf_masks[sample_id])
        ref_cov = leaf_area_coverage(gt_masks[sample_id], leaf_masks[sample_id])
        rows.append(
            {
                "sample_id": sample_id,
                "predicted_coverage": pred_cov,
                "reference_coverage": ref_cov,
            }
        )

    return pd.DataFrame(rows)

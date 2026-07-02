"""
Parse data/raw/annotations/dataset.json (Segments.ai export) into a flat
manifest of usable samples.

Proposal ref: §4.1.1, §4.2.3

This step ONLY extracts metadata + URLs — it does not download anything and
does not touch pixel data. That happens in src/data/generate_masks.py.

Output columns (data/processed/manifest.csv):
    sample_id       -- stem of the image filename, e.g. "1925" for "1925.JPG"
    uuid            -- Segments.ai sample uuid
    image_url       -- RGB image, hosted on Segments.ai's S3
    mask_url        -- segmentation_bitmap url (32-bit RGBA PNG, instance ids
                        encoded in the RGB channels, NOT a plain grayscale mask
                        -- see src/data/generate_masks.py for the decode step)
    label_status    -- LABELED or REVIEWED (already filtered)
    num_instances   -- number of GLS lesion instances annotated for this image
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from src.utils.progress import create_progress_bar

DEFAULT_KEEP_STATUSES = ("LABELED", "REVIEWED")


def load_raw_json(path: str) -> dict:
    """Load the Segments.ai export JSON."""
    with open(path, "r") as f:
        return json.load(f)


def extract_labeled_samples(data: dict, keep_statuses=DEFAULT_KEEP_STATUSES) -> list[dict]:
    """
    Walk data['dataset']['samples'] and keep only samples with a
    ground-truth label whose status is in keep_statuses.

    Samples with labels['ground-truth'] is None, or missing entirely, or
    with a status not in keep_statuses (e.g. 'UNLABELED'), are dropped.
    """
    samples = data["dataset"]["samples"]
    kept = []
    for s in samples:
        gt = s.get("labels", {}).get("ground-truth")
        if gt is None:
            continue
        if gt.get("label_status") not in keep_statuses:
            continue
        kept.append(s)
    return kept


def _sample_id_from_name(name: str) -> str:
    """'1925.JPG' -> '1925'. Falls back to the full name if there's no dot."""
    return Path(name).stem


def build_manifest(samples: list[dict]) -> pd.DataFrame:
    """Turn the filtered sample list into a flat DataFrame."""
    rows = []
    with create_progress_bar(total=len(samples), desc="Parsing dataset") as bar:
        for s in samples:
            gt = s["labels"]["ground-truth"]
            attrs = gt["attributes"]
            annotations = attrs.get("annotations", [])
            rows.append(
                {
                    "sample_id": _sample_id_from_name(s["name"]),
                    "uuid": s["uuid"],
                    "image_url": s["attributes"]["image"]["url"],
                    "mask_url": attrs["segmentation_bitmap"]["url"],
                    "label_status": gt["label_status"],
                    "num_instances": len(annotations),
                    "annotations": json.dumps(annotations, separators=(",", ":")),
                }
            )
            bar.update(1)
    df = pd.DataFrame(rows)

    dupes = df["sample_id"][df["sample_id"].duplicated()].unique()
    if len(dupes) > 0:
        raise ValueError(
            f"Duplicate sample_id(s) after stripping file extension: {list(dupes)}. "
            "Two source images share a filename stem -- resolve before continuing "
            "(e.g. fall back to `uuid` as sample_id)."
        )
    return df


def main(config_path: str = "configs/base.yaml") -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_json_path = config["paths"]["raw_annotations"]
    manifest_path = config["paths"]["manifest_csv"]
    keep_statuses = config["data"]["label_statuses"]

    data = load_raw_json(raw_json_path)
    samples = extract_labeled_samples(data, keep_statuses)
    manifest = build_manifest(samples)

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    print(f"Parsed {len(manifest)} labeled samples -> {manifest_path}")
    print(f"  (dropped {len(data['dataset']['samples']) - len(manifest)} unlabeled/other-status samples)")


if __name__ == "__main__":
    main()

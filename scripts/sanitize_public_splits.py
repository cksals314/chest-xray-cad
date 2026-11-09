#!/usr/bin/env python3
"""Back up full split manifests and retain only model-required columns.

Run from any directory:
    python scripts/sanitize_public_splits.py --dry-run
    python scripts/sanitize_public_splits.py --apply

The original CSVs are copied to archive/full_metadata_splits/ before any
in-place replacement. Patient ID is retained for patient-level leakage checks.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.constants import DISEASES
SPLIT_DIR = PROJECT_DIR / "splits"
BACKUP_DIR = PROJECT_DIR / "archive" / "full_metadata_splits"
REQUIRED_COLUMNS = ["Image Index", "Patient ID", *DISEASES]


def target_files() -> list[Path]:
    paths = [SPLIT_DIR / "final_trainval.csv", SPLIT_DIR / "final_test.csv"]
    for fold in range(1, 6):
        paths.extend([
            SPLIT_DIR / f"fold_{fold}" / "train.csv",
            SPLIT_DIR / f"fold_{fold}" / "val.csv",
        ])
    return [path for path in paths if path.is_file()]


def validate_frame(path: Path, frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    if frame["Image Index"].isna().any() or frame["Patient ID"].isna().any():
        raise ValueError(f"{path}: missing Image Index or Patient ID")
    if frame["Image Index"].duplicated().any():
        raise ValueError(f"{path}: duplicate Image Index")
    labels = frame[DISEASES].to_numpy()
    if not pd.Series(labels.reshape(-1)).isin([0, 1]).all():
        raise ValueError(f"{path}: disease labels must be binary 0/1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up and rewrite manifests")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list planned changes")
    args = parser.parse_args()
    if args.apply == args.dry_run:
        parser.error("Choose exactly one of --dry-run or --apply")

    paths = target_files()
    if not paths:
        raise FileNotFoundError(f"No split CSVs found under {SPLIT_DIR}")

    print(f"Required columns ({len(REQUIRED_COLUMNS)}): {REQUIRED_COLUMNS}")
    for path in paths:
        frame = pd.read_csv(
            path,
            dtype={"Image Index": "string", "Patient ID": "string"},
        )
        validate_frame(path, frame)
        already_slim = frame.columns.tolist() == REQUIRED_COLUMNS
        print(
            f"{path.relative_to(PROJECT_DIR)}: rows={len(frame):,}, "
            f"columns={len(frame.columns)} -> {len(REQUIRED_COLUMNS)}, "
            f"{'already minimal' if already_slim else 'would slim'}"
        )
        if not args.apply or already_slim:
            continue

        relative = path.relative_to(SPLIT_DIR)
        backup = BACKUP_DIR / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            raise FileExistsError(
                f"Backup already exists; refusing to overwrite it: {backup}"
            )
        shutil.copy2(path, backup)

        temp = path.with_suffix(path.suffix + ".tmp")
        frame[REQUIRED_COLUMNS].to_csv(temp, index=False)
        # Read back before replacing the original.
        check = pd.read_csv(
            temp,
            dtype={"Image Index": "string", "Patient ID": "string"},
        )
        validate_frame(temp, check)
        if len(check) != len(frame) or check.columns.tolist() != REQUIRED_COLUMNS:
            temp.unlink(missing_ok=True)
            raise RuntimeError(f"Read-back validation failed: {path}")
        temp.replace(path)
        print(f"  backed up -> {backup.relative_to(PROJECT_DIR)}")

    print("\nDry run complete." if args.dry_run else "\nManifest sanitization complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create patient-wise 5-fold GroupKFold splits for NIH ChestX-ray14.

Expected project layout:
    cxr-cad/
      data/
        Data_Entry_2017.csv
        train_val_list_NIH.txt
        test_list_NIH.txt
        images-224/
      src/prepare_data.py

Outputs:
    splits/fold_1/{train,val}.csv ... fold_5/{train,val}.csv
    reports/groupkfold/fold_summary.csv
    reports/groupkfold/fold_integrity_report.csv
    reports/groupkfold/fold_disease_distribution.csv

Each split CSV contains only the image filename, pseudonymous Patient ID,
and 14 binary disease-label columns required by the training pipeline.
"No Finding" is represented by fourteen zeros.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
METADATA_CSV = DATA_DIR / "Data_Entry_2017.csv"
TRAIN_VAL_LIST = DATA_DIR / "train_val_list_NIH.txt"
TEST_LIST = DATA_DIR / "test_list_NIH.txt"

# Output directory for the final GroupKFold manifests. Re-running this script
# rewrites the fold CSV files, so back up existing splits before regeneration.
OUTPUT_SPLIT_DIR = PROJECT_DIR / "splits"
REPORT_DIR = PROJECT_DIR / "reports" / "groupkfold"
N_SPLITS = 5

try:
    from .constants import DISEASES
except ImportError:  # Support running scripts as: python src/<script>.py
    from constants import DISEASES

IMAGE_COL = "Image Index"
PATIENT_COL = "Patient ID"
FINDINGS_COL = "Finding Labels"
# Minimal manifest schema used by Dataset, training, OOF, and evaluation code.
SPLIT_COLUMNS = [IMAGE_COL, PATIENT_COL, *DISEASES]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_image_list(path: Path) -> list[str]:
    """Read image filenames from an official NIH split-list text file."""
    if not path.exists():
        raise FileNotFoundError(f"Split-list file not found: {path}")

    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No image filenames found in: {path}")

    duplicates = pd.Series(names)[pd.Series(names).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            f"Duplicate filenames found in {path.name}; examples: {duplicates[:5]}"
        )
    return names


def add_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add one 0/1 column per disease, based on Finding Labels."""
    if FINDINGS_COL not in df.columns:
        raise KeyError(f"Required metadata column is missing: {FINDINGS_COL}")

    findings = df[FINDINGS_COL].fillna("").astype(str)
    finding_sets = findings.map(
        lambda value: {label.strip() for label in value.split("|") if label.strip()}
    )

    for disease in DISEASES:
        df[disease] = finding_sets.map(lambda labels: int(disease in labels)).astype(
            np.uint8
        )

    return df


def disease_distribution_rows(
    split_name: str,
    fold_number: int | str,
    split_df: pd.DataFrame,
    reference_counts: dict[str, int],
    reference_prevalence: dict[str, float],
) -> list[dict]:
    """Create long-format disease prevalence rows for one data split."""
    n_images = len(split_df)
    n_patients = split_df[PATIENT_COL].nunique()
    rows = []

    for disease in DISEASES:
        positive_count = int(split_df[disease].sum())
        prevalence = positive_count / n_images if n_images else np.nan
        ref_prev = reference_prevalence[disease]

        rows.append(
            {
                "Fold": fold_number,
                "Split": split_name,
                "N_Images": n_images,
                "N_Patients": n_patients,
                "Disease": disease,
                "Positive_Count": positive_count,
                "Prevalence": prevalence,
                "Reference_TrainVal_Positive_Count": reference_counts[disease],
                "Reference_TrainVal_Prevalence": ref_prev,
                "Prevalence_Difference_pp": (prevalence - ref_prev) * 100,
            }
        )

    return rows


def ensure_required_columns(df: pd.DataFrame) -> None:
    required = {IMAGE_COL, PATIENT_COL, FINDINGS_COL}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Metadata is missing required columns: {sorted(missing)}")


def main() -> None:
    print("=" * 72)
    print("NIH ChestX-ray14: Patient-wise 5-Fold GroupKFold")
    print("=" * 72)

    # 1) Read official image lists.
    train_val_names = read_image_list(TRAIN_VAL_LIST)
    test_names = read_image_list(TEST_LIST)

    overlap_official_lists = set(train_val_names) & set(test_names)
    if overlap_official_lists:
        raise ValueError(
            "Official Train/Val and Test lists overlap; examples: "
            f"{sorted(overlap_official_lists)[:5]}"
        )

    print(f"Official Train/Val images: {len(train_val_names):,}")
    print(f"Official Test images:      {len(test_names):,}")

    # 2) Read metadata and verify one metadata row per image.
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    metadata = pd.read_csv(
        METADATA_CSV,
        dtype={IMAGE_COL: "string", PATIENT_COL: "string"},
    )
    ensure_required_columns(metadata)

    metadata[IMAGE_COL] = metadata[IMAGE_COL].astype("string").str.strip()
    metadata[PATIENT_COL] = metadata[PATIENT_COL].astype("string").str.strip()

    if metadata[IMAGE_COL].isna().any() or (metadata[IMAGE_COL] == "").any():
        raise ValueError("Metadata contains missing/empty image filenames.")
    if metadata[PATIENT_COL].isna().any() or (metadata[PATIENT_COL] == "").any():
        raise ValueError("Metadata contains missing/empty Patient ID values.")

    duplicated_metadata = metadata.loc[
        metadata[IMAGE_COL].duplicated(keep=False), IMAGE_COL
    ].unique()
    if len(duplicated_metadata):
        raise ValueError(
            "Metadata contains duplicate Image Index rows; examples: "
            f"{duplicated_metadata[:5].tolist()}"
        )

    metadata_by_name = metadata.set_index(IMAGE_COL, drop=False)
    missing_metadata = sorted(set(train_val_names) - set(metadata_by_name.index))
    if missing_metadata:
        raise ValueError(
            "Some official Train/Val images are absent from Data_Entry_2017.csv; "
            f"examples: {missing_metadata[:10]}"
        )

    # Select only official Train/Val images. Test data is not used to make folds.
    train_val = metadata_by_name.loc[train_val_names].reset_index(drop=True).copy()
    train_val = add_binary_labels(train_val)

    # Confirm each listed image maps to exactly one patient.
    if train_val[IMAGE_COL].nunique() != len(train_val_names):
        raise AssertionError("Image count changed while constructing Train/Val data.")

    n_patients = train_val[PATIENT_COL].nunique()
    if n_patients < N_SPLITS:
        raise ValueError(
            f"Only {n_patients} patients found; cannot create {N_SPLITS} folds."
        )

    print(f"Metadata rows used:        {len(train_val):,}")
    print(f"Unique Train/Val patients: {n_patients:,}")
    print("Label columns added:       14")
    print("No Finding representation: fourteen zeros")

    # 3) Calculate reference disease prevalence on the full official Train/Val set.
    reference_counts = {d: int(train_val[d].sum()) for d in DISEASES}
    reference_prevalence = {
        d: reference_counts[d] / len(train_val) for d in DISEASES
    }

    # 4) GroupKFold: all images belonging to one patient remain in one side.
    splitter = GroupKFold(n_splits=N_SPLITS)
    groups = train_val[PATIENT_COL].astype(str).to_numpy()

    OUTPUT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    integrity_rows: list[dict] = []
    distribution_rows: list[dict] = []
    validation_image_occurrences: Counter[str] = Counter()
    validation_patient_occurrences: Counter[str] = Counter()

    # Include original Train/Val as the reference row set in the distribution file.
    distribution_rows.extend(
        disease_distribution_rows(
            split_name="Original_TrainVal",
            fold_number="Original",
            split_df=train_val,
            reference_counts=reference_counts,
            reference_prevalence=reference_prevalence,
        )
    )

    for fold_idx, (train_indices, val_indices) in enumerate(
        splitter.split(train_val, groups=groups), start=1
    ):
        fold_dir = OUTPUT_SPLIT_DIR / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = train_val.iloc[train_indices].reset_index(drop=True).copy()
        val_df = train_val.iloc[val_indices].reset_index(drop=True).copy()

        train_patients = set(train_df[PATIENT_COL].astype(str))
        val_patients = set(val_df[PATIENT_COL].astype(str))
        patient_overlap = train_patients & val_patients

        if patient_overlap:
            raise AssertionError(
                f"Fold {fold_idx}: patient leakage detected: "
                f"{sorted(patient_overlap)[:10]}"
            )

        train_images = set(train_df[IMAGE_COL].astype(str))
        val_images = set(val_df[IMAGE_COL].astype(str))
        if train_images & val_images:
            raise AssertionError(f"Fold {fold_idx}: image leakage detected.")
        if train_images | val_images != set(train_val[IMAGE_COL].astype(str)):
            raise AssertionError(f"Fold {fold_idx}: images missing from train/val union.")

        # Save only columns required by the model pipeline. The full NIH
        # metadata remains available in the source dataset under data/.
        train_df[SPLIT_COLUMNS].to_csv(fold_dir / "train.csv", index=False)
        val_df[SPLIT_COLUMNS].to_csv(fold_dir / "val.csv", index=False)

        for image_name in val_df[IMAGE_COL].astype(str):
            validation_image_occurrences[image_name] += 1
        for patient_id in val_df[PATIENT_COL].astype(str).unique():
            validation_patient_occurrences[patient_id] += 1

        summary_rows.append(
            {
                "Fold": fold_idx,
                "Train_Images": len(train_df),
                "Validation_Images": len(val_df),
                "Train_Patients": len(train_patients),
                "Validation_Patients": len(val_patients),
                "Patient_Overlap": len(patient_overlap),
                "Train_Image_Positive_Total": int(train_df[DISEASES].to_numpy().sum()),
                "Validation_Image_Positive_Total": int(val_df[DISEASES].to_numpy().sum()),
            }
        )

        integrity_rows.append(
            {
                "Fold": fold_idx,
                "Train_Images": len(train_df),
                "Validation_Images": len(val_df),
                "Train_Patients": len(train_patients),
                "Validation_Patients": len(val_patients),
                "Patient_Overlap_Count": len(patient_overlap),
                "Image_Overlap_Count": len(train_images & val_images),
                "Train_Val_Partition_Check": (
                    train_images | val_images == set(train_val[IMAGE_COL].astype(str))
                ),
                "Patient_Leakage_Check": len(patient_overlap) == 0,
            }
        )

        distribution_rows.extend(
            disease_distribution_rows(
                split_name="Train",
                fold_number=fold_idx,
                split_df=train_df,
                reference_counts=reference_counts,
                reference_prevalence=reference_prevalence,
            )
        )
        distribution_rows.extend(
            disease_distribution_rows(
                split_name="Validation",
                fold_number=fold_idx,
                split_df=val_df,
                reference_counts=reference_counts,
                reference_prevalence=reference_prevalence,
            )
        )

        print(
            f"Fold {fold_idx}: "
            f"Train {len(train_df):,} images / {len(train_patients):,} patients | "
            f"Val {len(val_df):,} images / {len(val_patients):,} patients | "
            "patient overlap = 0"
        )

    # 5) Global integrity checks: every Train/Val image and patient is in
    # exactly one validation fold across the five folds.
    expected_images = set(train_val[IMAGE_COL].astype(str))
    expected_patients = set(train_val[PATIENT_COL].astype(str).unique())

    all_images_once = (
        set(validation_image_occurrences) == expected_images
        and all(count == 1 for count in validation_image_occurrences.values())
    )
    all_patients_once = (
        set(validation_patient_occurrences) == expected_patients
        and all(count == 1 for count in validation_patient_occurrences.values())
    )

    if not all_images_once:
        missing = expected_images - set(validation_image_occurrences)
        repeated = [k for k, v in validation_image_occurrences.items() if v != 1]
        raise AssertionError(
            "Validation image coverage failed. "
            f"Missing examples={list(missing)[:5]}, repeated examples={repeated[:5]}"
        )
    if not all_patients_once:
        missing = expected_patients - set(validation_patient_occurrences)
        repeated = [k for k, v in validation_patient_occurrences.items() if v != 1]
        raise AssertionError(
            "Validation patient coverage failed. "
            f"Missing examples={list(missing)[:5]}, repeated examples={repeated[:5]}"
        )

    # Add the global checks to each Fold row for a self-contained report.
    for row in integrity_rows:
        row["Every_Image_Validation_Exactly_Once"] = all_images_once
        row["Every_Patient_Validation_Exactly_Once"] = all_patients_once
        row["Overall_Integrity_Pass"] = (
            row["Patient_Leakage_Check"]
            and row["Train_Val_Partition_Check"]
            and row["Image_Overlap_Count"] == 0
            and all_images_once
            and all_patients_once
        )

    # 6) Save reports.
    summary_df = pd.DataFrame(summary_rows)
    integrity_df = pd.DataFrame(integrity_rows)
    distribution_df = pd.DataFrame(distribution_rows)

    summary_path = REPORT_DIR / "fold_summary.csv"
    integrity_path = REPORT_DIR / "fold_integrity_report.csv"
    distribution_path = REPORT_DIR / "fold_disease_distribution.csv"

    summary_df.to_csv(summary_path, index=False)
    integrity_df.to_csv(integrity_path, index=False)
    distribution_df.to_csv(distribution_path, index=False)

    run_info = {
        "split_method": "sklearn.model_selection.GroupKFold",
        "n_splits": N_SPLITS,
        "group_column": PATIENT_COL,
        "official_source_list": TRAIN_VAL_LIST.name,
        "test_list_used_for_split": False,
        "n_train_val_images": len(train_val),
        "n_train_val_patients": n_patients,
        "n_test_images_in_official_list": len(test_names),
        "n_diseases": len(DISEASES),
        "no_finding_representation": "14 binary zeros",
        "brightness_candidates_removed": 0,
        "all_validation_images_exactly_once": bool(all_images_once),
        "all_validation_patients_exactly_once": bool(all_patients_once),
        "output_split_directory": str(OUTPUT_SPLIT_DIR.relative_to(PROJECT_DIR)),
    }
    (OUTPUT_SPLIT_DIR / "split_metadata.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 7) Final console summary.
    print("\n" + "=" * 72)
    print("SPLIT VALIDATION RESULT")
    print("=" * 72)
    print(f"Patient leakage across Train/Val:  PASS (0 overlap in all folds)")
    print(f"Every image used as Val once:       {'PASS' if all_images_once else 'FAIL'}")
    print(f"Every patient used as Val once:     {'PASS' if all_patients_once else 'FAIL'}")
    print(f"All fold integrity checks:          {'PASS' if integrity_df['Overall_Integrity_Pass'].all() else 'FAIL'}")
    print(f"\nNew splits: {OUTPUT_SPLIT_DIR}")
    print(f"Fold summary: {summary_path}")
    print(f"Integrity report: {integrity_path}")
    print(f"Disease distribution: {distribution_path}")
    print("\nGroupKFold splits and reports saved successfully.")

    if not integrity_df["Overall_Integrity_Pass"].all():
        raise AssertionError("At least one integrity check did not pass.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise

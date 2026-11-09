#!/usr/bin/env python3
"""Select disease-wise operating thresholds on OOF data and evaluate on NIH Test.

Threshold selection uses only OOF labels and predictions. The official Test set is
read only after thresholds have been selected, and is used only for final metrics.

Primary score paths:
  - densenet121: per-disease Platt-calibrated OOF probability
  - efficientnet_b0: per-disease Platt-calibrated OOF probability
  - soft_voting_ensemble: arithmetic mean of the two models' Platt probabilities

The ensemble construction is checked against the already-exported Test prediction
columns `soft_voting_ensemble_platt_<Disease>`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


PROJECT_DIR = Path(__file__).resolve().parents[1]
try:
    from .constants import DISEASES
except ImportError:  # Support running scripts as: python src/<script>.py
    from constants import DISEASES

DEFAULT_OOF = {
    "densenet121": Path("reports/oof/densenet121/densenet121_oof.csv.gz"),
    "efficientnet_b0": Path("reports/oof/efficientnet_b0/efficientnet_b0_oof.csv.gz"),
}
DEFAULT_PLATT = {
    "densenet121": Path("reports/calibration/densenet121/platt_parameters.csv"),
    "efficientnet_b0": Path("reports/calibration/efficientnet_b0/platt_parameters.csv"),
}
DEFAULT_TEST = Path("reports/test_evaluation/test_predictions_public.csv.gz")
DEFAULT_OUTPUT = Path("reports/operating_points")

MODEL_LABELS = {
    "densenet121": "DenseNet121 + Platt",
    "efficientnet_b0": "EfficientNet-B0 + Platt",
    "soft_voting_ensemble": "Soft Voting Ensemble + Platt",
}

CRITERIA = ("YoudenJ", "Sensitivity90", "Specificity90")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def require_columns(df: pd.DataFrame, required: List[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_platt_parameters(path: Path) -> Dict[str, Tuple[float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Platt parameter file not found: {path}")
    params_df = pd.read_csv(path)
    require_columns(params_df, ["Disease", "Slope", "Intercept"], path)
    if params_df["Disease"].duplicated().any():
        raise ValueError(f"Duplicate disease rows in Platt parameter file: {path}")
    params = {}
    for _, row in params_df.iterrows():
        disease = str(row["Disease"])
        slope = float(row["Slope"])
        intercept = float(row["Intercept"])
        if not np.isfinite(slope) or not np.isfinite(intercept):
            raise ValueError(f"Non-finite Platt parameter for {disease} in {path}")
        if slope <= 0:
            raise ValueError(
                f"Platt slope for {disease} is not positive ({slope}). "
                "Threshold ranking may not be preserved; inspect calibration parameters."
            )
        params[disease] = (slope, intercept)
    missing = sorted(set(DISEASES) - set(params))
    if missing:
        raise ValueError(f"Platt parameters missing diseases in {path}: {missing}")
    return params


def load_oof_model(model: str) -> pd.DataFrame:
    path = PROJECT_DIR / DEFAULT_OOF[model]
    if not path.exists():
        # Backward compatibility for locally regenerated, uncompressed OOF files.
        legacy_path = path.with_suffix("")
        if legacy_path.exists():
            path = legacy_path
        else:
            raise FileNotFoundError(f"OOF file not found: {path}")
    df = pd.read_csv(path)
    required = ["Image Index", "Patient ID", "Fold", *DISEASES]
    for disease in DISEASES:
        required.append(f"logit_{disease}")
    require_columns(df, required, path)

    if df["Image Index"].duplicated().any():
        raise ValueError(f"Duplicate Image Index rows in {path}")
    if df["Patient ID"].isna().any() or df["Image Index"].isna().any():
        raise ValueError(f"Missing patient/image identifiers in {path}")
    labels = df[DISEASES].to_numpy(dtype=np.float64)
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError(f"Labels in {path} must be binary 0/1")
    if set(pd.to_numeric(df["Fold"], errors="raise").astype(int).unique()) != {1, 2, 3, 4, 5}:
        raise ValueError(f"Expected OOF folds 1..5 in {path}")
    return df


def build_oof_scores() -> Tuple[pd.DataFrame, Dict[str, Dict[str, Tuple[float, float]]]]:
    oof_dense = load_oof_model("densenet121")
    oof_eff = load_oof_model("efficientnet_b0")

    # Join by stable identifiers rather than assuming CSV row order is identical.
    merged = oof_dense[["Image Index", "Patient ID", "Fold", *DISEASES]].merge(
        oof_eff[["Image Index", "Patient ID", "Fold", *DISEASES]],
        on="Image Index",
        how="outer",
        suffixes=("_dense", "_eff"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("DenseNet and EfficientNet OOF Image Index sets do not match")
    merged = merged.drop(columns="_merge")
    if not (merged["Patient ID_dense"].astype(str) == merged["Patient ID_eff"].astype(str)).all():
        raise ValueError("Patient ID mismatch between model OOF files")
    if not (pd.to_numeric(merged["Fold_dense"]).astype(int) == pd.to_numeric(merged["Fold_eff"]).astype(int)).all():
        raise ValueError("Fold assignment mismatch between model OOF files")
    for disease in DISEASES:
        if not (merged[f"{disease}_dense"].astype(int) == merged[f"{disease}_eff"].astype(int)).all():
            raise ValueError(f"Ground-truth label mismatch between OOF files: {disease}")

    merged = merged.rename(columns={
        "Patient ID_dense": "Patient ID",
        "Fold_dense": "Fold",
    })
    y = merged[[f"{d}_dense" for d in DISEASES]].to_numpy(dtype=np.uint8)
    merged[DISEASES] = y

    all_params: Dict[str, Dict[str, Tuple[float, float]]] = {}
    score_arrays: Dict[str, np.ndarray] = {}
    for model, source in (("densenet121", oof_dense), ("efficientnet_b0", oof_eff)):
        params_path = PROJECT_DIR / DEFAULT_PLATT[model]
        params = load_platt_parameters(params_path)
        all_params[model] = params
        # Align OOF logits to merged Image Index order.
        indexed = source.set_index("Image Index").loc[merged["Image Index"]]
        logits = np.column_stack([indexed[f"logit_{d}"].to_numpy(dtype=np.float64) for d in DISEASES])
        calibrated = np.column_stack([
            sigmoid(params[d][0] * logits[:, j] + params[d][1])
            for j, d in enumerate(DISEASES)
        ])
        score_arrays[model] = calibrated

    score_arrays["soft_voting_ensemble"] = (
        score_arrays["densenet121"] + score_arrays["efficientnet_b0"]
    ) / 2.0

    # Attach scores in a tidy long-form table for threshold selection.
    oof_scores = merged[["Image Index", "Patient ID", "Fold", *DISEASES]].copy()
    for model, values in score_arrays.items():
        for j, disease in enumerate(DISEASES):
            oof_scores[f"{model}_platt_{disease}"] = values[:, j]

    if len(oof_scores) != 86524:
        raise ValueError(f"Expected 86,524 OOF rows, found {len(oof_scores):,}")
    if oof_scores["Image Index"].nunique() != len(oof_scores):
        raise ValueError("OOF Image Index is not unique after merge")
    return oof_scores, all_params


def choose_thresholds(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite probabilities encountered during threshold selection")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Operating point labels must be binary")
    if np.unique(y_true).size != 2:
        raise ValueError("Both positive and negative samples are required for ROC thresholds")

    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    specificity = 1.0 - fpr
    j_values = tpr + specificity - 1.0

    # Youden's J: for ties, choose the highest threshold (more conservative).
    best_j = np.nanmax(j_values)
    j_idx_candidates = np.flatnonzero(np.isclose(j_values, best_j, rtol=1e-10, atol=1e-12))
    j_idx = j_idx_candidates[np.argmax(thresholds[j_idx_candidates])]

    # Sensitivity >= 0.90: choose the highest threshold meeting the target.
    sens_candidates = np.flatnonzero(tpr >= 0.90 - 1e-12)
    if len(sens_candidates) == 0:
        raise RuntimeError("No ROC threshold satisfies Sensitivity >= 0.90")
    sens_idx = sens_candidates[np.argmax(thresholds[sens_candidates])]

    # Specificity >= 0.90: choose the lowest threshold meeting the target,
    # thereby maximizing sensitivity subject to the specificity constraint.
    spec_candidates = np.flatnonzero(specificity >= 0.90 - 1e-12)
    if len(spec_candidates) == 0:
        raise RuntimeError("No ROC threshold satisfies Specificity >= 0.90")
    spec_idx = spec_candidates[np.argmin(thresholds[spec_candidates])]

    result = {}
    for name, idx in (("YoudenJ", j_idx), ("Sensitivity90", sens_idx), ("Specificity90", spec_idx)):
        result[name] = {
            "Threshold": float(thresholds[idx]),
            "OOF_Sensitivity": float(tpr[idx]),
            "OOF_Specificity": float(specificity[idx]),
            "OOF_YoudenJ": float(j_values[idx]),
        }
    return result


def confusion_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    pred = scores >= threshold
    positive = y_true == 1
    negative = ~positive
    tp = int(np.sum(pred & positive))
    fp = int(np.sum(pred & negative))
    tn = int(np.sum((~pred) & negative))
    fn = int(np.sum((~pred) & positive))

    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    youden = sensitivity + specificity - 1 if np.isfinite(sensitivity) and np.isfinite(specificity) else np.nan

    return {
        "Test_Sensitivity": sensitivity,
        "Test_Specificity": specificity,
        "Test_PPV": ppv,
        "Test_NPV": npv,
        "Test_F1": f1,
        "Test_YoudenJ": youden,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Test_Positive_Count": int(np.sum(positive)),
        "Test_Negative_Count": int(np.sum(negative)),
    }


def load_test_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test predictions file not found: {path}")
    test = pd.read_csv(path)
    required = ["Image Index", "Patient ID", *DISEASES]
    for model in ("densenet121", "efficientnet_b0", "soft_voting_ensemble"):
        required.extend([f"{model}_platt_{d}" for d in DISEASES])
    require_columns(test, required, path)
    if test["Image Index"].duplicated().any():
        raise ValueError("Duplicate Test Image Index rows")
    if len(test) != 25596:
        raise ValueError(f"Expected 25,596 Test rows, found {len(test):,}")
    for model in ("densenet121", "efficientnet_b0", "soft_voting_ensemble"):
        values = test[[f"{model}_platt_{d}" for d in DISEASES]].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite Test Platt probabilities for {model}")
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"Test probabilities outside [0,1] for {model}")

    # Confirm the stored soft-voting Platt scores equal the mean of model-specific
    # Platt probabilities. This prevents selecting thresholds on a mismatched score.
    dense = test[[f"densenet121_platt_{d}" for d in DISEASES]].to_numpy(dtype=np.float64)
    eff = test[[f"efficientnet_b0_platt_{d}" for d in DISEASES]].to_numpy(dtype=np.float64)
    ens = test[[f"soft_voting_ensemble_platt_{d}" for d in DISEASES]].to_numpy(dtype=np.float64)
    max_diff = float(np.max(np.abs(ens - (dense + eff) / 2.0)))
    if max_diff > 1e-5:
        raise ValueError(
            "Test soft_voting_ensemble_platt is not the arithmetic mean of the two "
            f"model Platt scores (max difference={max_diff:.6g}). Inspect evaluate_test.py."
        )
    print(f"Test ensemble consistency: PASS (max abs diff={max_diff:.3e})")
    return test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select operating thresholds on OOF predictions and evaluate on official Test."
    )
    parser.add_argument("--test-csv", type=Path, default=PROJECT_DIR / DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Prefer the compact public artifact; fall back to full local predictions
    # generated by evaluate_test.py when the public file is not present.
    if not args.test_csv.is_file() and args.test_csv == PROJECT_DIR / DEFAULT_TEST:
        legacy_test = PROJECT_DIR / "reports/test_evaluation/test_predictions.csv"
        if legacy_test.is_file():
            args.test_csv = legacy_test

    print("=" * 76)
    print("CXR14 OPERATING POINT ANALYSIS")
    print("=" * 76)
    print("Threshold selection: OOF only (Platt-calibrated probabilities)")
    print("Official Test predictions/labels will be opened only after thresholds are fixed.")

    # Phase 1: select and freeze every threshold using OOF data only.
    oof, _ = build_oof_scores()
    threshold_rows = []
    selected_thresholds: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    model_score_prefixes = {
        "densenet121": "densenet121_platt",
        "efficientnet_b0": "efficientnet_b0_platt",
        "soft_voting_ensemble": "soft_voting_ensemble_platt",
    }

    for model in model_score_prefixes:
        print(f"\n[{model}] selecting thresholds from OOF...")
        for disease in DISEASES:
            y_oof = oof[disease].to_numpy(dtype=np.uint8)
            s_oof = oof[f"{model}_platt_{disease}"].to_numpy(dtype=np.float64)
            selections = choose_thresholds(y_oof, s_oof)

            for criterion in CRITERIA:
                selected = selections[criterion]
                selected_thresholds[(model, disease, criterion)] = selected
                threshold_rows.append({
                    "Model": model,
                    "Disease": disease,
                    "Criterion": criterion,
                    **selected,
                    "OOF_Positive_Count": int(y_oof.sum()),
                    "OOF_Negative_Count": int((1 - y_oof).sum()),
                })
        print(f"[{model}] thresholds fixed")

    threshold_df = pd.DataFrame(threshold_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = args.output_dir / "thresholds_oof.csv"
    threshold_df.to_csv(threshold_path, index=False)

    print("\nAll operating thresholds have been selected from OOF data.")
    print("Saved frozen OOF thresholds:", threshold_path)
    print("Official Test labels have not been loaded for threshold selection.")

    # Phase 2: only now open the official Test prediction file and evaluate.
    test = load_test_predictions(args.test_csv)
    if test["Patient ID"].isna().any():
        raise ValueError("Missing Patient ID in Test predictions")
    if test["Patient ID"].astype(str).nunique() != 2797:
        raise ValueError(
            f"Expected 2,797 unique Test patients, found {test['Patient ID'].nunique():,}"
        )

    result_rows = []
    for model, prefix in model_score_prefixes.items():
        print(f"\n[{model}] evaluating frozen thresholds on official Test...")
        for disease in DISEASES:
            y_test = test[disease].to_numpy(dtype=np.uint8)
            s_test = test[f"{prefix}_{disease}"].to_numpy(dtype=np.float64)
            for criterion in CRITERIA:
                selected = selected_thresholds[(model, disease, criterion)]
                threshold = selected["Threshold"]
                metrics = confusion_metrics(y_test, s_test, threshold)
                result_rows.append({
                    "Model": model,
                    "Disease": disease,
                    "Criterion": criterion,
                    "Threshold_From_OOF": threshold,
                    "OOF_Target_Sensitivity": selected["OOF_Sensitivity"],
                    "OOF_Target_Specificity": selected["OOF_Specificity"],
                    **metrics,
                })
        print(f"[{model}] Test evaluation completed")

    result_df = pd.DataFrame(result_rows)
    summary_rows = []
    for (model, criterion), group in result_df.groupby(["Model", "Criterion"], sort=False):
        summary_rows.append({
            "Model": model,
            "Method": "Platt",
            "Criterion": criterion,
            "Diseases": int(group["Disease"].nunique()),
            "Macro_Test_Sensitivity": float(group["Test_Sensitivity"].mean()),
            "Macro_Test_Specificity": float(group["Test_Specificity"].mean()),
            "Macro_Test_PPV": float(group["Test_PPV"].mean()),
            "Macro_Test_NPV": float(group["Test_NPV"].mean()),
            "Macro_Test_F1": float(group["Test_F1"].mean()),
            "Mean_OOF_Sensitivity_at_Selection": float(group["OOF_Target_Sensitivity"].mean()),
            "Mean_OOF_Specificity_at_Selection": float(group["OOF_Target_Specificity"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)

    test_path = args.output_dir / "test_operating_points.csv"
    summary_path = args.output_dir / "operating_points_summary.csv"
    metadata_path = args.output_dir / "operating_points_metadata.json"

    result_df.to_csv(test_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "threshold_selection_dataset": "OOF predictions from 5-Fold Patient-wise GroupKFold",
        "threshold_selection_rows": int(len(oof)),
        "threshold_selection_patients": int(oof["Patient ID"].nunique()),
        "test_evaluation_dataset": "Official NIH ChestX-ray14 Test split",
        "test_rows": int(len(test)),
        "test_patients": int(test["Patient ID"].nunique()),
        "probability_method": "Per-disease Platt Scaling",
        "models": list(model_score_prefixes),
        "ensemble_rule": "Arithmetic mean of DenseNet121 and EfficientNet-B0 Platt probabilities",
        "threshold_rule": {
            "YoudenJ": "maximize sensitivity + specificity - 1; ties select the highest threshold",
            "Sensitivity90": "highest threshold with OOF sensitivity >= 0.90",
            "Specificity90": "lowest threshold with OOF specificity >= 0.90",
        },
        "prediction_rule": "positive if probability >= threshold",
        "test_labels_used_for_threshold_selection": False,
        "test_labels_opened_after_threshold_selection": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 76)
    print("OPERATING POINT ANALYSIS COMPLETED")
    print("=" * 76)
    print("Thresholds:", threshold_path)
    print("Test results:", test_path)
    print("Summary:", summary_path)
    print("Metadata:", metadata_path)
    print("\nSummary (macro average across 14 diseases):")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nNOTE: OOF selects thresholds; official Test labels are used only for evaluation.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit Temperature Scaling and per-disease Platt calibrators from OOF logits.

Only OOF predictions from the official Train/Val population are used here.
The official Test list/images/labels are never read by this script.
The output calibrators are evaluated later on the untouched official Test set.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

PROJECT_DIR = Path(__file__).resolve().parents[1]
try:
    from .constants import DISEASES
except ImportError:  # Support running scripts as: python src/<script>.py
    from constants import DISEASES
MODELS = ["densenet121", "efficientnet_b0"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-np.clip(x[positive], -80, 80)))
    exp_x = np.exp(np.clip(x[~positive], -80, 80))
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def binary_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def load_and_validate_oof(model_name: str, oof_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not oof_path.is_file():
        raise FileNotFoundError(f"OOF file not found: {oof_path}")
    df = pd.read_csv(oof_path, dtype={"Image Index": "string", "Patient ID": "string"})
    required = {"Image Index", "Patient ID", "Fold", *DISEASES}
    required |= {f"logit_{d}" for d in DISEASES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{oof_path} is missing columns: {missing}")
    if df["Image Index"].isna().any() or df["Patient ID"].isna().any():
        raise ValueError("OOF file contains missing image or patient IDs")
    if df["Image Index"].duplicated().any():
        raise ValueError("OOF file contains duplicate Image Index values")
    df["Fold"] = pd.to_numeric(df["Fold"], errors="raise").astype(int)
    if sorted(df["Fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("OOF file must contain predictions from Fold 1-5")
    # A patient must have all of their images in one validation fold only.
    patient_fold_counts = df.groupby("Patient ID")["Fold"].nunique()
    if (patient_fold_counts != 1).any():
        bad = patient_fold_counts[patient_fold_counts != 1].index.tolist()[:5]
        raise ValueError(f"Patients appear in multiple OOF folds: {bad}")

    official_list = PROJECT_DIR / "data" / "train_val_list_NIH.txt"
    if not official_list.is_file():
        raise FileNotFoundError(official_list)
    expected = {x.strip() for x in official_list.read_text(encoding="utf-8").splitlines() if x.strip()}
    observed = set(df["Image Index"].astype(str))
    if observed != expected:
        raise ValueError(
            f"OOF coverage mismatch for {model_name}: missing={len(expected-observed)}, "
            f"unexpected={len(observed-expected)}"
        )
    if len(df) != 86524:
        raise ValueError(f"Expected 86,524 OOF rows, found {len(df):,}")

    y = df[DISEASES].to_numpy(dtype=np.float64)
    x = df[[f"logit_{d}" for d in DISEASES]].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("OOF labels must be finite binary values")
    if not np.isfinite(x).all():
        raise ValueError("OOF logits contain NaN or infinite values")
    for j, disease in enumerate(DISEASES):
        if np.unique(y[:, j]).size != 2:
            raise ValueError(f"Cannot calibrate {disease}: OOF labels do not contain both classes")
    return df, x, y


def fit_global_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit one positive temperature by minimizing mean multilabel NLL."""
    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        z = logits / temperature
        # Equal number of samples per disease; matrix mean weights labels equally.
        return binary_nll(z, labels)

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded", options={"xatol": 1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = float(math.exp(result.x))
    return temperature, float(result.fun)


def fit_platt_per_disease(logits: np.ndarray, labels: np.ndarray) -> tuple[dict[str, dict[str, float]], float]:
    """Fit sigmoid(a * raw_logit + b) independently for each disease."""
    params: dict[str, dict[str, float]] = {}
    losses: list[float] = []
    for j, disease in enumerate(DISEASES):
        x = logits[:, j]
        y = labels[:, j]

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            a, b = float(theta[0]), float(theta[1])
            z = a * x + b
            p = sigmoid(z)
            eps = 1e-12
            loss = -(y * np.log(np.clip(p, eps, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, eps, 1.0)))
            # Very small L2 stabilization; the intercept is not penalized.
            value = float(loss.mean() + 1e-8 * a * a)
            residual = p - y
            grad_a = float(np.mean(residual * x) + 2e-8 * a)
            grad_b = float(np.mean(residual))
            return value, np.asarray([grad_a, grad_b], dtype=np.float64)

        result = minimize(
            fun=lambda t: objective(t)[0],
            x0=np.asarray([1.0, 0.0], dtype=np.float64),
            jac=lambda t: objective(t)[1],
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError(f"Platt fit failed for {disease}: {result.message}")
        if not result.success:
            raise RuntimeError(f"Platt optimizer did not converge for {disease}: {result.message}")
        a, b = [float(v) for v in result.x]
        if not np.isfinite([a, b]).all():
            raise RuntimeError(f"Non-finite Platt parameters for {disease}")
        params[disease] = {"slope": a, "intercept": b, "fit_nll": float(result.fun)}
        losses.append(float(result.fun))
    return params, float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit OOF-based probability calibrators (no Test access).")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--oof-dir", default="reports/oof")
    parser.add_argument("--output-dir", default="reports/calibration")
    args = parser.parse_args()

    oof_dir = Path(args.oof_dir)
    if not oof_dir.is_absolute():
        oof_dir = PROJECT_DIR / oof_dir
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = PROJECT_DIR / output_root

    for model_name in args.models:
        oof_path = oof_dir / model_name / f"{model_name}_oof.csv.gz"
        if not oof_path.is_file():
            # Backward compatibility for locally regenerated, uncompressed OOF files.
            oof_path = oof_dir / model_name / f"{model_name}_oof.csv"
        df, logits, labels = load_and_validate_oof(model_name, oof_path)
        print("\n" + "=" * 76)
        print(f"Fitting calibration parameters: {model_name}")
        print(f"OOF rows: {len(df):,} | patients: {df['Patient ID'].nunique():,}")
        print(f"Fold counts: {df['Fold'].value_counts().sort_index().to_dict()}")
        print("Official Test data: NOT READ")

        baseline_nll = binary_nll(logits, labels)
        temperature, temperature_fit_nll = fit_global_temperature(logits, labels)
        platt, platt_fit_nll = fit_platt_per_disease(logits, labels)

        result = {
            "model": model_name,
            "fit_method": "OOF calibration fit; final evaluation must use untouched official Test",
            "oof_file": str(oof_path.relative_to(PROJECT_DIR)) if oof_path.is_relative_to(PROJECT_DIR) else str(oof_path),
            "n_oof_images": int(len(df)),
            "n_oof_patients": int(df["Patient ID"].nunique()),
            "fold_counts": {str(k): int(v) for k, v in df["Fold"].value_counts().sort_index().items()},
            "diseases": DISEASES,
            "temperature_scaling": {
                "temperature": temperature,
                "fit_nll_on_oof_not_unbiased_evaluation": temperature_fit_nll,
            },
            "per_disease_platt": platt,
            "fit_nll_on_oof_not_unbiased_evaluation": {
                "uncalibrated": baseline_nll,
                "temperature_scaling": temperature_fit_nll,
                "per_disease_platt_macro": platt_fit_nll,
            },
            "warning": (
                "OOF fit NLL is an optimization/training diagnostic, not an unbiased calibration score. "
                "Do not select or report calibration performance from this fit data; evaluate once on the untouched Test set."
            ),
        }

        model_out = output_root / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        json_path = model_out / "calibration_parameters.json"
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        platt_rows = [
            {"Disease": d, "Slope": v["slope"], "Intercept": v["intercept"], "OOF_Fit_NLL": v["fit_nll"]}
            for d, v in platt.items()
        ]
        pd.DataFrame(platt_rows).to_csv(model_out / "platt_parameters.csv", index=False)
        pd.DataFrame([{
            "Model": model_name,
            "N_OOF_Images": len(df),
            "N_OOF_Patients": df["Patient ID"].nunique(),
            "Temperature": temperature,
            "OOF_Fit_NLL_Uncalibrated": baseline_nll,
            "OOF_Fit_NLL_Temperature": temperature_fit_nll,
            "OOF_Fit_NLL_Platt_Macro": platt_fit_nll,
        }]).to_csv(model_out / "calibration_fit_summary.csv", index=False)

        print(f"Temperature: {temperature:.6f}")
        print(f"OOF fit NLL (not unbiased evaluation): baseline={baseline_nll:.6f}, TS={temperature_fit_nll:.6f}, Platt={platt_fit_nll:.6f}")
        print(f"Saved: {json_path}")
        print("Calibration parameters fitted successfully.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-time final evaluation on the untouched official NIH Test split.

Fits no parameters. Loads final full-Train/Val models and OOF-fitted calibration
parameters, evaluates raw/temperature/Platt predictions, optional horizontal-flip
TTA, and soft-voting ensembles. Test labels are read only for final reporting.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import CXR14Dataset, DISEASES  # noqa: E402
from models import build_model  # noqa: E402

MODELS = ["densenet121", "efficientnet_b0"]
METADATA_CSV = PROJECT_DIR / "data" / "Data_Entry_2017.csv"
TEST_LIST = PROJECT_DIR / "data" / "test_list_NIH.txt"
TRAINVAL_MANIFEST = PROJECT_DIR / "splits" / "final_trainval.csv"
TEST_MANIFEST = PROJECT_DIR / "splits" / "final_test.csv"
IMAGE_DIR = PROJECT_DIR / "data" / "images-224"
FINAL_ROOT = PROJECT_DIR / "checkpoints" / "final"
CALIB_ROOT = PROJECT_DIR / "reports" / "calibration"
OUTPUT_ROOT = PROJECT_DIR / "reports" / "test_evaluation"


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-np.clip(x[pos], -80, 80)))
    e = np.exp(np.clip(x[~pos], -80, 80))
    out[~pos] = e / (1.0 + e)
    return out


def build_test_manifest() -> pd.DataFrame:
    if not METADATA_CSV.is_file() or not TEST_LIST.is_file():
        raise FileNotFoundError("Official NIH metadata or Test list is missing")
    test_names = [x.strip() for x in TEST_LIST.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(test_names) != 25596 or len(test_names) != len(set(test_names)):
        raise ValueError(f"Expected 25,596 unique official Test images; got {len(test_names):,}")

    metadata = pd.read_csv(
        METADATA_CSV,
        dtype={"Image Index": "string", "Patient ID": "string"},
    )
    if metadata["Image Index"].duplicated().any():
        raise ValueError("Data_Entry_2017.csv contains duplicate Image Index rows")
    indexed = metadata.set_index("Image Index", drop=False)
    missing = set(test_names) - set(indexed.index.astype(str))
    if missing:
        raise ValueError(f"Official Test images missing from metadata: {list(missing)[:5]}")
    test = indexed.loc[test_names].reset_index(drop=True).copy()

    findings = test["Finding Labels"].fillna("").astype(str)
    finding_sets = findings.map(lambda s: {v.strip() for v in s.split("|") if v.strip()})
    for disease in DISEASES:
        test[disease] = finding_sets.map(lambda labels: int(disease in labels)).astype(np.uint8)

    if test["Patient ID"].isna().any():
        raise ValueError("Official Test set contains missing Patient IDs")
    if len(test) != 25596 or test["Image Index"].nunique() != 25596:
        raise ValueError("Test manifest image count/uniqueness check failed")
    if not TRAINVAL_MANIFEST.is_file():
        raise FileNotFoundError(f"Train/Val manifest not found: {TRAINVAL_MANIFEST}")
    trainval = pd.read_csv(TRAINVAL_MANIFEST, usecols=["Patient ID"], dtype={"Patient ID": "string"})
    overlap = set(trainval["Patient ID"].astype(str)) & set(test["Patient ID"].astype(str))
    if overlap:
        raise ValueError(f"Train/Val and Test patient overlap: {list(overlap)[:10]}")

    # Confirm image files exist before allowing any inference.
    missing_images = [name for name in test_names if not (IMAGE_DIR / name).is_file()]
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} official Test image files are missing; examples={missing_images[:5]}")

    # Persist only the fields required by test inference and downstream analysis.
    # Demographic and acquisition metadata remain in the original NIH source CSV.
    test_manifest = test[["Image Index", "Patient ID", *DISEASES]].copy()
    TEST_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    test_manifest.to_csv(TEST_MANIFEST, index=False)
    print(
        f"Official Test manifest: PASS | images={len(test_manifest):,} | "
        f"patients={test_manifest['Patient ID'].nunique():,} | Train/Val patient overlap=0"
    )
    return test_manifest


def load_calibration(model_name: str) -> dict[str, Any]:
    path = CALIB_ROOT / model_name / "calibration_parameters.json"
    if not path.is_file():
        raise FileNotFoundError(f"Calibration parameter file missing: {path}. Run fit_calibrators.py first.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("model") != model_name:
        raise ValueError(f"Calibration model mismatch in {path}")
    if result.get("diseases") != DISEASES:
        raise ValueError(f"Disease order mismatch in {path}")
    return result


def infer_model(model_name: str, device: torch.device, batch_size: int, num_workers: int, use_amp: bool, use_tta: bool) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    checkpoint_path = FINAL_ROOT / model_name / "gamma_0" / "final_off" / "final_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != model_name:
        raise ValueError(f"Final checkpoint model mismatch: {checkpoint_path}")
    if checkpoint.get("use_clahe", False):
        raise ValueError(
            f"{model_name} final checkpoint indicates CLAHE=True, but this evaluator expects selected CLAHE OFF pipeline"
        )
    if checkpoint.get("num_classes") != 14 or checkpoint.get("diseases") != DISEASES:
        raise ValueError(f"Final checkpoint class schema mismatch: {checkpoint_path}")

    model = build_model(model_name, num_classes=len(DISEASES), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    ds = CXR14Dataset(csv_path=TEST_MANIFEST, image_dir=IMAGE_DIR, use_clahe=False)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    orig_parts: list[np.ndarray] = []
    flip_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda" and use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    orig_logits = model(images)
                    flip_logits = model(torch.flip(images, dims=[3])) if use_tta else None
            else:
                orig_logits = model(images)
                flip_logits = model(torch.flip(images, dims=[3])) if use_tta else None
            orig_parts.append(orig_logits.float().cpu().numpy())
            if flip_logits is not None:
                flip_parts.append(flip_logits.float().cpu().numpy())

    orig = np.concatenate(orig_parts, axis=0)
    flipped = np.concatenate(flip_parts, axis=0) if use_tta else None
    checkpoint_meta = {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
        "trainval_images": checkpoint.get("trainval_images"),
        "final_epochs": checkpoint.get("final_epochs"),
    }
    if orig.shape != (len(ds), len(DISEASES)):
        raise RuntimeError(f"Unexpected prediction shape: {orig.shape}")
    if not np.isfinite(orig).all() or (flipped is not None and not np.isfinite(flipped).all()):
        raise ValueError("Non-finite Test logits")
    del model, checkpoint, ds, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return orig, flipped, checkpoint_meta


def binary_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 0.0, 1.0)
    bin_ids = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    for k in range(n_bins):
        mask = bin_ids == k
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def metric_rows(y_true: np.ndarray, probability: np.ndarray, model_label: str, method: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    class_rows: list[dict[str, Any]] = []
    aucs: list[float] = []
    aps: list[float] = []
    eces: list[float] = []
    for j, disease in enumerate(DISEASES):
        y = y_true[:, j].astype(int)
        p = np.clip(probability[:, j], 0.0, 1.0)
        auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ap = float(average_precision_score(y, p))
        ece = binary_ece(y, p, n_bins=15)
        if np.isfinite(auc): aucs.append(auc)
        aps.append(ap)
        eces.append(ece)
        class_rows.append({
            "Model": model_label, "Method": method, "Disease": disease,
            "Positive_Count": int(y.sum()), "Negative_Count": int(len(y) - y.sum()),
            "AUROC": auc, "AUPRC": ap, "ECE": ece,
        })
    summary = {
        "Model": model_label,
        "Method": method,
        "Mean_AUROC": float(np.mean(aucs)) if aucs else float("nan"),
        "Mean_AUPRC": float(np.mean(aps)) if aps else float("nan"),
        "Mean_Per_Disease_ECE": float(np.mean(eces)) if eces else float("nan"),
        "Global_ECE": binary_ece(y_true.reshape(-1), probability.reshape(-1), n_bins=15),
        "Test_Images": int(y_true.shape[0]),
        "ECE_Bins": 15,
    }
    return summary, class_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Final once-only evaluation on official NIH Test set.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tta", action="store_true", help="Skip horizontal-flip TTA")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch size or number of workers")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {device}, but only {torch.cuda.device_count()} GPUs are available")
        torch.cuda.set_device(0 if device.index is None else device.index)
    use_amp = device.type == "cuda" and not args.no_amp
    use_tta = not args.no_tta

    print("=" * 76)
    print("FINAL OFFICIAL TEST EVALUATION")
    print("=" * 76)
    print(f"Device: {device} | AMP: {use_amp} | Horizontal Flip TTA: {use_tta}")
    print("This script reads Test labels only for final metrics. It fits no parameters.")

    test_df = build_test_manifest()
    y_true = test_df[DISEASES].to_numpy(dtype=np.int64)
    all_probs: dict[str, dict[str, np.ndarray]] = {}
    all_logits: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    summaries: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    metadata_rows: dict[str, Any] = {}

    for model_name in args.models:
        calibration = load_calibration(model_name)
        orig_logits, flip_logits, checkpoint_meta = infer_model(
            model_name, device, args.batch_size, args.num_workers, use_amp, use_tta
        )
        all_logits[model_name] = (orig_logits, flip_logits)
        temperature = float(calibration["temperature_scaling"]["temperature"])
        platt = calibration["per_disease_platt"]
        slopes = np.asarray([platt[d]["slope"] for d in DISEASES], dtype=np.float64)
        intercepts = np.asarray([platt[d]["intercept"] for d in DISEASES], dtype=np.float64)

        probs = {
            "raw": sigmoid(orig_logits),
            "temperature": sigmoid(orig_logits / temperature),
            "platt": sigmoid(orig_logits * slopes[None, :] + intercepts[None, :]),
        }
        if flip_logits is not None:
            probs["raw_tta"] = 0.5 * (probs["raw"] + sigmoid(flip_logits))
            probs["temperature_tta"] = 0.5 * (probs["temperature"] + sigmoid(flip_logits / temperature))
            probs["platt_tta"] = 0.5 * (
                probs["platt"] + sigmoid(flip_logits * slopes[None, :] + intercepts[None, :])
            )
        all_probs[model_name] = probs
        metadata_rows[model_name] = {
            **checkpoint_meta,
            "temperature": temperature,
            "platt_slopes": slopes.tolist(),
            "platt_intercepts": intercepts.tolist(),
        }

        for method, probability in probs.items():
            summary, per_class = metric_rows(y_true, probability, model_name, method)
            summaries.append(summary)
            class_rows.extend(per_class)
        print(f"[{model_name}] Test inference and metrics complete.")

    # Soft-voting ensembles: average post-sigmoid probabilities from both final models.
    if set(args.models) == set(MODELS):
        common_methods = sorted(set(all_probs[MODELS[0]]) & set(all_probs[MODELS[1]]))
        for method in common_methods:
            ensemble_probability = 0.5 * (all_probs[MODELS[0]][method] + all_probs[MODELS[1]][method])
            summary, per_class = metric_rows(y_true, ensemble_probability, "soft_voting_ensemble", method)
            summaries.append(summary)
            class_rows.extend(per_class)
            all_probs.setdefault("soft_voting_ensemble", {})[method] = ensemble_probability

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "test_summary.csv"
    per_class_path = OUTPUT_ROOT / "test_per_disease_metrics.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(class_rows).to_csv(per_class_path, index=False)

    # Store predictions for downstream Operating Point, calibration plots, and error analysis.
    predictions = test_df[["Image Index", "Patient ID", *DISEASES]].copy()
    for model_name, probs in all_probs.items():
        for method, values in probs.items():
            prefix = model_name
            for j, disease in enumerate(DISEASES):
                predictions[f"{prefix}_{method}_{disease}"] = values[:, j]
    prediction_path = OUTPUT_ROOT / "test_predictions.csv"
    predictions.to_csv(prediction_path, index=False, float_format="%.9g")
    (OUTPUT_ROOT / "evaluation_metadata.json").write_text(
        json.dumps({
            "test_images": int(len(test_df)),
            "test_patients": int(test_df["Patient ID"].nunique()),
            "models": metadata_rows,
            "tta_horizontal_flip": use_tta,
            "calibration_fit_source": "OOF predictions from official Train/Val only",
            "test_used_for_calibration_fit": False,
            "ece_bins": 15,
            "metrics": ["macro AUROC", "macro AUPRC (average precision)", "mean per-disease ECE", "global ECE"],
            "warning": "Do not adjust models, calibrators, thresholds, or preprocessing based on these Test results and then report the same Test metrics as unbiased.",
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 76)
    print("OFFICIAL TEST EVALUATION COMPLETED")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Summary: {summary_path}")
    print(f"Per-disease metrics: {per_class_path}")
    print(f"Predictions: {prediction_path}")
    print("Official Test was not used to fit calibration parameters.")


if __name__ == "__main__":
    main()

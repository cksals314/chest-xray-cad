#!/usr/bin/env python3
"""Extract out-of-fold logits for NIH ChestX-ray14 using existing CV checkpoints.

This script does NOT train a model and NEVER reads the official Test split.
For each fold, it loads that fold's best checkpoint and predicts only its
patient-disjoint validation CSV. The combined OOF file should contain every
official Train/Val image and patient exactly once.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from dataset import CXR14Dataset, DISEASES  # noqa: E402
from models import build_model  # noqa: E402


DEFAULT_FOLD_METRICS = {
    "densenet121": "reports/cross_validation/densenet121_fold_metrics.csv",
    "efficientnet_b0": "reports/cross_validation/efficientnet_b0_fold_metrics.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 5-fold out-of-fold logits from existing NIH CXR14 checkpoints."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["densenet121", "efficientnet_b0"],
    )
    parser.add_argument("--device", default="cuda:0", help="e.g. cuda:0 or cuda:1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fold-metrics", default=None)
    parser.add_argument("--image-dir", default="data/images-224")
    parser.add_argument("--output-dir", default="reports/oof")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Support raw state_dicts and common wrapped checkpoint formats."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint).__name__}")

    candidate = None
    for key in (
        "model_state_dict",
        "state_dict",
        "model_state",
        "model",
        "weights",
        "net",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(
            isinstance(v, torch.Tensor) for v in value.values()
        ):
            candidate = value
            break

    if candidate is None and checkpoint and all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        candidate = checkpoint

    if candidate is None:
        keys = list(checkpoint.keys())[:30]
        raise KeyError(
            "Could not locate model weights in checkpoint. "
            f"Top-level keys: {keys}"
        )

    # Remove a DDP prefix if one is present on every parameter name.
    if candidate and all(k.startswith("module.") for k in candidate):
        candidate = {k[len("module."):]: v for k, v in candidate.items()}

    return candidate


def load_fold_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Fold metrics CSV not found: {path}")
    frame = pd.read_csv(path)
    required = {"Fold", "Checkpoint"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["Fold"] = pd.to_numeric(frame["Fold"], errors="raise").astype(int)
    frame = frame[frame["Fold"].isin([1, 2, 3, 4, 5])].copy()
    if frame["Fold"].duplicated().any():
        raise ValueError(f"Duplicate Fold rows found in {path}")
    if sorted(frame["Fold"].tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError(
            "Fold metrics must contain exactly one row for each Fold 1-5. "
            f"Found: {sorted(frame['Fold'].tolist())}"
        )
    return frame.sort_values("Fold")


def predict_fold(
    model: torch.nn.Module,
    dataset: CXR14Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch_images, _ in loader:
            batch_images = batch_images.to(
                device, non_blocking=(device.type == "cuda")
            )
            if device.type == "cuda" and use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch_images)
            else:
                logits = model(batch_images)
            outputs.append(logits.float().cpu().numpy())

    if not outputs:
        raise RuntimeError("DataLoader returned no batches.")
    predictions = np.concatenate(outputs, axis=0)
    if predictions.shape != (len(dataset), len(DISEASES)):
        raise RuntimeError(
            f"Unexpected prediction shape {predictions.shape}; expected "
            f"({len(dataset)}, {len(DISEASES)})"
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite values found in OOF logits.")
    return predictions


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {args.device}, but only {torch.cuda.device_count()} CUDA device(s) exist."
            )
        torch.cuda.set_device(index)

    metrics_value = args.fold_metrics or DEFAULT_FOLD_METRICS[args.model]
    metrics_path = resolve_path(metrics_value)
    fold_metrics = load_fold_metrics(metrics_path)

    image_dir = resolve_path(args.image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir = resolve_path(args.output_dir)
    model_output_dir = output_dir / args.model
    model_output_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = device.type == "cuda" and not args.no_amp
    all_frames: list[pd.DataFrame] = []
    image_occurrences: Counter[str] = Counter()
    patient_occurrences: Counter[str] = Counter()

    print("=" * 76)
    print("NIH ChestX-ray14: Out-of-Fold Prediction Extraction")
    print("=" * 76)
    print(f"Model: {args.model}")
    print(f"Device: {device} | AMP: {amp_enabled}")
    print(f"Image directory: {image_dir}")
    print(f"Fold metrics: {metrics_path}")
    print("Official Test split: NOT USED")

    for row in fold_metrics.itertuples(index=False):
        fold = int(row.Fold)
        checkpoint_path = resolve_path(row.Checkpoint)
        val_csv = PROJECT_DIR / "splits" / f"fold_{fold}" / "val.csv"
        fold_output = model_output_dir / f"fold_{fold}_oof.csv.gz"

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Fold {fold} checkpoint not found: {checkpoint_path}"
            )
        if not val_csv.is_file():
            raise FileNotFoundError(f"Fold {fold} validation CSV not found: {val_csv}")

        print(f"\n[Fold {fold}] Loading validation data and checkpoint")
        dataset = CXR14Dataset(
            csv_path=val_csv,
            image_dir=image_dir,
            use_clahe=False,
        )

        model = build_model(
            args.model,
            num_classes=len(DISEASES),
            pretrained=False,
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        logits = predict_fold(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=amp_enabled,
        )
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))

        metadata = dataset.df[["Image Index", "Patient ID", *DISEASES]].copy()
        metadata["Fold"] = fold
        for index, disease in enumerate(DISEASES):
            metadata[f"logit_{disease}"] = logits[:, index]
            metadata[f"prob_{disease}"] = probabilities[:, index]

        metadata.to_csv(fold_output, index=False, float_format="%.9g", compression="gzip")
        all_frames.append(metadata)

        names = metadata["Image Index"].astype(str).tolist()
        patients = metadata["Patient ID"].astype(str).unique().tolist()
        image_occurrences.update(names)
        patient_occurrences.update(patients)

        print(
            f"[Fold {fold}] PASS | images={len(metadata):,} | "
            f"patients={len(patients):,} | saved={fold_output}"
        )

        del model, checkpoint, state_dict, dataset, logits, probabilities, metadata
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values("Image Index").reset_index(drop=True)

    # Verify exact coverage against the official Train/Val image list.
    official_list = resolve_path("data/train_val_list_NIH.txt")
    if not official_list.is_file():
        raise FileNotFoundError(f"Official Train/Val list not found: {official_list}")
    expected_images = {
        line.strip() for line in official_list.read_text().splitlines() if line.strip()
    }
    observed_images = set(combined["Image Index"].astype(str))

    missing = expected_images - observed_images
    unexpected = observed_images - expected_images
    duplicate_images = [name for name, count in image_occurrences.items() if count != 1]
    duplicate_patients = [pid for pid, count in patient_occurrences.items() if count != 1]

    if missing or unexpected or duplicate_images or duplicate_patients:
        raise AssertionError(
            "OOF coverage validation failed: "
            f"missing_images={len(missing)}, unexpected_images={len(unexpected)}, "
            f"duplicate_images={len(duplicate_images)}, "
            f"patients_in_multiple_val_folds={len(duplicate_patients)}"
        )

    expected_patients = combined["Patient ID"].astype(str).nunique()
    if len(combined) != len(expected_images):
        raise AssertionError(
            f"OOF row count {len(combined)} != official Train/Val count {len(expected_images)}"
        )

    final_output = model_output_dir / f"{args.model}_oof.csv.gz"
    combined.to_csv(final_output, index=False, float_format="%.9g", compression="gzip")

    print("\n" + "=" * 76)
    print("OOF EXTRACTION VALIDATION")
    print("=" * 76)
    print(f"OOF rows: {len(combined):,}")
    print(f"Unique images: {len(observed_images):,}")
    print(f"Unique patients: {expected_patients:,}")
    print("Each official Train/Val image appears exactly once: PASS")
    print("Each patient appears in exactly one validation fold: PASS")
    print(f"Official Test images used: 0")
    print(f"Combined OOF file: {final_output}")
    print("OOF extraction completed successfully.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train one final CXR14 model on the complete official Train/Val set.

The script deliberately has no validation loop and no early stopping. The
number of epochs is derived from the selected model's 5-fold Best_Epoch values:
    final_epochs = round(mean(Best_Epoch) * 0.8)

By default, the script uses the already selected settings:
- model: chosen on the command line (densenet121 or efficientnet_b0)
- gamma: 0 (weighted BCE; implemented through FocalLossWithLogits)
- CLAHE: OFF (use --clahe only when deliberately training a CLAHE model)
- all 86,524 official Train/Val images from splits/fold_1/{train,val}.csv
- no augmentation

Example:
    python src/train_final.py --model densenet121 --device cuda:0
    python src/train_final.py --model efficientnet_b0 --device cuda:1

Resume after interruption:
    python src/train_final.py --model densenet121 --device cuda:0 \\
        --resume checkpoints/final/densenet121/gamma_0/final_off/last_checkpoint.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import CXR14Dataset, DISEASES  # noqa: E402
from losses import FocalLossWithLogits  # noqa: E402
from models import build_model, get_parameter_groups  # noqa: E402

DEFAULT_CONFIG = PROJECT_DIR / "configs" / "train_config.json"
SPLIT_DIR = PROJECT_DIR / "splits"
IMAGE_DIR = PROJECT_DIR / "data" / "images-224"
CLAHE_IMAGE_DIR = PROJECT_DIR / "data" / "images-224-clahe"
TRAIN_VAL_LIST = PROJECT_DIR / "data" / "train_val_list_NIH.txt"
TEST_LIST = PROJECT_DIR / "data" / "test_list_NIH.txt"
METADATA_CSV = PROJECT_DIR / "data" / "Data_Entry_2017.csv"

CHECKPOINT_ROOT = PROJECT_DIR / "checkpoints" / "final"
LOG_ROOT = PROJECT_DIR / "logs" / "training_final"
REPORT_ROOT = PROJECT_DIR / "reports" / "training_final"

# Fold metric files produced by the completed experiments in this project.
CV_REPORT_ROOT = PROJECT_DIR / "reports" / "cross_validation"
DEFAULT_CV_METRICS = {
    "densenet121": CV_REPORT_ROOT / "densenet121_fold_metrics.csv",
    "efficientnet_b0": CV_REPORT_ROOT / "efficientnet_b0_fold_metrics.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a final CXR14 model on the complete official Train/Val set."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model", required=True, choices=["densenet121", "efficientnet_b0"]
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="e.g. cuda:0, cuda:1, or cpu (default: config value)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override automatic epoch count derived from the 5-fold Best_Epoch values.",
    )
    parser.add_argument(
        "--cv-fold-metrics", type=Path, default=None,
        help="Optional path to this model's completed 5-fold fold_metrics.csv.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--classifier-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", type=str, default="final_off")
    parser.add_argument(
        "--clahe", action="store_true",
        help="Use precomputed data/images-224-clahe (not the default selected setting).",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume from a last_checkpoint.pth created by this script.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow starting over in an output directory that already has a checkpoint.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate split, Test patient separation, and epoch selection without training.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config)
    for key in (
        "batch_size", "num_workers", "backbone_lr", "classifier_lr",
        "weight_decay", "seed", "device",
    ):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value

    settings["use_clahe"] = bool(args.clahe)
    settings["amp"] = bool(settings.get("amp", True)) and not args.no_amp

    required = [
        "batch_size", "num_workers", "backbone_lr", "classifier_lr",
        "weight_decay", "seed", "device", "pin_memory", "pretrained", "amp",
    ]
    missing = [key for key in required if key not in settings]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")
    if int(settings["batch_size"]) < 1:
        raise ValueError("batch_size must be >= 1")
    if int(settings["num_workers"]) < 0:
        raise ValueError("num_workers must be >= 0")
    if float(settings["backbone_lr"]) <= 0 or float(settings["classifier_lr"]) <= 0:
        raise ValueError("Learning rates must be positive")
    if float(settings["weight_decay"]) < 0:
        raise ValueError("weight_decay must be >= 0")
    return settings


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but CUDA is not available")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} GPU(s) are visible"
            )
    return device


def read_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required image list is missing: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate image names in {path}")
    return names


def load_complete_trainval() -> pd.DataFrame:
    """Reconstruct the full official Train/Val table from one complete fold partition."""
    train_path = SPLIT_DIR / "fold_1" / "train.csv"
    val_path = SPLIT_DIR / "fold_1" / "val.csv"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            "Expected GroupKFold files at splits/fold_1/train.csv and val.csv"
        )

    train_df = pd.read_csv(train_path, dtype={"Image Index": "string", "Patient ID": "string"})
    val_df = pd.read_csv(val_path, dtype={"Image Index": "string", "Patient ID": "string"})
    required = {"Image Index", "Patient ID", *DISEASES}
    for name, frame in (("train", train_df), ("validation", val_df)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Fold 1 {name}.csv is missing columns: {sorted(missing)}")
        if frame["Image Index"].isna().any() or frame["Patient ID"].isna().any():
            raise ValueError(f"Fold 1 {name}.csv contains missing image or patient IDs")
        labels = frame[DISEASES].to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Fold 1 {name}.csv has non-binary disease labels")

    train_patients = set(train_df["Patient ID"].astype(str))
    val_patients = set(val_df["Patient ID"].astype(str))
    overlap = train_patients & val_patients
    if overlap:
        raise RuntimeError(f"Patient leakage in Fold 1: {sorted(overlap)[:5]}")

    trainval = pd.concat([train_df, val_df], ignore_index=True)
    if trainval["Image Index"].duplicated().any():
        dupes = trainval.loc[trainval["Image Index"].duplicated(), "Image Index"].head().tolist()
        raise RuntimeError(f"Duplicate images in Fold 1 union: {dupes}")

    official_trainval = set(read_names(TRAIN_VAL_LIST))
    split_images = set(trainval["Image Index"].astype(str))
    if split_images != official_trainval:
        missing = sorted(official_trainval - split_images)
        extra = sorted(split_images - official_trainval)
        raise RuntimeError(
            "Fold 1 train+val does not exactly match the official Train/Val list. "
            f"Missing={missing[:5]}, extra={extra[:5]}"
        )

    # Ensure each of the five GroupKFold partitions covers exactly the same Train/Val images.
    for fold in range(1, 6):
        f_train = SPLIT_DIR / f"fold_{fold}" / "train.csv"
        f_val = SPLIT_DIR / f"fold_{fold}" / "val.csv"
        if not f_train.is_file() or not f_val.is_file():
            raise FileNotFoundError(f"Missing GroupKFold CSV(s) for Fold {fold}")
        f_train_df = pd.read_csv(f_train, usecols=["Image Index", "Patient ID"])
        f_val_df = pd.read_csv(f_val, usecols=["Image Index", "Patient ID"])
        p_train = set(f_train_df["Patient ID"].astype(str))
        p_val = set(f_val_df["Patient ID"].astype(str))
        if p_train & p_val:
            raise RuntimeError(f"Patient leakage found in Fold {fold}")
        f_images = set(f_train_df["Image Index"].astype(str)) | set(f_val_df["Image Index"].astype(str))
        if f_images != official_trainval:
            raise RuntimeError(f"Fold {fold} train+val does not cover the official Train/Val images")

    trainval = trainval.sort_values("Image Index").reset_index(drop=True)
    # CXR14Dataset reads a CSV path (not an in-memory DataFrame), so persist the
    # verified full Train+Val union once as a dedicated final-training manifest.
    full_trainval_csv = SPLIT_DIR / "final_trainval.csv"
    trainval.to_csv(full_trainval_csv, index=False)
    print(f"Verified full Train/Val manifest: {full_trainval_csv}")
    return trainval, full_trainval_csv


def verify_test_patient_separation(trainval: pd.DataFrame) -> None:
    test_names = set(read_names(TEST_LIST))
    metadata = pd.read_csv(
        METADATA_CSV,
        usecols=["Image Index", "Patient ID"],
        dtype={"Image Index": "string", "Patient ID": "string"},
    )
    known_images = set(metadata["Image Index"].dropna().astype(str))
    missing = test_names - known_images
    if missing:
        raise RuntimeError(f"{len(missing)} official Test images missing from metadata: {sorted(missing)[:5]}")

    test_rows = metadata[metadata["Image Index"].isin(test_names)]
    if test_rows["Patient ID"].isna().any():
        raise RuntimeError("Official Test metadata contains missing Patient IDs")

    train_patients = set(trainval["Patient ID"].astype(str))
    test_patients = set(test_rows["Patient ID"].astype(str))
    overlap = train_patients & test_patients
    if overlap:
        raise RuntimeError(
            "Patient overlap between final Train/Val and official Test: "
            f"{sorted(overlap)[:10]}"
        )
    print(
        "Official Test patient separation: PASS "
        f"(Train/Val patients={len(train_patients):,}, Test patients={len(test_patients):,}, overlap=0)"
    )


def calculate_pos_weight(df: pd.DataFrame) -> tuple[torch.Tensor, pd.DataFrame]:
    labels = df[DISEASES].to_numpy(dtype=np.float64)
    positive = labels.sum(axis=0)
    negative = len(labels) - positive
    if (positive <= 0).any():
        absent = [DISEASES[i] for i, value in enumerate(positive) if value <= 0]
        raise ValueError(f"Cannot calculate pos_weight; no positive examples for {absent}")
    weights = negative / positive
    report = pd.DataFrame({
        "Disease": DISEASES,
        "Positive_Count": positive.astype(int),
        "Negative_Count": negative.astype(int),
        "Prevalence": positive / len(labels),
        "pos_weight": weights,
    })
    return torch.tensor(weights, dtype=torch.float32), report


def derive_final_epochs(model_name: str, cv_metrics_path: Path) -> tuple[int, float]:
    if not cv_metrics_path.is_file():
        raise FileNotFoundError(
            f"Completed 5-fold metrics not found: {cv_metrics_path}. "
            "Pass --cv-fold-metrics to specify the correct file, or --epochs to override."
        )
    metrics = pd.read_csv(cv_metrics_path)
    required = {"Fold", "Best_Epoch"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"CV metrics file is missing columns: {sorted(missing)}")

    metrics = metrics.copy()
    metrics["Fold"] = pd.to_numeric(metrics["Fold"], errors="coerce")
    metrics["Best_Epoch"] = pd.to_numeric(metrics["Best_Epoch"], errors="coerce")
    metrics = metrics[metrics["Fold"].isin([1, 2, 3, 4, 5])]
    metrics = metrics.drop_duplicates(subset=["Fold"], keep=False)
    metrics = metrics.sort_values("Fold")
    if metrics["Fold"].astype(int).tolist() != [1, 2, 3, 4, 5]:
        raise ValueError(
            f"{cv_metrics_path} must contain exactly one completed row for each Fold 1-5"
        )
    if metrics["Best_Epoch"].isna().any() or (metrics["Best_Epoch"] < 1).any():
        raise ValueError("CV Best_Epoch values must be positive integers")

    mean_best_epoch = float(metrics["Best_Epoch"].mean())
    raw_epochs = mean_best_epoch * 0.8
    final_epochs = max(1, int(math.floor(raw_epochs + 0.5)))
    return final_epochs, mean_best_epoch


def make_train_loader(csv_path: Path, image_dir: Path, settings: dict[str, Any]) -> DataLoader:
    # Images are already 224x224. CLAHE mode points to the separately precomputed folder;
    # Dataset-side CLAHE is deliberately disabled to avoid applying CLAHE twice.
    dataset = CXR14Dataset(
        csv_path=csv_path,
        image_dir=image_dir,
        use_clahe=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=int(settings["num_workers"]),
        pin_memory=bool(settings["pin_memory"]) and torch.cuda.is_available(),
        persistent_workers=int(settings["num_workers"]) > 0,
        drop_last=False,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: FocalLossWithLogits,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    optimizer_steps = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp else nullcontext()
        )
        with amp_context:
            logits = model(images)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at batch {batch_idx}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(images.size(0))
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        optimizer_steps += 1

    if total_examples == 0:
        raise RuntimeError("No training batches were processed")
    return total_loss / total_examples, optimizer_steps


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def main() -> None:
    args = parse_args()
    if not args.run_tag or any(char in args.run_tag for char in "/\\"):
        raise ValueError("--run-tag must be a non-empty folder name without path separators")

    config = load_config(args.config)
    settings = resolve_settings(args, config)
    device = select_device(str(settings["device"]))
    set_seed(int(settings["seed"]))

    # Select the correct CV metrics file for automatic epoch calculation.
    cv_metrics_path = args.cv_fold_metrics or DEFAULT_CV_METRICS[args.model]
    final_epochs, mean_best_epoch = derive_final_epochs(args.model, cv_metrics_path)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be >= 1")
        final_epochs = int(args.epochs)

    trainval, trainval_csv = load_complete_trainval()
    verify_test_patient_separation(trainval)

    image_dir = CLAHE_IMAGE_DIR if settings["use_clahe"] else IMAGE_DIR
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    pos_weight, pos_weight_report = calculate_pos_weight(trainval)
    device_pos_weight = pos_weight.to(device)

    gamma = 0.0  # Selected configuration from the completed experiments.
    use_amp = bool(settings["amp"]) and device.type == "cuda"

    if args.dry_run:
        print("=" * 76)
        print("CXR14 FINAL TRAINING DRY RUN: PASS")
        print("=" * 76)
        print(f"Model: {args.model} | Gamma: {gamma:g} | CLAHE: {settings['use_clahe']}")
        print(f"Train/Val: {len(trainval):,} images / {trainval['Patient ID'].nunique():,} patients")
        print(f"CV metrics: {cv_metrics_path}")
        print(f"Mean Best_Epoch: {mean_best_epoch:.3f}")
        print(f"Final epochs: {final_epochs}")
        print(f"Device: {device} | AMP: {use_amp} | Batch size: {settings['batch_size']}")
        print(f"Image directory: {image_dir}")
        print("Official Test data was used only for patient-separation verification.")
        print("No model was instantiated and no training was started.")
        return

    run_root = Path(args.run_tag)
    checkpoint_dir = CHECKPOINT_ROOT / args.model / "gamma_0" / run_root
    log_dir = LOG_ROOT / args.model / run_root
    report_dir = REPORT_ROOT / args.model / run_root
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    final_checkpoint_path = checkpoint_dir / "final_model.pth"
    last_checkpoint_path = checkpoint_dir / "last_checkpoint.pth"
    history_path = log_dir / "history.csv"
    summary_path = report_dir / "final_training_summary.json"
    weight_path = report_dir / "pos_weight_trainval.csv"

    if final_checkpoint_path.exists() and not args.overwrite and args.resume is None:
        raise FileExistsError(
            f"Final checkpoint already exists: {final_checkpoint_path}. "
            "Use a new --run-tag, --overwrite, or --resume."
        )

    pos_weight_report.to_csv(weight_path, index=False)
    loader = make_train_loader(trainval_csv, image_dir, settings)

    model = build_model(
        args.model,
        num_classes=len(DISEASES),
        pretrained=bool(settings["pretrained"]),
    ).to(device)
    parameter_groups = get_parameter_groups(
        model,
        backbone_lr=float(settings["backbone_lr"]),
        classifier_lr=float(settings["classifier_lr"]),
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=final_epochs,
        eta_min=0.0,
    )
    criterion = FocalLossWithLogits(
        gamma=gamma,
        pos_weight=device_pos_weight,
        reduction="mean",
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    history_rows: list[dict[str, Any]] = []
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_state.get("model_name") != args.model:
            raise ValueError("Resume checkpoint model_name does not match --model")
        if bool(resume_state.get("use_clahe", False)) != bool(settings["use_clahe"]):
            raise ValueError("Resume checkpoint CLAHE setting does not match this run")
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if resume_state.get("scaler_state_dict"):
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        start_epoch = int(resume_state["epoch"]) + 1
        if start_epoch > final_epochs:
            raise ValueError(
                f"Checkpoint already completed epoch {start_epoch - 1}, "
                f"which is >= requested final_epochs={final_epochs}"
            )
        if history_path.is_file():
            history_rows = pd.read_csv(history_path).to_dict("records")
        print(f"Resumed from {args.resume} at epoch {start_epoch}/{final_epochs}")

    steps_per_epoch = math.ceil(len(trainval) / int(settings["batch_size"]))
    print("=" * 76)
    print("CXR14 FINAL TRAINING")
    print("=" * 76)
    print(f"Model: {args.model} | Gamma: {gamma:g} (weighted BCE)")
    print(f"Train/Val images: {len(trainval):,} | Patients: {trainval['Patient ID'].nunique():,}")
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Image directory: {image_dir}")
    print(f"CLAHE: {settings['use_clahe']} | Augmentation: False")
    print(f"CV metrics source: {cv_metrics_path}")
    print(f"Mean CV Best_Epoch: {mean_best_epoch:.3f} | x 0.8 => {mean_best_epoch * 0.8:.3f}")
    print(f"Final epochs: {final_epochs} | Batch size: {settings['batch_size']}")
    print(f"Batches/epoch: {steps_per_epoch:,} | Planned optimizer steps: {steps_per_epoch * final_epochs:,}")
    print("Validation / Early Stopping: NOT USED (full Train+Val final fit)")
    print(f"Checkpoint: {final_checkpoint_path}")

    start_time = time.perf_counter()
    total_optimizer_steps = 0

    for epoch in range(start_epoch, final_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, epoch_steps = train_one_epoch(
            model, loader, criterion, optimizer, scaler, device, use_amp
        )
        scheduler.step()
        total_optimizer_steps += epoch_steps
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start

        lr_values = [float(group["lr"]) for group in optimizer.param_groups]
        row = {
            "Epoch": epoch,
            "Final_Epochs": final_epochs,
            "Train_Loss": train_loss,
            "Backbone_LR_After_Step": lr_values[0] if lr_values else math.nan,
            "Classifier_LR_After_Step": lr_values[1] if len(lr_values) > 1 else math.nan,
            "Optimizer_Steps": epoch_steps,
            "Epoch_Seconds": epoch_seconds,
            "CLAHE": bool(settings["use_clahe"]),
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        # Save a resumable state after each fully completed epoch.
        resume_payload = {
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "final_epochs": final_epochs,
            "gamma": gamma,
            "use_clahe": bool(settings["use_clahe"]),
            "diseases": DISEASES,
            "num_classes": len(DISEASES),
            "config": settings,
        }
        atomic_torch_save(resume_payload, last_checkpoint_path)

        print(
            f"Epoch {epoch:02d}/{final_epochs} | train_loss={train_loss:.5f} | "
            f"steps={epoch_steps:,} | time={epoch_seconds / 60:.1f} min | "
            f"LR(backbone/classifier)={lr_values[0]:.2e}/{lr_values[1] if len(lr_values) > 1 else float('nan'):.2e}",
            flush=True,
        )

    total_seconds = time.perf_counter() - start_time
    final_payload = {
        "model_name": args.model,
        "model_state_dict": model.state_dict(),
        "num_classes": len(DISEASES),
        "diseases": DISEASES,
        "gamma": gamma,
        "use_clahe": bool(settings["use_clahe"]),
        "final_epochs": final_epochs,
        "mean_cv_best_epoch": mean_best_epoch,
        "epoch_factor": 0.8,
        "trainval_images": int(len(trainval)),
        "trainval_patients": int(trainval["Patient ID"].nunique()),
        "pos_weight": pos_weight.cpu().tolist(),
        "image_dir": str(image_dir.relative_to(PROJECT_DIR)),
        "training_manifest": str(trainval_csv.relative_to(PROJECT_DIR)),
        "config": settings,
        "training_method": "full official Train+Val; no validation split; no early stopping",
    }
    atomic_torch_save(final_payload, final_checkpoint_path)

    summary = {
        "model": args.model,
        "gamma": gamma,
        "clahe": bool(settings["use_clahe"]),
        "trainval_images": int(len(trainval)),
        "trainval_patients": int(trainval["Patient ID"].nunique()),
        "mean_cv_best_epoch": mean_best_epoch,
        "epoch_factor": 0.8,
        "final_epochs": final_epochs,
        "batch_size": int(settings["batch_size"]),
        "batches_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": steps_per_epoch * final_epochs,
        "completed_optimizer_steps_this_run": total_optimizer_steps,
        "total_seconds_this_run": total_seconds,
        "device": str(device),
        "amp": use_amp,
        "cv_fold_metrics": str(cv_metrics_path.relative_to(PROJECT_DIR)) if cv_metrics_path.is_relative_to(PROJECT_DIR) else str(cv_metrics_path),
        "checkpoint": str(final_checkpoint_path.relative_to(PROJECT_DIR)),
        "history": str(history_path.relative_to(PROJECT_DIR)),
        "pos_weight_report": str(weight_path.relative_to(PROJECT_DIR)),
        "training_manifest": str(trainval_csv.relative_to(PROJECT_DIR)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nFinal training completed.")
    print(f"Final model: {final_checkpoint_path}")
    print(f"Last resumable checkpoint: {last_checkpoint_path}")
    print(f"Training history: {history_path}")
    print(f"pos_weight report: {weight_path}")
    print(f"Summary: {summary_path}")
    print("Note: no Test images were loaded or evaluated by this script.")


if __name__ == "__main__":
    main()

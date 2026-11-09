#!/usr/bin/env python3
"""Train NIH ChestX-ray14 models with patient-wise 5-fold CV.

Example (one fold, smoke test):
    python src/train.py --model densenet121 --gamma 2 --folds 1 \\
        --epochs 1 --max-train-batches 2 --max-val-batches 2

Example (all folds for one gamma):
    python src/train.py --model densenet121 --gamma 0 --folds 1 2 3 4 5

Run gamma=0, 1, and 2 as separate experiments using identical folds/settings.
The script uses one GPU per process (no DDP) to keep the initial pipeline simple.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
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
from metrics import compute_multilabel_metrics  # noqa: E402
from models import build_model, get_parameter_groups  # noqa: E402


DEFAULT_CONFIG = PROJECT_DIR / "configs" / "train_config.json"
SPLIT_DIR = PROJECT_DIR / "splits"
IMAGE_DIR = PROJECT_DIR / "data" / "images-224"
CLAHE_IMAGE_DIR = PROJECT_DIR / "data" / "images-224-clahe"
METADATA_CSV = PROJECT_DIR / "data" / "Data_Entry_2017.csv"
TEST_LIST = PROJECT_DIR / "data" / "test_list_NIH.txt"
CHECKPOINT_ROOT = PROJECT_DIR / "checkpoints" / "training"
LOG_ROOT = PROJECT_DIR / "logs" / "training"
REPORT_ROOT = PROJECT_DIR / "reports" / "training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CXR14 5-fold training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=True, choices=["densenet121", "efficientnet_b0"])
    parser.add_argument("--gamma", required=True, type=float, choices=[0.0, 1.0, 2.0])
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--run-tag", type=str, default="cv",
                        help="Output namespace, e.g. 'smoke' for a test run or 'cv' for final CV")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--classifier-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="Debug only: limit train batches per epoch; 0 means no limit")
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="Debug only: limit validation batches; 0 means no limit")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def resolve_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "backbone_lr": args.backbone_lr,
        "classifier_lr": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "device": args.device,
        "use_clahe": args.clahe,
        "pretrained": args.pretrained,
        "amp": args.amp,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value

    required = [
        "seed", "epochs", "batch_size", "num_workers", "backbone_lr",
        "classifier_lr", "weight_decay", "patience", "min_delta",
        "ece_bins", "amp", "pin_memory", "pretrained", "use_clahe", "device",
    ]
    missing = [key for key in required if key not in settings]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")

    if settings["epochs"] < 1 or settings["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    if settings["num_workers"] < 0 or settings["patience"] < 1:
        raise ValueError("num_workers must be >=0 and patience must be >=1")
    if settings["backbone_lr"] <= 0 or settings["classifier_lr"] <= 0:
        raise ValueError("Learning rates must be positive")
    if settings["weight_decay"] < 0:
        raise ValueError("weight_decay must be >=0")
    if settings["ece_bins"] < 1:
        raise ValueError("ece_bins must be >=1")
    return settings


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_setting)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
    return device


def verify_patient_split(fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Fail fast if fold CSVs do not preserve Patient ID groups."""
    for name, frame in (("train", train_df), ("val", val_df)):
        missing = {"Image Index", "Patient ID", *DISEASES} - set(frame.columns)
        if missing:
            raise ValueError(f"Fold {fold} {name}.csv missing columns: {sorted(missing)}")

    train_patients = set(train_df["Patient ID"].astype(str))
    val_patients = set(val_df["Patient ID"].astype(str))
    overlap = train_patients & val_patients
    if overlap:
        raise RuntimeError(f"Patient leakage in fold {fold}: {list(overlap)[:5]}")

    for name, frame in (("train", train_df), ("val", val_df)):
        labels = frame[DISEASES].to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Fold {fold} {name} labels must be binary 0/1")


def verify_official_test_patient_separation(cv_df: pd.DataFrame) -> None:
    """Ensure the official held-out Test patients do not occur in CV data."""
    if not METADATA_CSV.exists() or not TEST_LIST.exists():
        raise FileNotFoundError("Data_Entry_2017.csv or test_list_NIH.txt is missing")

    test_names = {
        line.strip() for line in TEST_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    metadata = pd.read_csv(
        METADATA_CSV,
        usecols=["Image Index", "Patient ID"],
        dtype={"Image Index": "string", "Patient ID": "string"},
    )
    metadata_names = set(metadata["Image Index"].dropna().astype(str))
    missing_test_names = test_names - metadata_names
    if missing_test_names:
        raise RuntimeError(
            f"{len(missing_test_names)} official Test images are missing from metadata; "
            f"examples: {sorted(missing_test_names)[:5]}"
        )
    test_rows = metadata.loc[metadata["Image Index"].isin(test_names)]
    if test_rows["Patient ID"].isna().any():
        raise RuntimeError("Official Test metadata contains missing Patient ID values")
    test_patients = set(test_rows["Patient ID"].astype(str))
    cv_patients = set(cv_df["Patient ID"].dropna().astype(str))
    overlap = test_patients & cv_patients
    if overlap:
        raise RuntimeError(
            "Official Test and CV Train/Val share Patient IDs. Do not train until this is resolved. "
            f"Examples: {sorted(overlap)[:10]}"
        )
    print(
        f"Official Test patient separation: PASS "
        f"(CV patients={len(cv_patients):,}, Test patients={len(test_patients):,}, overlap=0)"
    )


def calculate_pos_weight(train_df: pd.DataFrame) -> tuple[torch.Tensor, pd.DataFrame]:
    labels = train_df[DISEASES].to_numpy(dtype=np.float32)
    positive = labels.sum(axis=0)
    negative = labels.shape[0] - positive
    if (positive <= 0).any():
        missing = [DISEASES[i] for i, count in enumerate(positive) if count <= 0]
        raise ValueError(f"Cannot calculate pos_weight; no positive samples for: {missing}")

    pos_weight = negative / positive
    report = pd.DataFrame(
        {
            "Disease": DISEASES,
            "Positive_Count": positive.astype(int),
            "Negative_Count": negative.astype(int),
            "Prevalence": positive / labels.shape[0],
            "pos_weight": pos_weight,
        }
    )
    return torch.tensor(pos_weight, dtype=torch.float32), report


def make_loader(
    csv_path: Path,
    settings: dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    # CLAHE is precomputed once into a separate image directory.
    # Do not apply CLAHE again inside Dataset (avoid double processing).
    selected_image_dir = CLAHE_IMAGE_DIR if bool(settings["use_clahe"]) else IMAGE_DIR
    if not selected_image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {selected_image_dir}. "
            "For CLAHE runs, execute src/precompute_clahe.py first."
        )

    dataset = CXR14Dataset(
        csv_path=csv_path,
        image_dir=selected_image_dir,
        use_clahe=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=shuffle,
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
    max_batches: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp else nullcontext()
        )
        with autocast_context:
            logits = model(images)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_idx}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No training batches were processed")
    return total_loss / total_samples


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: FocalLossWithLogits,
    device: torch.device,
    use_amp: bool,
    ece_bins: int,
    max_batches: int = 0,
) -> tuple[float, dict[str, float], list[dict[str, float | str | int]]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    targets_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp else nullcontext()
        )
        with autocast_context:
            logits = model(images)
            loss = criterion(logits, labels)

        probabilities = torch.sigmoid(logits.float())
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        targets_all.append(labels.cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())

    if total_samples == 0:
        raise RuntimeError("No validation batches were processed")

    y_true = np.concatenate(targets_all, axis=0)
    y_prob = np.concatenate(probabilities_all, axis=0)
    summary, per_class = compute_multilabel_metrics(
        y_true,
        y_prob,
        DISEASES,
        n_bins=ece_bins,
    )
    return total_loss / total_samples, summary, per_class


def update_fold_reports(
    metrics_dir: Path,
    fold_row: dict[str, Any],
    class_rows: list[dict[str, Any]],
    fold: int,
    run_tag: str,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fold_path = metrics_dir / "fold_metrics.csv"
    class_path = metrics_dir / "per_class_metrics.csv"

    new_fold = pd.DataFrame([fold_row])
    if fold_path.exists():
        old = pd.read_csv(fold_path)
        old = old[old["Fold"] != fold]
        new_fold = pd.concat([old, new_fold], ignore_index=True)
    new_fold = new_fold.sort_values("Fold")
    new_fold.to_csv(fold_path, index=False)

    new_class = pd.DataFrame(class_rows)
    if class_path.exists():
        old_class = pd.read_csv(class_path)
        old_class = old_class[old_class["Fold"] != fold]
        new_class = pd.concat([old_class, new_class], ignore_index=True)
    new_class = new_class.sort_values(["Fold", "Disease"])
    new_class.to_csv(class_path, index=False)

    # Summary across folds completed so far for this model/gamma combination.
    metric_df = pd.read_csv(fold_path)
    summary_row = {
        "Model": fold_row["Model"],
        "Gamma": fold_row["Gamma"],
        "Completed_Folds": int(metric_df["Fold"].nunique()),
    }
    for metric in ("Best_Val_AUROC", "Best_Val_AUPRC", "Best_Val_ECE"):
        values = pd.to_numeric(metric_df[metric], errors="coerce").dropna()
        summary_row[f"Mean_{metric.replace('Best_Val_', '')}"] = float(values.mean()) if len(values) else math.nan
        summary_row[f"Std_{metric.replace('Best_Val_', '')}"] = (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else math.nan
        )
    pd.DataFrame([summary_row]).to_csv(metrics_dir / "cv_summary.csv", index=False)

    # Cross-gamma comparison is progressively filled as gamma runs finish.
    # Keep debug/smoke results separate from reportable cross-validation results.
    model_dir = REPORT_ROOT / fold_row["Model"]
    comparison_rows = []
    for gamma_dir in sorted(model_dir.glob("gamma_*")):
        summary_file = gamma_dir / run_tag / "cv_summary.csv"
        if summary_file.exists():
            comparison_rows.append(pd.read_csv(summary_file))
    if comparison_rows:
        comparison = pd.concat(comparison_rows, ignore_index=True)
        comparison = comparison.sort_values("Gamma")
        comparison_path = model_dir / run_tag / "gamma_comparison.csv"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_path, index=False)


def run_fold(args: argparse.Namespace, settings: dict[str, Any], device: torch.device, fold: int) -> None:
    train_csv = SPLIT_DIR / f"fold_{fold}" / "train.csv"
    val_csv = SPLIT_DIR / f"fold_{fold}" / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Missing split CSV for fold {fold}: {train_csv} or {val_csv}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    verify_patient_split(fold, train_df, val_df)
    print(
        f"\n[Fold {fold}] Train {len(train_df):,} images / "
        f"{train_df['Patient ID'].nunique():,} patients | "
        f"Val {len(val_df):,} images / {val_df['Patient ID'].nunique():,} patients"
    )

    pos_weight, weight_report = calculate_pos_weight(train_df)
    device_pos_weight = pos_weight.to(device)
    train_loader = make_loader(train_csv, settings, shuffle=True)
    val_loader = make_loader(val_csv, settings, shuffle=False)

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
        T_max=int(settings["epochs"]),
        eta_min=0.0,
    )
    criterion = FocalLossWithLogits(
        gamma=args.gamma,
        pos_weight=device_pos_weight,
        reduction="mean",
    ).to(device)

    use_amp = bool(settings["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    gamma_tag = f"gamma_{args.gamma:g}"
    checkpoint_dir = CHECKPOINT_ROOT / args.model / gamma_tag / args.run_tag / f"fold_{fold}"
    log_dir = LOG_ROOT / args.model / gamma_tag / args.run_tag / f"fold_{fold}"
    metrics_dir = REPORT_ROOT / args.model / gamma_tag / args.run_tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    weight_report.to_csv(metrics_dir / f"pos_weight_fold_{fold}.csv", index=False)

    best_auc = -float("inf")
    best_epoch = 0
    best_summary: dict[str, float] | None = None
    best_per_class: list[dict[str, float | str | int]] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []
    best_checkpoint_path = checkpoint_dir / "best_model.pth"

    for epoch in range(1, int(settings["epochs"]) + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, use_amp, max_batches=args.max_train_batches,
        )
        val_loss, val_summary, val_per_class = validate(
            model, val_loader, criterion, device, use_amp,
            ece_bins=int(settings["ece_bins"]),
            max_batches=args.max_val_batches,
        )
        scheduler.step()

        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        row = {
            "Epoch": epoch,
            "Train_Loss": train_loss,
            "Val_Loss": val_loss,
            "Val_AUROC": val_summary["AUROC"],
            "Val_AUPRC": val_summary["AUPRC"],
            "Val_ECE": val_summary["ECE"],
            "AUROC_Valid_Classes": val_summary["AUROC_Valid_Classes"],
            "Backbone_LR": current_lrs[0] if current_lrs else math.nan,
            "Classifier_LR": current_lrs[1] if len(current_lrs) > 1 else math.nan,
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(log_dir / "history.csv", index=False)

        print(
            f"[Fold {fold}] Epoch {epoch:02d}/{settings['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"AUROC={val_summary['AUROC']:.4f} | "
            f"AUPRC={val_summary['AUPRC']:.4f} | ECE={val_summary['ECE']:.4f}"
        )

        current_auc = val_summary["AUROC"]
        improved = math.isfinite(current_auc) and (
            current_auc > best_auc + float(settings["min_delta"])
        )
        if improved:
            best_auc = current_auc
            best_epoch = epoch
            best_summary = val_summary
            best_per_class = val_per_class
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(DISEASES),
                    "diseases": DISEASES,
                    "gamma": float(args.gamma),
                    "fold": int(fold),
                    "best_epoch": int(epoch),
                    "best_val_auroc": float(val_summary["AUROC"]),
                    "best_val_auprc": float(val_summary["AUPRC"]),
                    "best_val_ece": float(val_summary["ECE"]),
                    "use_clahe": bool(settings["use_clahe"]),
                    "config": settings,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(settings["patience"]):
            print(
                f"[Fold {fold}] Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, best AUROC={best_auc:.4f}"
            )
            break

    if best_summary is None or best_per_class is None:
        raise RuntimeError(
            f"Fold {fold}: no finite validation AUROC was obtained. "
            "For a debug run, increase --max-val-batches or use a full validation pass."
        )

    fold_row = {
        "Model": args.model,
        "Gamma": float(args.gamma),
        "Fold": int(fold),
        "Best_Epoch": int(best_epoch),
        "Best_Val_AUROC": float(best_summary["AUROC"]),
        "Best_Val_AUPRC": float(best_summary["AUPRC"]),
        "Best_Val_ECE": float(best_summary["ECE"]),
        "Train_Images": int(len(train_df)),
        "Validation_Images": int(len(val_df)),
        "Train_Patients": int(train_df["Patient ID"].nunique()),
        "Validation_Patients": int(val_df["Patient ID"].nunique()),
        "Checkpoint": str(best_checkpoint_path.relative_to(PROJECT_DIR)),
        "CLAHE": bool(settings["use_clahe"]),
    }
    class_rows = [
        {"Model": args.model, "Gamma": float(args.gamma), "Fold": int(fold), **row}
        for row in best_per_class
    ]
    update_fold_reports(metrics_dir, fold_row, class_rows, fold, args.run_tag)
    print(f"[Fold {fold}] Best checkpoint: {best_checkpoint_path}")


def main() -> None:
    args = parse_args()
    if len(set(args.folds)) != len(args.folds) or any(fold not in range(1, 6) for fold in args.folds):
        raise ValueError("--folds must contain unique fold numbers from 1 to 5")
    if not args.run_tag or any(ch in args.run_tag for ch in "/\\"):
        raise ValueError("--run-tag must be a non-empty folder name without path separators")

    config = load_config(args.config)
    settings = resolve_settings(args, config)
    device = select_device(str(settings["device"]))

    selected_image_dir = CLAHE_IMAGE_DIR if bool(settings["use_clahe"]) else IMAGE_DIR
    if not selected_image_dir.is_dir():
        raise FileNotFoundError(
            f"Selected image directory does not exist: {selected_image_dir}. "
            "For CLAHE runs, execute src/precompute_clahe.py first."
        )

    all_split_paths = [SPLIT_DIR / f"fold_{fold}" / "train.csv" for fold in range(1, 6)]
    if not all(path.exists() for path in all_split_paths):
        raise FileNotFoundError(
            "Expected GroupKFold data at splits/fold_1..fold_5/train.csv. "
            "Check that splits/ contains the final GroupKFold output."
        )

    # Verify the fixed official Test set is patient-disjoint from CV data.
    full_cv = pd.read_csv(all_split_paths[0], usecols=["Patient ID"])
    fold1_val = pd.read_csv(SPLIT_DIR / "fold_1" / "val.csv", usecols=["Patient ID"])
    verify_official_test_patient_separation(pd.concat([full_cv, fold1_val], ignore_index=True))

    print("=" * 72)
    print("CXR14 Training Configuration")
    print("=" * 72)
    print(f"Model: {args.model} | gamma: {args.gamma:g} | folds: {args.folds}")
    print(f"Device: {device} | AMP: {bool(settings['amp']) and device.type == 'cuda'}")
    print(f"Epochs: {settings['epochs']} | Batch size: {settings['batch_size']} | Patience: {settings['patience']}")
    print(f"CLAHE: {settings['use_clahe']} ({'precomputed images' if settings['use_clahe'] else 'original images'}) | Pretrained: {settings['pretrained']}")
    if args.max_train_batches or args.max_val_batches:
        print("WARNING: batch limits are enabled; this is a smoke/debug run, not a reportable experiment.")

    for fold in args.folds:
        fold_seed = int(settings["seed"]) + int(fold)
        set_seed(fold_seed)
        run_fold(args, settings, device, fold)

    print("\nRequested folds completed.")
    print(f"Checkpoints: {CHECKPOINT_ROOT / args.model / f'gamma_{args.gamma:g}' / args.run_tag}")
    print(f"Reports: {REPORT_ROOT / args.model / f'gamma_{args.gamma:g}' / args.run_tag}")


if __name__ == "__main__":
    main()

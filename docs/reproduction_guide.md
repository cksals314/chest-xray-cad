# Reproduction Guide

This guide describes the environment and commands for reproducing the CXR-CAD workflow. The commands assume Linux, a CUDA-capable NVIDIA GPU for model training, and a project root named `chest-xray-cad`.

## 1. Environment setup

Tested environment:

- Python 3.12.14
- PyTorch 2.6.0+cu124
- Torchvision 0.21.0+cu124
- CUDA 12.4 PyTorch wheels

Create an environment and install dependencies:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Verify the installation:

```bash
python - <<'PY'
import torch, torchvision, numpy, pandas, sklearn, scipy, cv2
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("scipy:", scipy.__version__)
print("opencv:", cv2.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

The exact CUDA wheel build may need to be changed for a different CUDA/platform combination. The provided requirements file reflects the project environment used for the reported experiments.

## 2. Prepare the dataset

Place the required files under `data/`:

```text
data/
├── Data_Entry_2017.csv
├── train_val_list_NIH.txt
├── test_list_NIH.txt
└── images-224/
```

The image directory must contain the 224 × 224 PNG files whose names appear in the NIH metadata and official split lists. The dataset itself is not committed to GitHub.

Run the quality check and brightness audit:

```bash
python src/quality_check.py
python src/brightness_check.py
```

Generate patient-wise five-fold split manifests:

```bash
python src/prepare_data.py
```

The script writes `splits/fold_1/` through `splits/fold_5/`, `splits/final_trainval.csv`, `splits/final_test.csv`, and GroupKFold reports. Each manifest contains `Image Index`, `Patient ID`, and the 14 binary labels. The script uses the official Train/Val list to generate folds; the official Test list is kept separate.

If using the already committed manifests, do not regenerate them unless you intend to reproduce the split construction. `splits/split_metadata.json` describes the verified split configuration.

## 3. Optional CLAHE preprocessing

CLAHE is not part of the selected final configuration. To reproduce the optional precomputation experiment:

```bash
python src/precompute_clahe.py \
  --input-dir data/images-224 \
  --output-dir data/images-224-clahe \
  --clip-limit 2.0 \
  --grid-x 8 \
  --grid-y 8
```

The precomputed folder is local and is not committed. When training the CLAHE experiment, use `--clahe`. The selected final configuration uses CLAHE OFF.

## 4. Cross-validation training

The shared default configuration is `configs/train_config.json`. It specifies the maximum CV epoch budget, batch size, optimizer learning rates, weight decay, early-stopping patience, AMP, and CLAHE setting. Command-line arguments can override these values.

### 4.1 DenseNet121: gamma comparison

```bash
python src/train.py --model densenet121 --gamma 0 --folds 1 2 3 4 5 --run-tag cv
python src/train.py --model densenet121 --gamma 1 --folds 1 2 3 4 5 --run-tag cv
python src/train.py --model densenet121 --gamma 2 --folds 1 2 3 4 5 --run-tag cv
```

### 4.2 DenseNet121: CLAHE comparison

```bash
python src/train.py --model densenet121 --gamma 0 --clahe --folds 1 2 3 4 5 --run-tag cv_clahe
```

### 4.3 EfficientNet-B0: selected loss/preprocessing

```bash
python src/train.py --model efficientnet_b0 --gamma 0 --folds 1 2 3 4 5 --run-tag cv_off
```

The run tags only organize output directories. Training checkpoints and detailed logs are local/large artifacts and are excluded from the standard public Git history. Consolidated Fold metrics used by final training are tracked under `reports/cross_validation/`.

For a short pipeline smoke test, the training script supports `--max-train-batches` and `--max-val-batches`; do not use those limits for reported experiments.

## 5. Final model training

The final training script reads the consolidated Fold metrics from `reports/cross_validation/` and derives the number of final epochs from the mean CV best epoch (factor 0.8). First run a dry-run:

```bash
python src/train_final.py --model densenet121 --dry-run
python src/train_final.py --model efficientnet_b0 --dry-run
```

Expected selected settings:

- DenseNet121: gamma 0, CLAHE OFF, 4 epochs.
- EfficientNet-B0: gamma 0, CLAHE OFF, 3 epochs.

Then train if new checkpoints are required:

```bash
python src/train_final.py --model densenet121
python src/train_final.py --model efficientnet_b0
```

Existing final checkpoints are included under `checkpoints/final/`. The training script prevents accidental overwrite unless `--overwrite` or a new `--run-tag` is supplied.

## 6. OOF extraction (optional; requires Fold checkpoints)

OOF extraction does not train models. It loads each Fold's best checkpoint and predicts that Fold's patient-disjoint Validation set:

```bash
python src/extract_oof.py --model densenet121 --device cuda:0
python src/extract_oof.py --model efficientnet_b0 --device cuda:0
```

**Requirement:** all 10 corresponding Fold `best_model.pth` files must exist under `checkpoints/training/`. They are not included in the standard public release. The precomputed aggregate OOF files are included, so a user who only wants to reproduce calibration and downstream analyses can skip this step.

The script writes gzip-compressed Fold-level and aggregate OOF CSVs under `reports/oof/<model>/`.

## 7. Fit calibration parameters

Calibration is fitted from OOF logits only. The script does not open the official Test labels:

```bash
python src/fit_calibrators.py --models densenet121 efficientnet_b0
```

Outputs are saved under `reports/calibration/<model>/`, including `calibration_parameters.json` and `platt_parameters.csv`.

## 8. Official Test evaluation (optional inference rerun)

The final Test evaluator requires final model checkpoints, calibration parameters, the official metadata/list files, and all Test images:

```bash
python src/evaluate_test.py \
  --models densenet121 efficientnet_b0 \
  --device cuda:0 \
  --batch-size 64 \
  --num-workers 4
```

By default, horizontal-flip TTA is evaluated as well. Use `--no-tta` to skip it. This script writes full local predictions to `reports/test_evaluation/test_predictions.csv`; the full file is ignored by Git. The compact public artifact `test_predictions_public.csv.gz` contains the selected Platt probabilities needed for downstream analyses.

Do not tune models, calibrators, thresholds, or preprocessing based on the official Test results and then report the same Test metrics as unbiased estimates.

## 9. Operating-point analysis

The default script uses compressed aggregate OOF files, saved Platt parameters, and the compact public Test prediction artifact:

```bash
python src/operating_points.py
```

Thresholds are selected and saved from OOF data before the Test prediction file is opened. Outputs:

- `reports/operating_points/thresholds_oof.csv`
- `reports/operating_points/test_operating_points.csv`
- `reports/operating_points/operating_points_summary.csv`
- `reports/operating_points/operating_points_metadata.json`

## 10. Disease-wise ranking analysis

```bash
python src/analyze_disease_metrics.py
```

The script reads `test_predictions_public.csv.gz` and the frozen Test operating-point results, then writes disease-wise AUROC/AUPRC and combined ranking/operating-point reports under `reports/disease_level_analysis/`.

## 11. Useful artifact formats

Pandas can read the tracked `.csv.gz` artifacts directly:

```python
import pandas as pd
pred = pd.read_csv("reports/test_evaluation/test_predictions_public.csv.gz")
oof = pd.read_csv("reports/oof/densenet121/densenet121_oof.csv.gz")
```

The project uses gzip-compressed aggregate prediction files to reduce repository size while preserving the CSV content. The public Test prediction artifact intentionally omits age, gender, and view-position columns.

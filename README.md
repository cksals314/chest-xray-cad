# Chest X-ray Multi-Disease CAD (CXR-CAD)

A multi-label chest X-ray classification project based on the NIH ChestX-ray14 dataset. The project compares ImageNet-pretrained DenseNet121 and EfficientNet-B0 models, applies probability calibration, combines model probabilities with soft voting, and evaluates disease-wise operating points.

> **Research prototype only.** This project is not a medical device and must not be used for clinical diagnosis, triage, or treatment decisions.

## Project scope

This repository covers the workflow from data quality inspection and preprocessing through model training, calibration, Test evaluation, disease-wise analysis, and operating-point selection. Grad-CAM / visual explanation work is handled separately by another project contributor.

## Dataset

- NIH ChestX-ray14: 112,120 frontal chest radiographs from 30,805 patients, with 14 image-level thoracic disease labels.
- The project uses 224 × 224 resized PNG images, the NIH metadata CSV, and the official Train/Val and Test image lists.
- The image dataset is **not included** in this repository. See [`docs/data_description.md`](docs/data_description.md) for the expected local layout and sources.
- The official NIH Train/Val and Test lists are kept separate. Five-fold validation is performed with patient-level `GroupKFold` to prevent patient overlap.

## Selected approach

- **Models:** ImageNet-pretrained DenseNet121 and EfficientNet-B0; each classifier outputs 14 logits.
- **Loss:** class-weighted BCEWithLogits (the custom focal-loss implementation with `gamma=0`).
- **Preprocessing:** grayscale-to-RGB, tensor conversion, ImageNet normalization; no data augmentation.
- **CLAHE:** evaluated, but not selected.
- **Calibration:** per-disease Platt Scaling fitted on out-of-fold (OOF) logits.
- **Ensemble:** arithmetic mean of DenseNet121 and EfficientNet-B0 Platt probabilities.
- **Operating points:** disease-specific thresholds selected from OOF predictions, then evaluated on the untouched official Test split.

## Main Test results

Macro averages across the 14 diseases on the official Test split:

| Model / method | AUROC | AUPRC (Average Precision) | Mean per-disease ECE |
|---|---:|---:|---:|
| DenseNet121 + Platt | 0.8044 | 0.2699 | 0.0120 |
| EfficientNet-B0 + Platt | 0.7878 | 0.2448 | 0.0119 |
| **Soft Voting Ensemble + Platt** | **0.8083** | **0.2744** | **0.0120** |

For the selected Soft Voting Ensemble, the macro Test operating-point results were:

| OOF threshold criterion | Sensitivity | Specificity | PPV | NPV | F1 |
|---|---:|---:|---:|---:|---:|
| Youden J | 0.8316 | 0.6206 | 0.1442 | 0.9757 | 0.2340 |
| Sensitivity ≥ 0.90 on OOF | 0.9442 | 0.3856 | 0.1084 | 0.9857 | 0.1863 |
| Specificity ≥ 0.90 on OOF | 0.6309 | 0.8141 | 0.1944 | 0.9617 | 0.2774 |

These are research benchmark results, not evidence of clinical readiness. The disease prevalence and operating-point results vary substantially by label; see the detailed report.

## Repository structure

```text
cxr-cad/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── train_config.json
├── data/                         # local only; not tracked
├── src/                          # preprocessing, training, calibration, evaluation
├── scripts/                      # utility scripts
├── splits/                        # public 16-column split manifests
├── checkpoints/final/             # final DenseNet121 / EfficientNet-B0 weights
├── reports/
│   ├── cross_validation/          # consolidated CV comparison and Fold metrics
│   ├── calibration/
│   ├── oof/                       # compressed aggregate OOF predictions
│   ├── test_evaluation/           # compact public Test predictions and metrics
│   ├── operating_points/
│   └── disease_level_analysis/
└── docs/
    ├── experiment_report.md
    ├── reproduction_guide.md
    └── data_description.md
```

## Documentation

- [Detailed experiment report](docs/experiment_report.md): methods, experiments, metrics, interpretation, and limitations from preprocessing through operating-point analysis.
- [Reproduction guide](docs/reproduction_guide.md): environment setup, expected data layout, and execution order.
- [Data description](docs/data_description.md): labels, split design, and shared artifact schemas.

## Quick start

1. Create a Python 3.12 environment and install packages from `requirements.txt` (the file pins the CUDA 12.4 PyTorch build used in this project).
2. Download and place the 224 × 224 NIH images and required NIH metadata/list files under `data/` as described in the reproduction guide.
3. Use the committed split manifests and run the scripts in the documented order.

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

See [`docs/reproduction_guide.md`](docs/reproduction_guide.md) before running training or evaluation.

## Data and paper attribution

The image files are not redistributed here. Obtain the dataset from the [NIH ChestX-ray dataset download page](https://nihcc.app.box.com/v/ChestXray-NIHCC). This project also uses a 224 × 224 resized image package; see the [Kaggle resized dataset listing](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized) and confirm its files match the expected NIH filenames and metadata.

Please acknowledge the NIH Clinical Center and cite the original dataset paper:

Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3462–3471. https://doi.org/10.1109/CVPR.2017.369

The NIH dataset was subsequently expanded to 14 disease labels and is commonly referred to as ChestX-ray14. Follow the attribution instructions on the [NIH download page](https://nihcc.app.box.com/v/ChestXray-NIHCC).

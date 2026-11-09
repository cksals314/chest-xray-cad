# Data Description

## 1. Dataset source and local layout

The project uses NIH ChestX-ray14 and 224 × 224 resized PNG images. The original image dataset is not stored in this Git repository. Obtain the data from the NIH download page and the referenced 224 × 224 resized image source, then verify all filenames against the official metadata and image lists.

Expected local layout:

```text
data/
├── Data_Entry_2017.csv
├── train_val_list_NIH.txt
├── test_list_NIH.txt
└── images-224/
    ├── 00000001_000.png
    ├── ...
    └──
```

`data/` is excluded by `.gitignore`. Do not upload the source images or the original full metadata CSV to this repository.

Sources:

- [NIH ChestX-ray dataset download page](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [224 × 224 resized NIH ChestX-ray14 image package](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)
- [Original dataset paper, Wang et al. (2017)](https://doi.org/10.1109/CVPR.2017.369)

## 2. Label schema

The model predicts the following 14 labels in this exact order:

```text
Atelectasis
Cardiomegaly
Effusion
Infiltration
Mass
Nodule
Pneumonia
Pneumothorax
Consolidation
Edema
Emphysema
Fibrosis
Pleural_Thickening
Hernia
```

Each label is binary (`0` or `1`) and multiple labels may be positive for one image. `No Finding` is encoded as all 14 labels equal to zero; it is not a fifteenth class.

## 3. Official partitions and GroupKFold

| Manifest | Images | Patients | Purpose |
|---|---:|---:|---|
| `splits/final_trainval.csv` | 86,524 | 28,008 | Full official Train/Val population for final training |
| `splits/final_test.csv` | 25,596 | 2,797 | Official held-out Test population |

The 5-fold manifests are stored as:

```text
splits/
├── split_metadata.json
├── final_trainval.csv
├── final_test.csv
├── fold_1/{train,val}.csv
├── fold_2/{train,val}.csv
├── fold_3/{train,val}.csv
├── fold_4/{train,val}.csv
└── fold_5/{train,val}.csv
```

Every public split CSV contains exactly 16 columns:

1. `Image Index`
2. `Patient ID`
3. the 14 disease labels in the order above

Patient ID is retained because the GroupKFold manifests and integrity checks must preserve patient-level separation. Demographic and acquisition fields (age, gender, view position, image dimensions, and pixel spacing) are not included in the public split manifests.

The split metadata records GroupKFold settings and integrity checks. Verified properties include no patient overlap between fold Train and Validation, each Train/Val image appearing in Validation exactly once across the five folds, and no patient overlap between official Train/Val and Test.

## 4. Main shared prediction artifacts

### 4.1 Compact Test predictions

`reports/test_evaluation/test_predictions_public.csv.gz` contains 25,596 rows and 58 columns:

| Column group | Count | Description |
|---|---:|---|
| `Image Index` | 1 | Image filename/key |
| `Patient ID` | 1 | Pseudonymous patient grouping key |
| Disease labels | 14 | Ground-truth 0/1 labels |
| `densenet121_platt_<Disease>` | 14 | DenseNet121 Platt probabilities |
| `efficientnet_b0_platt_<Disease>` | 14 | EfficientNet-B0 Platt probabilities |
| `soft_voting_ensemble_platt_<Disease>` | 14 | Arithmetic-mean Ensemble Platt probabilities |
| **Total** | **58** | |

The file deliberately omits age, gender, view position, raw logits, Temperature Scaling predictions, and TTA variants. It is intended for disease-wise ranking analysis, operating-point work, and downstream subgroup analysis after joining an authorized copy of the official NIH metadata if required.

### 4.2 Aggregate OOF predictions

The aggregate OOF files are:

- `reports/oof/densenet121/densenet121_oof.csv.gz`
- `reports/oof/efficientnet_b0/efficientnet_b0_oof.csv.gz`

Each contains 86,524 rows and 45 columns: image/patient identifiers, 14 labels, Fold assignment, 14 logits, and 14 sigmoid probabilities. OOF predictions are produced by the Fold model that did not train on the corresponding validation patients.

### 4.3 Operating-point artifacts

- `thresholds_oof.csv`: per-model, per-disease thresholds selected using OOF data.
- `test_operating_points.csv`: Test sensitivity, specificity, PPV, NPV, F1, confusion counts, and threshold criterion.
- `operating_points_summary.csv`: macro averages across 14 diseases.

## 5. Reading compressed CSV files

Pandas reads `.csv.gz` directly:

```python
import pandas as pd

predictions = pd.read_csv(
    "reports/test_evaluation/test_predictions_public.csv.gz",
    dtype={"Image Index": "string", "Patient ID": "string"},
)
```

No manual decompression is required.

## 6. Data sharing note

The public Test prediction file includes Image Index and Patient ID to preserve image-level linkage and patient-group analysis. It omits demographic/acquisition metadata. Although the source dataset is publicly available and de-identified, Patient ID is still a pseudonymous linkage field; share this artifact in accordance with the project's data-governance decision and the NIH source's attribution instructions.

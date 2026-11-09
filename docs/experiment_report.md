# CXR-CAD Experiment Report

## 1. Objective

The objective was to build and evaluate a 14-label multi-label classifier for frontal chest X-ray images from NIH ChestX-ray14. Two ImageNet-pretrained CNN backbones, DenseNet121 and EfficientNet-B0, were trained and compared. The selected models were calibrated using out-of-fold predictions, combined using soft voting, and evaluated on the official NIH Test split. Disease-specific operating thresholds were selected from OOF data and then evaluated on Test data without using Test labels for threshold selection.

This is a research prototype, not a clinically validated diagnostic system.

## 2. Dataset and labels

The dataset contains 112,120 resized 224 × 224 PNG images from 30,805 unique patients. Each image can have multiple image-level labels. The 14 target labels are:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

`No Finding` is represented by fourteen zeros. It is not an additional output class.

The official image lists were retained:

| Partition | Images | Patients |
|---|---:|---:|
| Official Train/Val | 86,524 | 28,008 |
| Official Test | 25,596 | 2,797 |
| Total in these partitions | 112,120 | 30,805 |

The official Test split was not used to construct the five CV folds. Patient IDs were used as grouping variables so that images from one patient could not appear in both Train and Validation within a fold.

## 3. Data quality inspection and preprocessing

### 3.1 Integrity checks

The image directory was checked against the metadata and official image lists. All images used by the project were 224 × 224. The inspected image modes were 111,601 grayscale `L` images and 519 `LA` images; the alpha channel of the `LA` images was fully opaque. No image was removed by the quality/brightness audit used for this project.

### 3.2 Brightness audit

An image-level brightness audit flagged 126 extreme-brightness candidates across the dataset:

- Mean intensity below 50: 75 images.
- Mean intensity above 195: 51 images.
- Age parsed successfully for 122 candidates: 83 were under 18 and 39 were adults; 4 ages were invalid/unavailable.

These candidates were retained. The extreme values could include valid pediatric images or clinically meaningful acquisition/content variation, so automatic deletion based only on brightness was considered too risky. The shared Train/Val extreme-image table omits Patient ID and retains image-level brightness statistics.

### 3.3 Image transformation

The selected image pipeline was:

1. Load the 224 × 224 PNG image.
2. Convert to grayscale (`L`).
3. Convert the grayscale image to three-channel RGB.
4. Convert to a tensor in the `[0, 1]` range.
5. Normalize using ImageNet channel statistics: mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

No data augmentation was applied in the final training pipeline.

### 3.4 CLAHE experiment

CLAHE was tested with `clipLimit=2.0` and `tileGridSize=(8, 8)`. A precomputed CLAHE version of the 224 × 224 images was generated so that CLAHE would not be applied twice during training. The original images were preserved.

DenseNet121 with `gamma=0` was compared using five patient-wise folds:

| Metric (mean ± fold SD) | CLAHE OFF | CLAHE ON |
|---|---:|---:|
| AUROC | **0.819253 ± 0.006935** | 0.815676 ± 0.006396 |
| AUPRC (Average Precision) | **0.219918 ± 0.012519** | 0.209754 ± 0.011844 |
| ECE | **0.234622 ± 0.015771** | 0.257407 ± 0.040358 |

CLAHE did not improve the evaluated metrics. CLAHE OFF was selected for subsequent model training and final evaluation.

## 4. Patient-wise data splitting

The official Train/Val list was partitioned using `sklearn.model_selection.GroupKFold(n_splits=5)` with `Patient ID` as the group. The public split manifests contain only 16 columns: `Image Index`, `Patient ID`, and the 14 binary disease labels.

| Fold | Train images | Validation images | Train patients | Validation patients |
|---|---:|---:|---:|---:|
| 1 | 69,219 | 17,305 | 22,407 | 5,601 |
| 2 | 69,219 | 17,305 | 22,406 | 5,602 |
| 3 | 69,219 | 17,305 | 22,406 | 5,602 |
| 4 | 69,219 | 17,305 | 22,406 | 5,602 |
| 5 | 69,220 | 17,304 | 22,407 | 5,601 |

Integrity checks passed:

- No patient overlap between Train and Validation within any fold.
- Each fold's Train and Validation images partitioned the complete official Train/Val image set.
- Each Train/Val image appeared in Validation exactly once across the five folds.
- No patient overlap between the official Train/Val and Test partitions (28,008 vs. 2,797 patients; overlap = 0).

## 5. Models and training configuration

### 5.1 Architectures

Two torchvision ImageNet-pretrained models were used:

- DenseNet121
- EfficientNet-B0

The original 1,000-class classification heads were replaced with fully connected layers producing 14 raw logits. Sigmoid was applied only when converting logits to probabilities for evaluation, calibration, or thresholding.

### 5.2 Loss function

A custom multi-label focal-loss implementation was used. It is based on numerically stable `BCEWithLogits` and class-wise `pos_weight` values calculated from the current training partition only:

\[
\mathcal{L}_{focal} = (1-p_t)^\gamma \, \mathcal{L}_{BCEWithLogits,pos\_weight}
\]

where \(p_t = yp + (1-y)(1-p)\), and \(p=\sigma(z)\). With \(\gamma=0\), the implementation is exactly weighted `BCEWithLogitsLoss`. Positive weights were computed separately for each disease using the negative-to-positive count ratio in the corresponding Fold Train partition. For final training, weights were recomputed on the complete Train/Val set.

### 5.3 Main configuration

| Setting | Value |
|---|---:|
| Random seed | 42 |
| Maximum CV epochs | 30 |
| Batch size | 32 |
| DataLoader workers | 4 |
| Backbone learning rate | 1e-4 |
| Classifier learning rate | 5e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Early stopping patience | 5 epochs |
| Early stopping monitor | Validation AUROC |
| ECE bins | 15 |
| Mixed precision | Enabled on CUDA |
| Pretrained weights | Enabled |
| CLAHE | Disabled in selected configuration |
| Data augmentation | None |

`epochs=30` is the maximum CV training budget, not the final-model epoch count.

## 6. Focal-loss gamma experiment

The gamma comparison was performed with DenseNet121 under the selected non-CLAHE preprocessing configuration. Each setting used five patient-wise folds.

| Gamma | Mean AUROC ± SD | Mean AUPRC ± SD | Mean ECE ± SD |
|---:|---:|---:|---:|
| **0** | **0.819253 ± 0.006935** | **0.219918 ± 0.012519** | **0.234622 ± 0.015771** |
| 1 | 0.816698 ± 0.007147 | 0.218925 ± 0.009345 | 0.278542 ± 0.023905 |
| 2 | 0.815492 ± 0.006340 | 0.214639 ± 0.011525 | 0.322932 ± 0.025814 |

Gamma 0 achieved the best mean AUROC, AUPRC, and ECE among the tested settings. It was therefore selected. In this implementation, gamma 0 corresponds exactly to class-weighted BCEWithLogitsLoss.

## 7. Five-fold model comparison

After selecting gamma 0 and CLAHE OFF, DenseNet121 and EfficientNet-B0 were compared using the five patient-wise folds.

| Model | Mean AUROC ± SD | Mean AUPRC ± SD | Mean ECE ± SD | Mean best epoch |
|---|---:|---:|---:|---:|
| **DenseNet121** | **0.819253 ± 0.006935** | **0.219918 ± 0.012519** | **0.234622 ± 0.015771** | 5.0 |
| EfficientNet-B0 | 0.807096 ± 0.004610 | 0.196219 ± 0.008090 | 0.256237 ± 0.012624 | 3.8 |

DenseNet121 outperformed EfficientNet-B0 on the reported mean CV ranking and calibration metrics. Both models were retained for later soft-voting comparison.

## 8. Final model training

Each final model was trained using all 86,524 official Train/Val images and 28,008 patients. The epoch count was scaled from the mean CV best epoch by a factor of 0.8, reflecting the approximate fraction of Train/Val samples used to train each CV Fold:

\[
E_{final} = \max(1, \mathrm{round}(0.8 \times \overline{E}_{best, CV}))
\]

| Model | Mean CV best epoch | Final epochs |
|---|---:|---:|
| DenseNet121 | 5.0 | 4 |
| EfficientNet-B0 | 3.8 | 3 |

The final runs used gamma 0, CLAHE OFF, AdamW, CosineAnnealingLR, and the full Train/Val-derived `pos_weight`. Test data was not used for gradient updates or epoch selection; the final training script checks only the Train/Val–Test patient separation.

Final checkpoints:

- `checkpoints/final/densenet121/gamma_0/final_off/final_model.pth`
- `checkpoints/final/efficientnet_b0/gamma_0/final_off/final_model.pth`

## 9. Out-of-fold predictions and probability calibration

### 9.1 OOF prediction extraction

For each model, predictions were generated on the Validation Fold for which the corresponding model had not been trained. The combined OOF files contain 86,524 images and 28,008 patients per model. DenseNet121 and EfficientNet-B0 OOF files were verified to have matching image IDs, patient IDs, Fold assignments, and labels. The combined OOF files also matched the concatenation of their five Fold-level exports.

### 9.2 Calibration methods

Two calibration approaches were fitted using OOF logits only:

- **Temperature Scaling:** one positive scalar temperature per model.
- **Platt Scaling:** one slope and intercept per disease per model, applied as \(\sigma(a_d z_d + b_d)\).

Fitted temperatures:

| Model | Temperature |
|---|---:|
| DenseNet121 | 1.161068 |
| EfficientNet-B0 | 1.229136 |

The OOF fitting diagnostics were:

| Model | Uncalibrated OOF fit NLL | Temperature OOF fit NLL | Platt macro OOF fit NLL |
|---|---:|---:|---:|
| DenseNet121 | 0.427957 | 0.425411 | 0.134883 |
| EfficientNet-B0 | 0.464469 | 0.459569 | 0.136880 |

**These values are optimization-fit diagnostics on the same OOF rows used to fit the calibrators; they are not unbiased calibration estimates.** Final calibration assessment is based on the official Test results below. Calibration was not fitted on Test labels.

## 10. Official Test evaluation and TTA

The final checkpoints were evaluated on 25,596 official Test images from 2,797 patients. Metrics were calculated as macro averages across the 14 diseases. AUPRC uses `sklearn.metrics.average_precision_score` (Average Precision). ECE uses 15 bins.

### 10.1 Without Test-Time Augmentation

| Model | Probability method | Macro AUROC | Macro AUPRC (AP) | Mean per-disease ECE | Global ECE |
|---|---|---:|---:|---:|---:|
| DenseNet121 | Raw | 0.804381 | 0.269877 | 0.298131 | 0.298131 |
| DenseNet121 | Temperature | 0.804381 | 0.269877 | 0.303505 | 0.303503 |
| DenseNet121 | **Platt** | **0.804381** | **0.269877** | **0.011974** | **0.009112** |
| EfficientNet-B0 | Raw | 0.787792 | 0.244758 | 0.327020 | 0.327020 |
| EfficientNet-B0 | Temperature | 0.787792 | 0.244758 | 0.333276 | 0.333276 |
| EfficientNet-B0 | **Platt** | **0.787792** | **0.244758** | **0.011936** | **0.011062** |
| Soft Voting Ensemble | Raw | 0.807768 | 0.271602 | 0.312575 | 0.312575 |
| Soft Voting Ensemble | Temperature | 0.808029 | 0.272085 | 0.318390 | 0.318390 |
| Soft Voting Ensemble | **Platt** | **0.808302** | **0.274402** | **0.011981** | **0.008994** |

Platt Scaling substantially reduced ECE on Test. Positive per-disease Platt slopes preserve the ranking of each disease's scores, so AUROC and Average Precision remain unchanged for individual models. The ensemble scores are the arithmetic mean of model-specific calibrated probabilities.

### 10.2 Horizontal-flip TTA

TTA averages the original-view probability and horizontal-flip probability after applying the corresponding calibration transform.

| Model | Probability method | Macro AUROC | Macro AUPRC (AP) | Mean per-disease ECE |
|---|---|---:|---:|---:|
| DenseNet121 | Raw + TTA | 0.802309 | 0.271053 | 0.311417 |
| DenseNet121 | Temperature + TTA | 0.802557 | 0.271775 | 0.316731 |
| DenseNet121 | Platt + TTA | 0.802293 | 0.273449 | 0.016854 |
| EfficientNet-B0 | Raw + TTA | 0.789437 | 0.245544 | 0.340908 |
| EfficientNet-B0 | Temperature + TTA | 0.789530 | 0.245731 | 0.346153 |
| EfficientNet-B0 | Platt + TTA | 0.789647 | 0.246998 | 0.012602 |
| Soft Voting Ensemble | Raw + TTA | 0.806282 | 0.270325 | 0.326162 |
| Soft Voting Ensemble | Temperature + TTA | 0.806614 | 0.270986 | 0.331441 |
| Soft Voting Ensemble | Platt + TTA | 0.806771 | **0.277034** | 0.015979 |

TTA did not consistently improve AUROC or calibration. It slightly improved the Ensemble AUPRC but reduced AUROC and increased ECE compared with the non-TTA Platt Ensemble. The non-TTA Platt Ensemble was retained as the primary result.

## 11. Soft Voting Ensemble

For each disease, the Ensemble probability is the arithmetic mean of the two per-model Platt-calibrated probabilities:

\[
p_{ens,d} = \frac{p_{DenseNet121,d} + p_{EfficientNetB0,d}}{2}
\]

On the official Test set, the selected Platt Ensemble achieved macro AUROC 0.808302, macro AUPRC 0.274402, mean per-disease ECE 0.011981, and global ECE 0.008994. Its Test probabilities were checked against the arithmetic mean of the two component probabilities; the maximum absolute difference was approximately \(7.5\times10^{-10}\).

## 12. Disease-wise ranking and operating-point results

The following table combines disease-wise Test ranking metrics with the Test confusion metrics obtained from the OOF-frozen Youden J thresholds for the Soft Voting Ensemble.

| Disease | AUROC | AUPRC (AP) | Sensitivity | Specificity | PPV | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Atelectasis | 0.7585 | 0.3182 | 81.0% | 56.0% | 21.1% | 33.5% |
| Cardiomegaly | 0.8892 | 0.3376 | 83.5% | 77.4% | 13.8% | 23.7% |
| Effusion | 0.8210 | 0.4962 | 89.9% | 54.9% | 30.6% | 45.7% |
| Infiltration | 0.7001 | 0.3927 | 86.2% | 35.9% | 29.6% | 44.0% |
| Mass | 0.8018 | 0.2767 | 79.2% | 64.8% | 13.9% | 23.6% |
| Nodule | 0.7383 | 0.1967 | 70.1% | 63.9% | 11.6% | 19.9% |
| Pneumonia | 0.7212 | 0.0454 | 84.1% | 42.4% | 2.7% | 5.2% |
| Pneumothorax | 0.8642 | 0.4035 | 84.0% | 74.3% | 27.5% | 41.4% |
| Consolidation | 0.7429 | 0.1569 | 90.9% | 40.4% | 10.4% | 18.7% |
| Edema | 0.8497 | 0.1693 | 91.4% | 60.9% | 8.1% | 14.8% |
| Emphysema | 0.9128 | 0.4006 | 86.9% | 80.8% | 16.8% | 28.2% |
| Fibrosis | 0.8201 | 0.0980 | 66.7% | 82.1% | 6.1% | 11.1% |
| Pleural_Thickening | 0.7619 | 0.1280 | 78.7% | 60.2% | 8.5% | 15.3% |
| Hernia | 0.9345 | 0.4220 | 91.9% | 74.7% | 1.2% | 2.4% |

### 12.1 Interpretation

- **Effusion** had the strongest AUPRC and F1 among the 14 labels in this operating-point table. Its PPV was also the highest (30.6%).
- **Pneumothorax** combined a relatively high AUROC/AUPRC with a more balanced sensitivity and specificity than several other labels.
- **Infiltration** had the lowest AUROC (0.7001) and low specificity at the Youden threshold (35.9%), indicating substantial false-positive burden at that operating point.
- **Pneumonia** had low AUPRC (0.0454), PPV (2.7%), and F1 (5.2%). This suggests a ranking/precision challenge in addition to threshold sensitivity.
- **Hernia** had the highest AUROC (0.9345) and a high AUPRC (0.4220), yet its PPV and F1 at the selected Youden threshold were only 1.2% and 2.4%. Ranking ability and fixed-threshold performance are different properties, and the low positive prevalence makes PPV especially sensitive to false positives.
- **Fibrosis** had high specificity (82.1%) but lower sensitivity (66.7%), reflecting a more conservative operating point for this disease.

### 12.2 Operating-point selection criteria

All thresholds were selected per disease from OOF probabilities, then frozen before the Test prediction/label file was loaded:

- **Youden J:** maximize sensitivity + specificity − 1; ties select the higher threshold.
- **Sensitivity90:** select the highest threshold whose OOF sensitivity is at least 0.90.
- **Specificity90:** select the lowest threshold whose OOF specificity is at least 0.90.

Positive prediction rule: probability ≥ threshold.

### 12.3 Macro Test operating-point comparison

| Model | OOF threshold criterion | Test Sensitivity | Test Specificity | Test PPV | Test NPV | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| DenseNet121 | Youden J | 0.8197 | 0.6271 | 0.1432 | 0.9749 | 0.2325 |
| DenseNet121 | Sensitivity90 | 0.9479 | 0.3674 | 0.1061 | 0.9860 | 0.1827 |
| DenseNet121 | Specificity90 | 0.6244 | 0.8158 | 0.1957 | 0.9608 | 0.2779 |
| EfficientNet-B0 | Youden J | 0.8068 | 0.6112 | 0.1354 | 0.9734 | 0.2210 |
| EfficientNet-B0 | Sensitivity90 | 0.9521 | 0.3203 | 0.1003 | 0.9845 | 0.1734 |
| EfficientNet-B0 | Specificity90 | 0.5739 | 0.8259 | 0.1887 | 0.9576 | 0.2637 |
| Soft Voting Ensemble | Youden J | **0.8316** | 0.6206 | 0.1442 | 0.9757 | 0.2340 |
| Soft Voting Ensemble | Sensitivity90 | 0.9442 | 0.3856 | 0.1084 | 0.9857 | 0.1863 |
| Soft Voting Ensemble | Specificity90 | 0.6309 | 0.8141 | 0.1944 | 0.9617 | **0.2774** |

Sensitivity90 increased Test sensitivity but decreased specificity and PPV. Specificity90 increased specificity and produced the highest macro F1 for the Ensemble (0.2774), narrowly below DenseNet121's 0.2779 in its own Specificity90 operating point. Youden J offers a middle-ground operating point, rather than a universally optimal clinical threshold.

## 13. Limitations and discussion

1. **Weak image-level labels:** NIH disease labels are derived from radiology reports and are not equivalent to independently adjudicated pixel-level or image-level ground truth.
2. **Class imbalance:** Positive prevalence differs substantially by disease. This contributes to low PPV for rare findings even when sensitivity or AUROC is comparatively high.
3. **OOF–Test distribution differences:** All 14 disease positive rates were higher in the official Test split than in the Train/Val OOF population. This can affect PPV/NPV and may accompany changes in score distributions. Prevalence shift alone does not explain specificity changes.
4. **Calibration transfer:** Platt parameters were fitted on OOF predictions generated by Fold models and then applied to final models trained on all Train/Val data. The training-population/model-size mismatch may affect calibration transfer. Test ECE was low in this experiment, but this approach should be validated on an independent cohort before any practical use.
5. **Resolution and preprocessing:** The experiment used pre-resized 224 × 224 images. Fine radiographic detail may be lost compared with higher-resolution processing.
6. **Limited augmentation/model search:** No data augmentation was used, and the gamma comparison was performed with DenseNet121; the EfficientNet-B0 comparison used gamma 0. More architectures and training strategies may improve results.
7. **Single dataset/test cohort:** Results are from one official NIH Test split and do not establish generalization to other hospitals, devices, patient groups, or acquisition protocols.
8. **Operating points are not clinical thresholds:** Thresholds were selected by statistical criteria on OOF predictions. They should not be interpreted as clinically approved decision thresholds.

## 14. Outputs and artifacts

Important outputs in the repository:

| Path | Description |
|---|---|
| `reports/cross_validation/gamma_comparison.csv` | DenseNet121 gamma 0/1/2 comparison |
| `reports/cross_validation/clahe_comparison.csv` | DenseNet121 CLAHE OFF/ON comparison |
| `reports/cross_validation/model_comparison.csv` | DenseNet121 vs EfficientNet-B0 CV summary |
| `reports/cross_validation/*_fold_metrics.csv` | Consolidated Fold-level metrics used by final epoch selection |
| `reports/oof/*/*.csv.gz` | Aggregate OOF logits/labels/probabilities |
| `reports/calibration/*/platt_parameters.csv` | Per-disease Platt parameters |
| `reports/test_evaluation/test_predictions_public.csv.gz` | Compact image-level Test labels and Platt probabilities |
| `reports/operating_points/thresholds_oof.csv` | Frozen OOF-selected thresholds |
| `reports/operating_points/test_operating_points.csv` | Disease-wise Test confusion metrics |
| `reports/disease_level_analysis/test_disease_ranking_operating_points.csv` | Combined Test ranking and operating-point results |

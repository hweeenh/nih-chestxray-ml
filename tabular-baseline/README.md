# NIH Chest X-ray: Tabular Metadata Baseline

## Overview

This project establishes a **metadata-only baseline** for predicting Effusion using the NIH ChestX-ray14 dataset.

**Key finding:** Tabular metadata alone yields a mean ROC-AUC of approximately **0.67** for Effusion prediction, establishing a baseline for comparison against image-based models to be developed.

## Methodology

- **Dataset:** NIH ChestX-ray14 (`Data_Entry_2017.csv`)
- **Target:** Binary Effusion label derived from `Finding Labels`
- **Features:** Patient age, gender, view position (AP/PA), and follow-up number
- **Models:** Logistic Regression, Random Forest, XGBoost
- **Evaluation:** 5-fold Stratified Group Cross-Validation, grouped by patient ID
- **Interpretability:** SHAP feature importance and dependence plots

## Results

| Model | ROC-AUC | Average Precision |
|---|---:|---:|
| Logistic Regression | 0.637 | 0.178 |
| Random Forest | 0.561 | 0.152 |
| XGBoost | **0.675** | **0.192** |

XGBoost achieved the strongest ranking performance. However, it produced almost no positive predictions at the default 0.5 threshold, highlighting the importance of evaluating threshold-dependent metrics separately.

SHAP identified follow-up number as the most influential feature, followed by patient age. Removing follow-up number reduced ROC-AUC to 0.582, suggesting substantial dependence on this feature and possible dataset-specific patterns.

## Clinical Interpretation

Metadata contains some predictive signal for Effusion, but its performance is limited. View position had relatively little influence on the fitted model, despite differences in Effusion prevalence between AP and PA images in the dataset. These findings represent associations with dataset labels, not causal relationships or confirmed clinical diagnoses.

## Next Phase: Image-Based Modelling

**Phase 3 — CNN on chest X-ray images**

The next stage will train an image-based model to learn visual features directly from chest X-ray pixels. Its performance will be compared against this metadata-only ROC-AUC baseline of 0.675 to investigate whether radiographic image features provide additional predictive value.

The longer-term goal is to compare metadata-only, image-only, and potentially combined models using patient-level validation.

## Limitations

- NIH labels may be noisy and are not independently verified diagnoses.
- Follow-up number may capture dataset-specific patterns.
- Internal cross-validation does not establish generalisation to other hospitals.
- This educational project is not intended for clinical diagnosis.
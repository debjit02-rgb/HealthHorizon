# Health Horizon - Model Evaluation Report

> **DISCLAIMER:** Health Horizon is strictly an educational and research prototype. It is NOT a medical diagnostic tool and must not be used for clinical decision-making.

## 1. Test Dataset Overview
- **Dataset**: NIH ChestX-ray14 (Pneumonia vs. No Pneumonia)
- **Patient-Level Splitting**: Verified zero patient overlap between train, validation, and test splits.
- **Total Test Images**: 9
- **Positive (Pneumonia) Cases**: 4
- **Negative Cases**: 5
- **Decision Threshold**: 0.5

## 2. Quantitative Performance Metrics

| Metric | Score | Clinical / Technical Interpretation |
| :--- | :--- | :--- |
| **Binary Cross-Entropy** | 0.6317 | Negative log-likelihood of ground truth |
| **Accuracy** | 55.56% | Overall proportion of correct predictions |
| **Precision (PPV)** | 0.5000 | Proportion of predicted pneumonia cases that are true |
| **Recall / Sensitivity** | 0.5000 | Proportion of actual pneumonia cases detected |
| **Specificity** | 0.6000 | Proportion of healthy scans correctly identified |
| **F1-Score** | 0.5000 | Harmonic mean of precision and recall |
| **ROC-AUC** | 0.7500 | Area under the Receiver Operating Characteristic curve |

## 3. Confusion Matrix Breakdown
- **True Negatives (TN)**: 3
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 2

## 4. Visualizations
- Confusion Matrix: `confusion_matrix.png`
- ROC Curve: `roc_curve.png`
- Training History: `training_history.png`

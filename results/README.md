# Health Horizon: Real-Data Test Results

These files document a small end-to-end test using real NIH ChestX-ray14
images for binary Pneumonia vs. No Pneumonia classification.

## Dataset used

- 61 images from 59 unique patients
- 30 Pneumonia cases and 31 No Pneumonia cases
- Patient-level train/validation/test split
- Test partition: 9 images

## Test metrics

| Metric | Result |
| --- | ---: |
| Accuracy | 55.56% |
| Precision | 50.00% |
| Recall | 50.00% |
| Specificity | 60.00% |
| F1-score | 50.00% |
| ROC-AUC | 0.7500 |

This is a pipeline verification run, not a statistically reliable clinical
evaluation. The sample is intentionally small and the model is not a medical
diagnostic tool.

## Files

- `evaluation_report.md` — detailed metric report
- `evaluation_metrics.json` — machine-readable metrics
- `confusion_matrix.png` — classification error summary
- `roc_curve.png` — ROC curve
- `training_history.png` — training and validation history

# Health Horizon: Chest X-ray Classification Pipeline

> **IMPORTANT MEDICAL & REGULATORY DISCLAIMER**  
> Health Horizon is strictly an **educational and research prototype** developed for academic experimentation with transfer learning and deep neural networks on chest radiographs.  
> **It is NOT a medical device, software as a medical device (SaMD), or diagnostic tool.** It has NOT been reviewed, cleared, or approved by the US FDA, EMA, or any national health authority. It must NOT be used for clinical decision-making, patient triage, or disease diagnosis.

---

## 1. Project Overview

**Health Horizon** provides an end-to-end machine learning pipeline built on **TensorFlow/Keras** for chest radiograph classification based on the **NIH ChestX-ray14** dataset schema.

While the primary model addresses binary classification (**Pneumonia vs. No Pneumonia**), the dataset loader, loss formulation, and classification heads are architected modularly so they can easily scale to all **14 thoracic disease labels**:
*Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax.*

### Key Architecture & Pipeline Highlights
1. **Patient-Level Train / Validation / Test Splitting**:
   In NIH ChestX-ray14, individual patients frequently have multiple longitudinal scans. Random image splitting causes severe patient data leakage across splits. Our pipeline splits strictly on `Patient ID`, guaranteeing zero patient overlap between train, validation, and test sets.
2. **Transfer Learning Backbones**:
   Supports lightweight `MobileNetV2` (ideal for rapid training and edge deployment) and `ResNet50`.
3. **Targeted Classification Head**:
   - `GlobalAveragePooling2D`
   - `Dense(128, activation='relu')`
   - `Dropout(0.4)` for regularization
   - Final `Dense(1, activation='sigmoid')` outputting disease probability
4. **Class-Imbalance Handling**:
   Automatically calculates and applies inverse-frequency balanced class weights during training.
5. **Comprehensive Clinical & Technical Metrics**:
   - Binary Cross-Entropy Loss
   - Accuracy
   - Precision (Positive Predictive Value)
   - Recall / Sensitivity
   - Specificity (True Negative Rate)
   - F1-Score
   - ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
6. **Automated Diagnostic Visualizations**:
   - Annotated Confusion Matrix (`confusion_matrix.png`)
   - ROC Curve with AUC score (`roc_curve.png`)
   - Training & Validation Loss/Metric histories (`training_history.png`)
   - Markdown summary report (`evaluation_report.md`) and JSON metrics (`evaluation_metrics.json`)

---

## 2. Repository File Structure

```text
HealthHorizon/
├── data_loader.py          # NIH metadata parsing, patient-level splitting, tf.data pipeline
├── model.py                # Transfer learning backbones (MobileNetV2 / ResNet50) & CNN head
├── train.py                # End-to-end training script with callbacks and class weights
├── evaluate.py             # Evaluation module generating metrics and diagnostic plots
├── predict.py              # CLI prediction tool for single radiographs
├── demo.py                 # Self-contained one-command reviewer demo
├── demo.ipynb              # Jupyter / Colab interactive demonstration notebook
├── results/                # Small real-data test metrics and visualizations
├── requirements.txt        # Python dependency manifest
├── README.md               # Complete project documentation and guide
└── .gitignore              # Ignores venvs, cache, large checkpoints, datasets
```

---

## 3. Installation & Environment Setup

### Prerequisites
- Python 3.9 - 3.11
- pip and virtualenv

### Setup Steps
```bash
# 1. Clone the repository
git clone https://github.com/debjit02-rgb/HeathHorizon.git
cd HeathHorizon

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 4. Quickstart: Reviewer One-Command Demo

For evaluators and reviewers wanting to verify the complete pipeline without manually downloading the full 40+ GB NIH dataset, run:

```bash
python demo.py
```

This single command automatically:
1. Synthesizes a sample dataset conforming to the NIH ChestX-ray14 schema (`data/sample/`).
2. Performs patient-level train, validation, and test splitting (verifying 0% patient leakage).
3. Builds and trains the MobileNetV2 CNN model.
4. Evaluates the model on the held-out test patient partition.
5. Saves the confusion matrix, ROC curve, and training history plots to `outputs/`.
6. Executes sample inferences on test images and displays predictions side-by-side with ground truth.

---

## 5. Usage Guide

### A. Small Real-Data NIH Test

The repository includes a small downloader for a first real-data test. It
downloads metadata and a patient-unique sample of NIH ChestX-ray14 images; it
does not download the complete 40+ GB archive:

```bash
python download_real_sample.py
python train.py \
    --metadata data/real_sample/metadata.csv \
    --image_dir data/real_sample/images \
    --epochs 2 \
    --batch_size 8 \
    --output_model checkpoints/real_sample_best.h5 \
    --output_history outputs/real_sample_history.json
python evaluate.py \
    --model checkpoints/real_sample_best.h5 \
    --test_csv outputs/splits/test.csv \
    --output_dir outputs/real_sample_evaluation
```

This small run is intended to verify the pipeline with real images. Its
metrics are not statistically reliable because the sample is deliberately
small. The downloaded images and generated model files are excluded from Git.
The committed summary and plots are available in [`results/`](results/).

### B. Training the Model (`train.py`)

To train on your own NIH ChestX-ray14 dataset or a custom subset:
```bash
python train.py \
    --metadata path/to/Data_Entry_2017.csv \
    --image_dir path/to/images/ \
    --backbone MobileNetV2 \
    --epochs 10 \
    --batch_size 16 \
    --lr 0.0001 \
    --output_model checkpoints/best_model.h5
```

If you don't have the dataset locally yet, pass `--sample_run`:
```bash
python train.py --sample_run --epochs 3 --batch_size 8
```

### C. Evaluating the Model (`evaluate.py`)

Run quantitative evaluation and generate all visual plots:
```bash
python evaluate.py \
    --model checkpoints/best_model.h5 \
    --test_csv outputs/splits/test.csv \
    --output_dir outputs \
    --threshold 0.5
```

Generated outputs:
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/training_history.png`
- `outputs/evaluation_report.md`
- `outputs/evaluation_metrics.json`

### D. Command-Line Prediction (`predict.py`)

Predict pneumonia probability for an individual chest X-ray image:

```bash
python predict.py --image path/to/image.png
```

Example CLI Output:
```text
============================================================
          Health Horizon - Image Prediction
============================================================
Input Image:           00000003_000.png
Full Path:             /Users/.../data/sample/images/00000003_000.png
Pneumonia Probability: 0.8124 (81.24%)
Classification:        Pneumonia (Confidence: 81.24%)
Decision Threshold:    0.50
Actual Class:          Pneumonia
Prediction Match:      CORRECT
------------------------------------------------------------
DISCLAIMER: Health Horizon is strictly an educational and research prototype.
It is NOT a medical diagnostic tool and must not be used for clinical decision-making.
============================================================
```

To output machine-readable JSON for integration into backend pipelines:
```bash
python predict.py --image path/to/image.png --json
```

---

## 6. Interactive Notebook (`demo.ipynb`)

Launch the notebook to inspect step-by-step visualizations:
```bash
jupyter notebook demo.ipynb
```
The notebook guides through:
1. Environment verification
2. Patient-level data partitioning & leak checks
3. Sample X-ray inspection
4. MobileNetV2 transfer learning model inspection
5. Model training run
6. Evaluation metrics computation
7. Side-by-side image prediction visualization

---

## 7. Extending to All 14 NIH Disease Labels

`data_loader.py` and `model.py` are built with multi-label support:
- `data_loader.py` automatically extracts one-hot columns for all 14 diseases: `Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`, `Edema`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Hernia`.
- In `model.py`, call:
  ```python
  model = build_chest_xray_model(base_model_name="MobileNetV2", num_classes=14)
  ```
  This creates a 14-output sigmoid head with binary cross-entropy loss, naturally handling multi-label thoracic disease co-occurrences.

---

## 8. License & Attribution

- Built as an open educational and research codebase.
- Dataset attribution: NIH Clinical Center ChestX-ray14 dataset (Wang et al., CVPR 2017).

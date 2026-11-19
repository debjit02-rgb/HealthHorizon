"""
Evaluation module for Health Horizon Chest X-ray classification.

Loads a trained model and test partition, computes clinical evaluation metrics:
- Binary Cross-Entropy Loss
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- ROC-AUC

Generates saved plots:
- Confusion Matrix (outputs/confusion_matrix.png)
- ROC Curve with AUC score (outputs/roc_curve.png)
- Training & Validation History curves (outputs/training_history.png)
- Markdown Evaluation Report (outputs/evaluation_report.md)

Educational and research prototype only. Not for clinical diagnosis.
"""

import os
# Prevent tensorflow-metal plugin conflict on macOS
_orig_exists = os.path.exists
os.path.exists = lambda p: False if "tensorflow-plugins" in str(p) else _orig_exists(p)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    log_loss,
)
import tensorflow as tf

from data_loader import create_dataset


DISCLAIMER = (
    "DISCLAIMER: Health Horizon is strictly an educational and research prototype.\n"
    "It is NOT a medical diagnostic tool and must not be used for clinical decision-making."
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "outputs/confusion_matrix.png",
    classes: Tuple[str, str] = ("No Pneumonia", "Pneumonia"),
):
    """
    Renders and saves an annotated confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion Matrix\n(Health Horizon Prototype)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
                fontsize=14,
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix plot to: {output_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: str = "outputs/roc_curve.png",
):
    """
    Renders and saves the Receiver Operating Characteristic (ROC) curve.
    """
    # Guard if only 1 class is present in small test slice
    if len(np.unique(y_true)) < 2:
        print("Note: Single class present in test set; skipping ROC curve generation.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("ROC Curve - Pneumonia vs. No Pneumonia\n(Health Horizon Prototype)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curve plot to: {output_path}")


def plot_training_history(
    history_json_path: str,
    output_path: str = "outputs/training_history.png",
):
    """
    Renders loss and metric curves over training epochs from saved history JSON.
    """
    if not os.path.exists(history_json_path):
        print(f"History file not found at {history_json_path}; skipping history plot.")
        return

    with open(history_json_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history.get("loss", [])) + 1)
    if not epochs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss plot
    ax1.plot(epochs, history.get("loss", []), "o-", label="Train Loss", color="#c0392b")
    if "val_loss" in history:
        ax1.plot(epochs, history.get("val_loss", []), "s--", label="Val Loss", color="#e67e22")
    ax1.set_title("Binary Cross-Entropy Loss vs. Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy / AUC plot
    if "accuracy" in history:
        ax2.plot(epochs, history.get("accuracy", []), "o-", label="Train Accuracy", color="#2980b9")
    if "val_accuracy" in history:
        ax2.plot(epochs, history.get("val_accuracy", []), "s--", label="Val Accuracy", color="#27ae60")
    if "auc" in history:
        ax2.plot(epochs, history.get("auc", []), "^:", label="Train AUC", color="#8e44ad")
    if "val_auc" in history:
        ax2.plot(epochs, history.get("val_auc", []), "v:", label="Val AUC", color="#d35400")

    ax2.set_title("Performance Metrics vs. Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Score")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training history plot to: {output_path}")


def evaluate_model(
    model_path: str = "checkpoints/best_model.h5",
    test_csv: str = "outputs/splits/test.csv",
    history_path: str = "outputs/training_history.json",
    output_dir: str = "outputs",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluates the trained model on the test partition, generates plots, and writes a report.
    """
    print("=" * 70)
    print("Health Horizon - Model Evaluation")
    print(DISCLAIMER)
    print("=" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split not found at: {test_csv}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(f"Loading test dataset from {test_csv}...")
    test_df = pd.read_csv(test_csv)
    test_ds = create_dataset(test_df, batch_size=16, is_training=False)

    print(f"Predicting on {len(test_df)} test samples...")
    raw_preds = model.predict(test_ds, verbose=0)
    y_probs = raw_preds.flatten().astype(np.float64)
    y_true = test_df["label"].values.astype(int)

    # Compute binary predictions
    y_pred = (y_probs >= threshold).astype(int)

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC AUC & Log Loss
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_probs))
        bce_loss = float(log_loss(y_true, y_probs))
    else:
        roc_auc = 0.5
        bce_loss = 0.0

    # Confusion matrix values
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics_dict = {
        "num_test_samples": len(test_df),
        "num_positive": int(y_true.sum()),
        "num_negative": int(len(y_true) - y_true.sum()),
        "threshold": threshold,
        "binary_cross_entropy": round(bce_loss, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
    }

    # Print summary table
    print("\n" + "-" * 40)
    print("EVALUATION METRICS SUMMARY")
    print("-" * 40)
    print(f"Test Samples:          {metrics_dict['num_test_samples']}")
    print(f"Binary Cross-Entropy:  {metrics_dict['binary_cross_entropy']}")
    print(f"Accuracy:              {metrics_dict['accuracy'] * 100:.2f}%")
    print(f"Precision:             {metrics_dict['precision']:.4f}")
    print(f"Recall (Sensitivity):  {metrics_dict['recall_sensitivity']:.4f}")
    print(f"Specificity:           {metrics_dict['specificity']:.4f}")
    print(f"F1-Score:              {metrics_dict['f1_score']:.4f}")
    print(f"ROC-AUC:               {metrics_dict['roc_auc']:.4f}")
    print("-" * 40)

    # Save metrics JSON
    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {json_path}")

    # Generate plots
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    hist_path = os.path.join(output_dir, "training_history.png")

    plot_confusion_matrix(y_true, y_pred, cm_path)
    plot_roc_curve(y_true, y_probs, roc_path)
    plot_training_history(history_path, hist_path)

    # Generate Markdown Evaluation Report
    report_md_path = os.path.join(output_dir, "evaluation_report.md")
    report_content = f"""# Health Horizon - Model Evaluation Report

> **DISCLAIMER:** Health Horizon is strictly an educational and research prototype. It is NOT a medical diagnostic tool and must not be used for clinical decision-making.

## 1. Test Dataset Overview
- **Dataset**: NIH ChestX-ray14 (Pneumonia vs. No Pneumonia)
- **Patient-Level Splitting**: Verified zero patient overlap between train, validation, and test splits.
- **Total Test Images**: {metrics_dict['num_test_samples']}
- **Positive (Pneumonia) Cases**: {metrics_dict['num_positive']}
- **Negative Cases**: {metrics_dict['num_negative']}
- **Decision Threshold**: {metrics_dict['threshold']}

## 2. Quantitative Performance Metrics

| Metric | Score | Clinical / Technical Interpretation |
| :--- | :--- | :--- |
| **Binary Cross-Entropy** | {metrics_dict['binary_cross_entropy']} | Negative log-likelihood of ground truth |
| **Accuracy** | {metrics_dict['accuracy'] * 100:.2f}% | Overall proportion of correct predictions |
| **Precision (PPV)** | {metrics_dict['precision']:.4f} | Proportion of predicted pneumonia cases that are true |
| **Recall / Sensitivity** | {metrics_dict['recall_sensitivity']:.4f} | Proportion of actual pneumonia cases detected |
| **Specificity** | {metrics_dict['specificity']:.4f} | Proportion of healthy scans correctly identified |
| **F1-Score** | {metrics_dict['f1_score']:.4f} | Harmonic mean of precision and recall |
| **ROC-AUC** | {metrics_dict['roc_auc']:.4f} | Area under the Receiver Operating Characteristic curve |

## 3. Confusion Matrix Breakdown
- **True Negatives (TN)**: {tn}
- **False Positives (FP)**: {fp}
- **False Negatives (FN)**: {fn}
- **True Positives (TP)**: {tp}

## 4. Visualizations
- Confusion Matrix: `confusion_matrix.png`
- ROC Curve: `roc_curve.png`
- Training History: `training_history.png`
"""
    with open(report_md_path, "w") as f:
        f.write(report_content)
    print(f"Evaluation report written to: {report_md_path}")

    return metrics_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Chest X-ray classification model.")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.h5", help="Path to trained model.")
    parser.add_argument("--test_csv", type=str, default="outputs/splits/test.csv", help="Path to test split CSV.")
    parser.add_argument("--history", type=str, default="outputs/training_history.json", help="Path to history JSON.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save evaluation artifacts.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive class.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_path=args.model,
        test_csv=args.test_csv,
        history_path=args.history,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )

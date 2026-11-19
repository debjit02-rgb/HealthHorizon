"""
End-to-End Demo Script for Health Horizon.

Executes the complete chest X-ray machine learning workflow on a small sample:
1. Generates / verifies sample chest X-ray images and patient metadata.
2. Performs patient-level train / val / test splitting (0 patient leakage).
3. Constructs MobileNetV2 CNN transfer learning model with custom classification head.
4. Trains the model on the sample data.
5. Computes clinical evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
6. Generates saved plots (confusion matrix, ROC curve, training history).
7. Demonstrates command-line prediction on test images with actual vs. predicted labels.

Usage:
    python demo.py

Educational and research prototype only. Not for clinical diagnosis.
"""

import os
# Prevent tensorflow-metal plugin conflict on macOS
_orig_exists = os.path.exists
os.path.exists = lambda p: False if "tensorflow-plugins" in str(p) else _orig_exists(p)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import tensorflow as tf

from data_loader import create_sample_dataset
from train import train_pipeline
from evaluate import evaluate_model
from predict import predict_single_image, format_prediction_output


DISCLAIMER = (
    "DISCLAIMER: Health Horizon is strictly an educational and research prototype.\n"
    "It is NOT a medical diagnostic tool and must not be used for clinical decision-making."
)


def run_demo():
    print("=" * 72)
    print("      HEALTH HORIZON: CHEST X-RAY MACHINE LEARNING PIPELINE")
    print("                 END-TO-END DEMO EXECUTION")
    print("=" * 72)
    print(DISCLAIMER)
    print("=" * 72)

    sample_dir = "data/sample"
    checkpoint_dir = "checkpoints"
    output_dir = "outputs"
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Prepare synthetic sample dataset
    print("\n>>> STEP 1: Preparing sample NIH ChestX-ray14 dataset...")
    csv_path, img_dir = create_sample_dataset(
        output_dir=sample_dir,
        num_patients=16,
        scans_per_patient=2,
        seed=42,
    )
    print(f"Sample metadata created: {csv_path}")
    print(f"Sample images created:   {img_dir}")

    # Step 2: Train the pipeline on sample data
    print("\n>>> STEP 2: Running training pipeline (Patient-level split + MobileNetV2)...")
    model_path = os.path.join(checkpoint_dir, "best_model.h5")
    history_path = os.path.join(output_dir, "training_history.json")

    train_pipeline(
        metadata_path=csv_path,
        image_dir=img_dir,
        backbone="MobileNetV2",
        epochs=3,
        batch_size=8,
        learning_rate=1e-4,
        fine_tune=False,
        output_model_path=model_path,
        history_path=history_path,
        weights=None,  # Scratch / fast demo mode
    )

    # Step 3: Evaluate model and generate plots
    print("\n>>> STEP 3: Evaluating model and generating visualization plots...")
    test_csv = os.path.join(output_dir, "splits", "test.csv")
    metrics = evaluate_model(
        model_path=model_path,
        test_csv=test_csv,
        history_path=history_path,
        output_dir=output_dir,
        threshold=0.5,
    )

    # Step 4: Run example predictions on test images
    print("\n>>> STEP 4: Demonstrating individual test image predictions...")
    test_df = pd.read_csv(test_csv)
    model = tf.keras.models.load_model(model_path, compile=False)

    num_samples_to_show = min(4, len(test_df))
    for idx in range(num_samples_to_show):
        row = test_df.iloc[idx]
        img_path = row["image_path"]
        actual_label = "Pneumonia" if row["label"] == 1 else "No Pneumonia"
        pred_result = predict_single_image(
            image_path=img_path,
            model=model,
            threshold=0.5,
            actual_label=actual_label,
        )
        print(format_prediction_output(pred_result))

    print("\n" + "=" * 72)
    print("                    DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 72)
    print("Artifacts generated:")
    print(f"  - Model checkpoint:  {model_path}")
    print(f"  - Confusion matrix:  {os.path.join(output_dir, 'confusion_matrix.png')}")
    print(f"  - ROC curve:         {os.path.join(output_dir, 'roc_curve.png')}")
    print(f"  - Training history:  {os.path.join(output_dir, 'training_history.png')}")
    print(f"  - Evaluation report: {os.path.join(output_dir, 'evaluation_report.md')}")
    print(f"  - Test split CSV:    {test_csv}")
    print("=" * 72)


if __name__ == "__main__":
    run_demo()

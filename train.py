"""
Training script for Health Horizon Chest X-ray classification.

Orchestrates data loading with patient-level splitting, model creation,
class-imbalance weighting, model training with callbacks, and artifact saving.

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
from typing import Dict, Any, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from data_loader import (
    load_and_preprocess_metadata,
    split_data_by_patient,
    create_dataset,
    create_sample_dataset,
    TARGET_LABEL,
)
from model import build_chest_xray_model, unfreeze_for_fine_tuning


DISCLAIMER = (
    "DISCLAIMER: Health Horizon is strictly an educational and research prototype.\n"
    "It is NOT a medical diagnostic tool and must not be used for clinical decision-making."
)


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Computes balanced class weights to address chest X-ray disease class imbalance.
    """
    total = len(labels)
    num_pos = np.sum(labels == 1)
    num_neg = total - num_pos

    if num_pos == 0 or num_neg == 0:
        return {0: 1.0, 1: 1.0}

    weight_for_0 = (1.0 / num_neg) * (total / 2.0)
    weight_for_1 = (1.0 / num_pos) * (total / 2.0)
    return {0: float(weight_for_0), 1: float(weight_for_1)}


def train_pipeline(
    metadata_path: str,
    image_dir: str,
    backbone: str = "MobileNetV2",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    fine_tune: bool = False,
    fine_tune_epochs: int = 3,
    output_model_path: str = "checkpoints/best_model.h5",
    history_path: str = "outputs/training_history.json",
    splits_dir: str = "outputs/splits",
    weights: Optional[str] = "imagenet",
) -> Dict[str, Any]:
    """
    Runs the complete training routine and saves model and history artifacts.
    """
    print("=" * 70)
    print("Health Horizon - Training Pipeline")
    print(DISCLAIMER)
    print("=" * 70)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    # 1. Load and parse metadata
    print(f"\n[1/5] Loading metadata from {metadata_path}...")
    df = load_and_preprocess_metadata(
        csv_path=metadata_path,
        target_disease=TARGET_LABEL,
        image_dir=image_dir,
    )
    print(f"Total available images: {len(df)}")
    print(f"Pneumonia positive cases: {int(df['label'].sum())} ({df['label'].mean()*100:.2f}%)")

    # 2. Patient-level split
    print("\n[2/5] Performing patient-level train / val / test split...")
    train_df, val_df, test_df = split_data_by_patient(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )
    print(f"Train split: {len(train_df)} images ({train_df['patient_id'].nunique()} patients)")
    print(f"Val split:   {len(val_df)} images ({val_df['patient_id'].nunique()} patients)")
    print(f"Test split:  {len(test_df)} images ({test_df['patient_id'].nunique()} patients)")

    # Save splits for consistent downstream evaluation and reproducibility
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    print(f"Saved partition splits to {splits_dir}/")

    # Compute class weights from training partition
    class_weights = compute_class_weights(train_df["label"].values)
    print(f"Class weights (0: No Pneumonia, 1: Pneumonia): {class_weights}")

    # 3. Create tf.data datasets
    print("\n[3/5] Building tf.data input pipelines...")
    train_ds = create_dataset(train_df, batch_size=batch_size, is_training=True)
    val_ds = create_dataset(val_df, batch_size=batch_size, is_training=False)

    # 4. Build model
    print(f"\n[4/5] Building {backbone} transfer learning model...")
    model = build_chest_xray_model(
        base_model_name=backbone,
        input_shape=(224, 224, 3),
        num_classes=1,
        dense_units=128,
        dropout_rate=0.4,
        learning_rate=learning_rate,
        freeze_base=True,
        weights=weights,
    )
    model.summary(line_length=80)

    # Setup callbacks
    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=output_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 5. Train initial stage
    print(f"\n[5/5] Training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1,
    )

    combined_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # Optional fine-tuning phase
    if fine_tune:
        print("\nStarting fine-tuning phase (unfreezing top backbone layers)...")
        model = unfreeze_for_fine_tuning(model, unfreeze_layers=20, fine_tune_lr=learning_rate * 0.1)
        ft_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            class_weight=class_weights,
            callbacks=cb_list,
            verbose=1,
        )
        for k, vals in ft_history.history.items():
            if k in combined_history:
                combined_history[k].extend([float(v) for v in vals])
            else:
                combined_history[k] = [float(v) for v in vals]

    # Save best model to ensure file exists even if early stopped
    if not os.path.exists(output_model_path):
        model.save(output_model_path)

    # Save training history as JSON
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"\nTraining completed! Saved history to {history_path}")
    print(f"Best model saved to {output_model_path}")

    return combined_history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Health Horizon Chest X-ray Classification CNN."
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/sample/sample_metadata.csv",
        help="Path to NIH ChestX-ray14 metadata CSV.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/sample/images",
        help="Directory containing X-ray images.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="MobileNetV2",
        choices=["MobileNetV2", "ResNet50"],
        help="Transfer learning backbone.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Whether to execute a second fine-tuning stage.",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="checkpoints/best_model.h5",
        help="Destination path for best saved model.",
    )
    parser.add_argument(
        "--output_history",
        type=str,
        default="outputs/training_history.json",
        help="Destination path for training history JSON.",
    )
    parser.add_argument(
        "--sample_run",
        action="store_true",
        help="Automatically generate synthetic sample dataset if data does not exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # If sample_run requested or files missing, generate sample dataset
    if args.sample_run or not (os.path.exists(args.metadata) and os.path.exists(args.image_dir)):
        print("Data files not found or --sample_run flag passed. Creating sample dataset...")
        csv_p, img_d = create_sample_dataset("data/sample", num_patients=16, scans_per_patient=2)
        args.metadata = csv_p
        args.image_dir = img_d

    train_pipeline(
        metadata_path=args.metadata,
        image_dir=args.image_dir,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fine_tune=args.fine_tune,
        output_model_path=args.output_model,
        history_path=args.output_history,
        weights=None if args.sample_run else "imagenet",
    )

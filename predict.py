"""
Command-line inference script for Health Horizon Chest X-ray classification.

Accepts an input image, runs inference through the trained CNN model,
and outputs:
- Input image name
- Predicted probability of Pneumonia
- Predicted class ('Pneumonia' vs 'No Pneumonia')
- Actual class (if provided via --actual or resolved from metadata)

Example usage:
    python predict.py --image data/sample/images/00000001_000.png

Educational and research prototype only. Not for clinical diagnosis.
"""

import os
# Prevent tensorflow-metal plugin conflict on macOS
_orig_exists = os.path.exists
os.path.exists = lambda p: False if "tensorflow-plugins" in str(p) else _orig_exists(p)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf


DISCLAIMER = (
    "DISCLAIMER: Health Horizon is strictly an educational and research prototype.\n"
    "It is NOT a medical diagnostic tool and must not be used for clinical decision-making."
)


def load_and_preprocess_single_image(
    image_path: str,
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Loads an image from disk, converts to RGB, resizes to target_size,
    normalizes pixels to [0, 1], and adds a batch dimension.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Use PIL to safely open standard PNG, JPEG, TIFF
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Add batch dimension: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def predict_single_image(
    image_path: str,
    model: tf.keras.Model,
    threshold: float = 0.5,
    actual_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs model inference on an image and returns formatted prediction details.
    """
    img_batch = load_and_preprocess_single_image(image_path)
    prob_pred = model.predict(img_batch, verbose=0)[0][0]
    prob_pred = float(prob_pred)

    predicted_class = "Pneumonia" if prob_pred >= threshold else "No Pneumonia"
    confidence = prob_pred if prob_pred >= threshold else (1.0 - prob_pred)

    result = {
        "image_name": os.path.basename(image_path),
        "image_path": os.path.abspath(image_path),
        "predicted_probability": round(prob_pred, 4),
        "confidence": round(confidence * 100, 2),
        "predicted_class": predicted_class,
        "threshold": threshold,
    }

    if actual_label is not None:
        result["actual_class"] = actual_label
        result["is_correct"] = bool(predicted_class.lower() == actual_label.lower())

    return result


def format_prediction_output(result: Dict[str, Any]) -> str:
    """
    Formats the prediction result into a readable report string.
    """
    lines = [
        "",
        "============================================================",
        "          Health Horizon - Image Prediction",
        "============================================================",
        f"Input Image:           {result['image_name']}",
        f"Full Path:             {result['image_path']}",
        f"Pneumonia Probability: {result['predicted_probability']:.4f} ({result['predicted_probability']*100:.2f}%)",
        f"Classification:        {result['predicted_class']} (Confidence: {result['confidence']:.2f}%)",
        f"Decision Threshold:    {result['threshold']:.2f}",
    ]

    if "actual_class" in result:
        lines.append(f"Actual Class:          {result['actual_class']}")
        verdict = "CORRECT" if result.get("is_correct") else "INCORRECT"
        lines.append(f"Prediction Match:      {verdict}")

    lines.append("------------------------------------------------------------")
    lines.append(DISCLAIMER)
    lines.append("============================================================")
    return "\n".join(lines)


def lookup_actual_label_from_metadata(
    image_name: str,
    metadata_csv: Optional[str] = None,
) -> Optional[str]:
    """
    Attempts to lookup ground truth label from metadata CSV if available.
    """
    if metadata_csv is None or not os.path.exists(metadata_csv):
        return None

    try:
        df = pd.read_csv(metadata_csv)
        col_img = "Image Index" if "Image Index" in df.columns else "image_index"
        col_lbl = "Finding Labels" if "Finding Labels" in df.columns else "finding_labels"

        match = df[df[col_img] == image_name]
        if not match.empty:
            finding = str(match[col_lbl].values[0])
            if "Pneumonia" in finding.split("|"):
                return "Pneumonia"
            elif finding.strip() == "No Finding":
                return "No Pneumonia"
            else:
                return f"No Pneumonia ({finding})"
    except Exception:
        pass
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Chest X-ray classification inference with Health Horizon."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the chest radiograph image to predict.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best_model.h5",
        help="Path to trained model (.h5 or SavedModel).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification decision threshold (default: 0.5).",
    )
    parser.add_argument(
        "--actual",
        type=str,
        default=None,
        help="Actual ground truth class (e.g. 'Pneumonia' or 'No Pneumonia').",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/sample/sample_metadata.csv",
        help="Optional path to metadata CSV to auto-lookup ground truth.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print prediction as machine-readable JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model file '{args.model}' not found. Please train first via 'python train.py' "
            "or run 'python demo.py'."
        )

    # Load model
    model = tf.keras.models.load_model(args.model, compile=False)

    # Resolve ground truth label
    actual = args.actual
    if actual is None and args.metadata:
        actual = lookup_actual_label_from_metadata(os.path.basename(args.image), args.metadata)

    # Run inference
    result = predict_single_image(
        image_path=args.image,
        model=model,
        threshold=args.threshold,
        actual_label=actual,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_prediction_output(result))

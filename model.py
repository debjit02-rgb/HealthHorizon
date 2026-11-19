"""
Model architecture module for Health Horizon.

Defines transfer learning architectures using MobileNetV2 or ResNet50
with custom classification head:
- GlobalAveragePooling2D
- Dense(128, activation='relu')
- Dropout
- Dense(num_classes, activation='sigmoid')

Compiled with Binary Cross-Entropy loss and accuracy, precision, recall, and ROC-AUC metrics.
Educational and research prototype only. Not for clinical diagnosis.
"""

import os
# Prevent tensorflow-metal plugin conflict on macOS
_orig_exists = os.path.exists
os.path.exists = lambda p: False if "tensorflow-plugins" in str(p) else _orig_exists(p)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics

# Use legacy Adam on macOS Apple Silicon if available to avoid M1/M2 optimizer slowdown
try:
    AdamOptimizer = tf.keras.optimizers.legacy.Adam
except AttributeError:
    AdamOptimizer = tf.keras.optimizers.Adam



def build_chest_xray_model(
    base_model_name: str = "MobileNetV2",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    dense_units: int = 128,
    dropout_rate: float = 0.4,
    learning_rate: float = 1e-4,
    freeze_base: bool = True,
    weights: Optional[str] = "imagenet",
) -> tf.keras.Model:
    """
    Constructs and compiles the Chest X-ray classification model.

    Args:
        base_model_name: "MobileNetV2" (default, lightweight) or "ResNet50".
        input_shape: Input image dimensions (H, W, C).
        num_classes: 1 for binary (Pneumonia vs No Pneumonia); 14 for multi-label.
        dense_units: Neurons in intermediate dense layer (specified as 128).
        dropout_rate: Dropout rate for regularization.
        learning_rate: Initial Adam learning rate.
        freeze_base: Whether to freeze base backbone weights.
        weights: "imagenet" or None (for testing/scratch training).

    Returns:
        Compiled tf.keras.Model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Select transfer learning backbone
    base_name = base_model_name.lower().replace("_", "")
    if "mobilenet" in base_name:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
        )
    elif "resnet" in base_name:
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
        )
    else:
        raise ValueError(f"Unsupported backbone '{base_model_name}'. Choose 'MobileNetV2' or 'ResNet50'.")

    base_model.trainable = not freeze_base

    # Feature extraction
    x = base_model(inputs, training=not freeze_base)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_128")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    # Sigmoid activation for binary / multi-label disease probabilities
    outputs = layers.Dense(
        num_classes,
        activation="sigmoid",
        name="pneumonia_probability" if num_classes == 1 else "disease_probabilities",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"HealthHorizon_{base_model_name}")

    # Compile with Binary Cross-Entropy loss and specified clinical metrics
    model.compile(
        optimizer=AdamOptimizer(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(name="binary_crossentropy"),
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(curve="ROC", name="auc"),
        ],
    )

    return model


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    unfreeze_layers: int = 20,
    fine_tune_lr: float = 1e-5,
) -> tf.keras.Model:
    """
    Unfreezes the top N layers of the backbone for fine-tuning at a lower learning rate.
    """
    # Locate backbone layer (typically index 1 in the model)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        return model

    backbone.trainable = True
    # Freeze all except top unfreeze_layers
    for layer in backbone.layers[:-unfreeze_layers]:
        layer.trainable = False

    model.compile(
        optimizer=AdamOptimizer(learning_rate=fine_tune_lr),
        loss=tf.keras.losses.BinaryCrossentropy(name="binary_crossentropy"),
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(curve="ROC", name="auc"),
        ],
    )
    return model

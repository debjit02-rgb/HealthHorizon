"""
Data loader and preprocessing module for Health Horizon.

Handles NIH ChestX-ray14 metadata parsing, patient-level train/validation/test
splitting (preventing patient data leakage across splits), image preprocessing,
and tf.data pipeline creation for binary classification (Pneumonia vs. No Pneumonia)
with modular design to support all 14 NIH disease labels.

Educational and research prototype only. Not for clinical diagnosis.
"""

import os
# Prevent tensorflow-metal plugin conflict on macOS
_orig_exists = os.path.exists
os.path.exists = lambda p: False if "tensorflow-plugins" in str(p) else _orig_exists(p)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Standard 14 disease labels in the NIH ChestX-ray14 dataset
NIH_14_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

TARGET_LABEL = "Pneumonia"


def parse_patient_id(image_index: str) -> str:
    """
    Extracts the patient ID from standard NIH ChestX-ray filename format.
    NIH format: '<patient_id>_<scan_number>.png' (e.g., '00000001_000.png' -> '00000001').
    """
    if "_" in str(image_index):
        return str(image_index).split("_")[0]
    return str(image_index)


def load_and_preprocess_metadata(
    csv_path: str,
    target_disease: str = TARGET_LABEL,
    image_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads NIH metadata CSV, ensures patient identifiers are present,
    creates binary target label column (1 if target disease is in Finding Labels, 0 otherwise),
    and filters to only images existing on disk if image_dir is provided.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize column names if needed
    col_mapping = {
        "Image Index": "image_index",
        "Finding Labels": "finding_labels",
        "Patient ID": "patient_id",
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

    if "image_index" not in df.columns:
        raise ValueError("Metadata CSV must contain an 'Image Index' or 'image_index' column.")

    if "finding_labels" not in df.columns:
        raise ValueError("Metadata CSV must contain a 'Finding Labels' or 'finding_labels' column.")

    # Ensure patient_id column exists; extract from filename if missing
    if "patient_id" not in df.columns:
        df["patient_id"] = df["image_index"].apply(parse_patient_id)
    else:
        df["patient_id"] = df["patient_id"].astype(str)

    # Binary label for target disease
    df["label"] = df["finding_labels"].apply(
        lambda labels: 1 if target_disease in str(labels).split("|") else 0
    )

    # One-hot/multi-label encoding for all 14 labels to support future expansion
    for disease in NIH_14_LABELS:
        col_name = f"disease_{disease.lower()}"
        df[col_name] = df["finding_labels"].apply(
            lambda labels: 1 if disease in str(labels).split("|") else 0
        )

    # If image_dir is specified, verify file existence and attach full path
    if image_dir is not None:
        df["image_path"] = df["image_index"].apply(lambda img: os.path.join(image_dir, str(img)))
        initial_len = len(df)
        df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
        if len(df) < initial_len:
            print(f"Filtered {initial_len - len(df)} rows whose image files were not found in {image_dir}.")

    return df


def split_data_by_patient(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into train, validation, and test sets at the PATIENT level.
    Ensures no patient appears in more than one partition, eliminating data leakage.
    """
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = (
        train_ratio / total,
        val_ratio / total,
        test_ratio / total,
    )

    unique_patients = df["patient_id"].unique()

    # Stratify by patient's positive status if possible, otherwise simple random split
    patient_has_positive = (
        df.groupby("patient_id")["label"].max().reindex(unique_patients).values
    )

    try:
        train_patients, test_val_patients = train_test_split(
            unique_patients,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=patient_has_positive,
        )
    except ValueError:
        # Fallback if too few samples in positive class for stratified split
        train_patients, test_val_patients = train_test_split(
            unique_patients,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
        )

    val_prop_of_remaining = val_ratio / (val_ratio + test_ratio)
    test_val_labels = (
        df.groupby("patient_id")["label"].max().reindex(test_val_patients).values
    )

    try:
        val_patients, test_patients = train_test_split(
            test_val_patients,
            train_size=val_prop_of_remaining,
            random_state=random_state,
            stratify=test_val_labels,
        )
    except ValueError:
        val_patients, test_patients = train_test_split(
            test_val_patients,
            train_size=val_prop_of_remaining,
            random_state=random_state,
        )

    train_df = df[df["patient_id"].isin(set(train_patients))].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(set(val_patients))].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(set(test_patients))].reset_index(drop=True)

    # Sanity check: verify 0 patient overlap
    train_set = set(train_df["patient_id"])
    val_set = set(val_df["patient_id"])
    test_set = set(test_df["patient_id"])
    assert len(train_set.intersection(val_set)) == 0, "Patient overlap between train and val!"
    assert len(train_set.intersection(test_set)) == 0, "Patient overlap between train and test!"
    assert len(val_set.intersection(test_set)) == 0, "Patient overlap between val and test!"

    return train_df, val_df, test_df


def load_and_preprocess_image(
    image_path: tf.Tensor,
    img_size: Tuple[int, int] = (224, 224),
) -> tf.Tensor:
    """
    Loads an image file, decodes it as RGB (NIH X-rays are grayscale or RGB),
    resizes it to target dimensions, and scales pixels to [0, 1].
    """
    img_raw = tf.io.read_file(image_path)
    # Decode image as 3-channel RGB for pretrained models (MobileNetV2 / ResNet50)
    image = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """
    Data augmentation for training chest X-rays.
    Applies subtle random horizontal flip, brightness, and contrast adjustments
    while preserving anatomical validity (no vertical flip for chest radiographs).
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def create_dataset(
    df: pd.DataFrame,
    image_col: str = "image_path",
    label_col: str = "label",
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    is_training: bool = False,
    shuffle_buffer: int = 256,
) -> tf.data.Dataset:
    """
    Builds an optimized tf.data.Dataset pipeline from dataframe paths and labels.
    """
    image_paths = df[image_col].values.astype(str)
    labels = df[label_col].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(df), shuffle_buffer))

    def _process_train(path, label):
        img = load_and_preprocess_image(path, img_size)
        img = augment_image(img)
        return img, label

    def _process_eval(path, label):
        img = load_and_preprocess_image(path, img_size)
        return img, label

    map_fn = _process_train if is_training else _process_eval
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def create_sample_dataset(
    output_dir: str = "data/sample",
    num_patients: int = 12,
    scans_per_patient: int = 2,
    img_size: Tuple[int, int] = (224, 224),
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Creates a synthetic sample dataset and metadata CSV adhering to NIH ChestX-ray14
    format for rapid testing and self-contained demonstrations.
    Returns (csv_path, images_dir).
    """
    from PIL import Image

    np.random.seed(seed)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    records: List[Dict[str, Any]] = []

    for p_idx in range(1, num_patients + 1):
        patient_id = f"{p_idx:08d}"
        # ~35% of patients have pneumonia findings
        patient_has_pneumonia = (p_idx % 3 == 0)

        for scan_idx in range(scans_per_patient):
            image_name = f"{patient_id}_{scan_idx:03d}.png"
            image_path = os.path.join(images_dir, image_name)

            if not os.path.exists(image_path):
                # Generate synthetic chest radiograph-like gradient pattern
                base = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
                y, x = np.ogrid[: img_size[0], : img_size[1]]
                # Lung field simulation (two oval lobes)
                cx1, cy1 = img_size[1] * 0.35, img_size[0] * 0.5
                cx2, cy2 = img_size[1] * 0.65, img_size[0] * 0.5
                lobe1 = ((x - cx1) ** 2 / (40**2) + (y - cy1) ** 2 / (65**2)) < 1.0
                lobe2 = ((x - cx2) ** 2 / (40**2) + (y - cy2) ** 2 / (65**2)) < 1.0
                base[lobe1 | lobe2] = 160

                if patient_has_pneumonia:
                    # Infiltrate patch
                    infiltrate = ((x - cx1) ** 2 / (20**2) + (y - cy1) ** 2 / (20**2)) < 1.0
                    base[infiltrate] = 230

                # Add noise
                noise = np.random.randint(0, 30, size=img_size, dtype=np.uint8)
                synthetic = np.clip(base + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(synthetic).convert("RGB")
                img.save(image_path)

            finding = "Pneumonia" if patient_has_pneumonia else "No Finding"
            records.append({
                "Image Index": image_name,
                "Finding Labels": finding,
                "Follow-up #": scan_idx,
                "Patient ID": patient_id,
                "Patient Age": 40 + p_idx,
                "Patient Gender": "M" if p_idx % 2 == 0 else "F",
                "View Position": "PA",
            })

    csv_path = os.path.join(output_dir, "sample_metadata.csv")
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return csv_path, images_dir

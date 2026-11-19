"""Download a small, patient-unique NIH ChestX-ray14 sample for testing.

This intentionally downloads only a small real-data sample. It is not a
replacement for the complete NIH ChestX-ray14 download.
"""

from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd


METADATA_URL = (
    "https://raw.githubusercontent.com/mrdchry/NIH-CHEST-XRAY/master/"
    "Data_Entry_2017.csv"
)
IMAGE_URL = (
    "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/"
    "images_1024x1024/{image_name}"
)

OUTPUT_DIR = Path("data/real_sample")
IMAGE_DIR = OUTPUT_DIR / "images"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
SAMPLES_PER_CLASS = 30


def download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "HealthHorizon/1.0"})
    with urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    source_csv = OUTPUT_DIR / "Data_Entry_2017.csv"
    if not source_csv.exists():
        print("Downloading NIH metadata...")
        download(METADATA_URL, source_csv)

    df = pd.read_csv(source_csv)
    df["patient_id"] = df["Patient ID"].astype(str)
    df["label"] = df["Finding Labels"].apply(
        lambda value: int("Pneumonia" in str(value).split("|"))
    )

    # One image per patient keeps this first test free from patient duplication.
    positive = df[df["label"] == 1].drop_duplicates("patient_id")
    negative = df[df["label"] == 0].drop_duplicates("patient_id")
    selected = pd.concat(
        [positive.head(SAMPLES_PER_CLASS), negative.head(SAMPLES_PER_CLASS)]
    ).sample(frac=1, random_state=42)

    downloaded_rows = []
    for _, row in selected.iterrows():
        image_name = row["Image Index"]
        destination = IMAGE_DIR / image_name
        if not destination.exists():
            print(f"Downloading {image_name}...")
            download(IMAGE_URL.format(image_name=image_name), destination)
        downloaded_rows.append(row)

    sample = pd.DataFrame(downloaded_rows).drop(
        columns=["patient_id", "label"], errors="ignore"
    )
    sample.to_csv(METADATA_PATH, index=False)
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved images:   {IMAGE_DIR}")
    print(f"Images: {len(sample)} | Patients: {sample['patient_id'].nunique()}")
    print(f"Pneumonia: {int(sample['label'].sum())} | No Pneumonia: {int((sample['label'] == 0).sum())}")


if __name__ == "__main__":
    main()

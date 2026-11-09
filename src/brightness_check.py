"""Measure brightness statistics for NIH ChestX-ray14 images.

The analysis records per-image intensity statistics and exports extreme
Train/Val examples for manual review. No images are removed or modified.
Results are written under ``reports/brightness/``.
"""


from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm


# ==========================================
# 1. Configuration
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images-224"

CSV_PATH = DATA_DIR / "Data_Entry_2017.csv"
TRAIN_VAL_LIST = DATA_DIR / "train_val_list_NIH.txt"
TEST_LIST = DATA_DIR / "test_list_NIH.txt"

REPORT_DIR = PROJECT_DIR / "reports" / "brightness"


# ==========================================
# 2. Histogram quantile
# ==========================================

def get_histogram_quantile(histogram, quantile, offset=0):
    """Return the intensity value at the requested histogram quantile."""

    cumulative = np.cumsum(histogram)
    target = max(1, int(np.ceil(histogram.sum() * quantile)))

    return int(np.searchsorted(cumulative, target) + offset)


# ==========================================
# 3. Calculate image brightness statistics
# ==========================================

def calculate_brightness(image_path):
    """Calculate global and non-black pixel brightness statistics."""

    with Image.open(image_path) as image:

        grayscale = image.convert("L")

        array = np.asarray(
            grayscale,
            dtype=np.uint8,
        )

    histogram = np.bincount(
        array.ravel(),
        minlength=256,
    )

    pixel_count = array.size

    intensity_values = np.arange(256)

    mean = (
        intensity_values @ histogram
    ) / pixel_count

    variance = (
        ((intensity_values - mean) ** 2) @ histogram
    ) / pixel_count

    std = np.sqrt(variance)

    p01 = get_histogram_quantile(histogram, 0.01)
    p50 = get_histogram_quantile(histogram, 0.50)
    p99 = get_histogram_quantile(histogram, 0.99)

    # Pixel values <= 5 are treated as near-black.
    dark_ratio = histogram[:6].sum() / pixel_count

    # Pixel values >= 250 are treated as near-white.
    bright_ratio = histogram[250:].sum() / pixel_count

    # Calculate statistics excluding near-black pixels.
    # This helps reduce the influence of black background.
    nonblack_histogram = histogram[6:]
    nonblack_count = nonblack_histogram.sum()

    if nonblack_count > 0:

        nonblack_mean = (
            np.arange(6, 256) @ nonblack_histogram
        ) / nonblack_count

        nonblack_p50 = get_histogram_quantile(
            nonblack_histogram,
            0.50,
            offset=6,
        )

    else:

        nonblack_mean = 0.0
        nonblack_p50 = 0

    return {
        "Mean": float(mean),
        "Std": float(std),
        "P01": p01,
        "P50": p50,
        "P99": p99,
        "Dark_Ratio": float(dark_ratio),
        "Bright_Ratio": float(bright_ratio),
        "Nonblack_Mean": float(nonblack_mean),
        "Nonblack_P50": int(nonblack_p50),
    }


# ==========================================
# 4. Main
# ==========================================

def main():
    """Analyze all images and save brightness statistics and extremes."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading metadata...")

    metadata = pd.read_csv(
        CSV_PATH,
        usecols=["Image Index", "Patient ID"],
    )

    metadata["Patient ID"] = (
        metadata["Patient ID"].astype(str)
    )

    with open(TRAIN_VAL_LIST, "r") as f:
        train_val_images = {
            line.strip()
            for line in f
            if line.strip()
        }

    with open(TEST_LIST, "r") as f:
        test_images = {
            line.strip()
            for line in f
            if line.strip()
        }

    def get_split(filename):

        if filename in train_val_images:
            return "Train_Val"

        if filename in test_images:
            return "Test"

        raise ValueError(
            f"Image missing from official split lists: {filename}"
        )

    metadata["Split"] = metadata["Image Index"].apply(
        get_split
    )

    records = []

    print("\nCalculating brightness statistics...")

    for row in tqdm(
        metadata.itertuples(index=False),
        total=len(metadata),
        desc="Brightness check",
    ):

        filename = row[0]
        patient_id = row[1]
        split = row[2]

        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        stats = calculate_brightness(image_path)

        records.append({
            "Image Index": filename,
            "Patient ID": patient_id,
            "Split": split,
            **stats,
        })

    result = pd.DataFrame(records)

    # Save statistics for every image.
    stats_path = (
        REPORT_DIR / "brightness_stats.csv"
    )

    result.to_csv(
        stats_path,
        index=False,
    )

    print(f"\nSaved: {stats_path}")

    # --------------------------------------
    # Print overall distributions
    # --------------------------------------

    metric_columns = [
        "Mean",
        "Std",
        "P01",
        "P50",
        "P99",
        "Dark_Ratio",
        "Bright_Ratio",
        "Nonblack_Mean",
        "Nonblack_P50",
    ]

    quantiles = [
        0.00,
        0.01,
        0.05,
        0.50,
        0.95,
        0.99,
        1.00,
    ]

    for split_name in ["Train_Val", "Test"]:

        subset = result[
            result["Split"] == split_name
        ]

        print(f"\n========== {split_name} ==========")

        print(f"Images: {len(subset):,}")

        summary = subset[
            metric_columns
        ].quantile(quantiles)

        print(
            summary.round(4).to_string()
        )

    # --------------------------------------
    # Save extreme examples for inspection
    # --------------------------------------

    train_val = result[
        result["Split"] == "Train_Val"
    ]

    extremes = pd.concat([
        train_val.nsmallest(30, "Nonblack_Mean"),
        train_val.nlargest(30, "Nonblack_Mean"),
        train_val.nlargest(30, "Bright_Ratio"),
    ]).drop_duplicates(
        subset=["Image Index"]
    )

    extremes_path = (
        REPORT_DIR / "brightness_extremes_trainval.csv"
    )

    extremes.to_csv(
        extremes_path,
        index=False,
    )

    print(f"\nSaved extreme examples: {extremes_path}")

    print("\nBrightness analysis completed.")
    print("No images were removed.")


if __name__ == "__main__":
    main()

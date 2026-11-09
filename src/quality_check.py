"""Validate NIH ChestX-ray14 metadata, official splits, and image integrity.

The script writes a human-readable report and a CSV containing detected issues
under ``reports/data_quality/``. It does not modify or remove source images.
"""


from pathlib import Path
from collections import Counter

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

REPORT_DIR = PROJECT_DIR / "reports" / "data_quality"

EXPECTED_IMAGE_SIZE = (224, 224)

try:
    from .constants import DISEASES
except ImportError:  # Support running scripts as: python src/<script>.py
    from constants import DISEASES

VALID_LABELS = set(DISEASES) | {"No Finding"}

issues = []


# ==========================================
# 2. Issue recording
# ==========================================

def add_issue(severity, category, filename, detail):
    """Append one validation issue to the in-memory issue list."""
    issues.append({
        "Severity": severity,
        "Category": category,
        "Filename": filename,
        "Detail": detail,
    })


# ==========================================
# 3. Check required files
# ==========================================

def check_required_files():
    """Fail early when required NIH metadata, lists, or images are missing."""
    required_paths = [
        IMAGE_DIR,
        CSV_PATH,
        TRAIN_VAL_LIST,
        TEST_LIST,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path does not exist: {path}"
            )


# ==========================================
# 4. Load and validate metadata
# ==========================================

def check_metadata():
    """Load metadata, validate required fields, and derive binary labels."""
    print("\n[1/4] Checking metadata...")

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "Image Index",
        "Finding Labels",
        "Patient ID",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing CSV column: {column}"
            )

    df = df[required_columns].copy()

    # Check missing values
    for column in required_columns:
        missing_count = df[column].isna().sum()

        if missing_count > 0:
            add_issue(
                "ERROR",
                "Missing metadata",
                "",
                f"{column}: {missing_count} missing values",
            )

    # Normalize text columns
    df["Image Index"] = (
        df["Image Index"].fillna("").astype(str).str.strip()
    )

    df["Finding Labels"] = (
        df["Finding Labels"].fillna("").astype(str).str.strip()
    )

    df["Patient ID"] = df["Patient ID"].astype("string")

    # Check duplicate image entries
    duplicate_count = df["Image Index"].duplicated().sum()

    if duplicate_count > 0:
        add_issue(
            "ERROR",
            "Duplicate metadata",
            "",
            f"{duplicate_count} duplicated image records",
        )

    # Check labels
    for row in df.itertuples(index=False):
        filename = row[0]
        label_string = row[1]

        labels = [
            label.strip()
            for label in label_string.split("|")
            if label.strip()
        ]

        unknown_labels = set(labels) - VALID_LABELS

        if not labels:
            add_issue(
                "ERROR",
                "Empty label",
                filename,
                "No disease label was found.",
            )

        if unknown_labels:
            add_issue(
                "ERROR",
                "Unknown label",
                filename,
                f"Unknown labels: {unknown_labels}",
            )

        if "No Finding" in labels and labels != ["No Finding"]:
            add_issue(
                "ERROR",
                "Invalid No Finding label",
                filename,
                f"Unexpected label combination: {labels}",
            )

        if len(labels) != len(set(labels)):
            add_issue(
                "WARNING",
                "Duplicate label",
                filename,
                f"Repeated labels: {labels}",
            )

    # Create 14 binary labels
    for disease in DISEASES:
        df[disease] = df["Finding Labels"].apply(
            lambda x: int(disease in x.split("|"))
        )

    print(f"Metadata records: {len(df):,}")
    print(f"Unique patients: {df['Patient ID'].nunique():,}")

    print("\nDisease positive counts:")

    for disease in DISEASES:
        print(f"  {disease:22s}: {df[disease].sum():,}")

    return df


# ==========================================
# 5. Check official data splits
# ==========================================

def load_image_list(path):
    """Read an official split-list file and return its image filenames."""
    with open(path, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
        ]


def check_official_splits(df):
    """Check that the official Train/Val and Test lists are consistent."""
    print("\n[2/4] Checking official train/test splits...")

    train_val_entries = load_image_list(TRAIN_VAL_LIST)
    test_entries = load_image_list(TEST_LIST)

    train_val_images = set(train_val_entries)
    test_images = set(test_entries)

    metadata_images = set(df["Image Index"])

    # Check duplicate entries in list files
    if len(train_val_entries) != len(train_val_images):
        add_issue(
            "ERROR",
            "Duplicate split entries",
            "",
            "Duplicate entries found in train_val_list_NIH.txt",
        )

    if len(test_entries) != len(test_images):
        add_issue(
            "ERROR",
            "Duplicate split entries",
            "",
            "Duplicate entries found in test_list_NIH.txt",
        )

    # Check overlap
    overlap = train_val_images & test_images

    if overlap:
        add_issue(
            "ERROR",
            "Split overlap",
            "",
            f"{len(overlap)} images overlap between train_val and test",
        )

    # Check image names against metadata
    missing_train_val = train_val_images - metadata_images
    missing_test = test_images - metadata_images

    if missing_train_val:
        add_issue(
            "ERROR",
            "Missing metadata images",
            "",
            f"{len(missing_train_val)} train_val images missing from CSV",
        )

    if missing_test:
        add_issue(
            "ERROR",
            "Missing metadata images",
            "",
            f"{len(missing_test)} test images missing from CSV",
        )

    # Check complete coverage
    listed_images = train_val_images | test_images
    missing_from_lists = metadata_images - listed_images

    if missing_from_lists:
        add_issue(
            "ERROR",
            "Missing split entries",
            "",
            f"{len(missing_from_lists)} metadata images are not in official lists",
        )

    print(f"Train/Validation images: {len(train_val_images):,}")
    print(f"Test images:             {len(test_images):,}")
    print(f"Total listed images:     {len(listed_images):,}")

    return train_val_images, test_images


# ==========================================
# 6. Check image files
# ==========================================

def check_images(metadata_images):
    """Inspect image existence, decodability, dimensions, and formats."""
    print("\n[3/4] Checking image files...")

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
    )

    disk_images = {path.name for path in image_paths}

    # Check missing and extra files
    missing_images = metadata_images - disk_images
    extra_images = disk_images - metadata_images

    for filename in missing_images:
        add_issue(
            "ERROR",
            "Missing image",
            filename,
            "Image file does not exist.",
        )

    for filename in extra_images:
        add_issue(
            "WARNING",
            "Extra image",
            filename,
            "Image exists but is not listed in metadata.",
        )

    print(f"Image files found: {len(image_paths):,}")

    size_counts = Counter()
    mode_counts = Counter()
    format_counts = Counter()

    for path in tqdm(image_paths, desc="Validating images"):
        try:
            with Image.open(path) as img:

                # Read and decode the complete image
                img.load()

                size = img.size
                mode = img.mode
                image_format = img.format

                size_counts[size] += 1
                mode_counts[mode] += 1
                format_counts[image_format] += 1

                # Check image dimensions
                if size != EXPECTED_IMAGE_SIZE:
                    add_issue(
                        "WARNING",
                        "Unexpected image size",
                        path.name,
                        f"Expected {EXPECTED_IMAGE_SIZE}, got {size}",
                    )

                # Check format
                if image_format != "PNG":
                    add_issue(
                        "WARNING",
                        "Unexpected image format",
                        path.name,
                        f"Expected PNG, got {image_format}",
                    )

                # Check suspicious pixel values
                extrema = img.getextrema()

                if isinstance(extrema[0], tuple):
                    channel_extrema = extrema
                else:
                    channel_extrema = [extrema]

                min_value = min(
                    channel[0] for channel in channel_extrema
                )

                max_value = max(
                    channel[1] for channel in channel_extrema
                )

                if max_value == 0:
                    add_issue(
                        "WARNING",
                        "All-black image",
                        path.name,
                        "All pixel values are zero.",
                    )

                elif all(
                    low == high
                    for low, high in channel_extrema
                ):
                    add_issue(
                        "WARNING",
                        "Constant image",
                        path.name,
                        f"All pixels have the same value: {min_value}",
                    )

        except Exception as e:
            add_issue(
                "ERROR",
                "Corrupted image",
                path.name,
                str(e),
            )

    print("\nImage size distribution:")

    for size, count in size_counts.most_common():
        print(f"  {size}: {count:,}")

    print("\nImage mode distribution:")

    for mode, count in mode_counts.most_common():
        print(f"  {mode}: {count:,}")

    print("\nImage format distribution:")

    for image_format, count in format_counts.most_common():
        print(f"  {image_format}: {count:,}")

    return len(image_paths)


# ==========================================
# 7. Save quality report
# ==========================================

def save_report(df, image_count, train_val_count, test_count):
    """Write the quality summary and detailed issue CSV to the report folder."""
    print("\n[4/4] Saving quality report...")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    issue_df = pd.DataFrame(
        issues,
        columns=["Severity", "Category", "Filename", "Detail"]
    )

    issue_path = REPORT_DIR / "image_issues.csv"
    issue_df.to_csv(issue_path, index=False)

    error_count = sum(
        issue["Severity"] == "ERROR"
        for issue in issues
    )

    warning_count = sum(
        issue["Severity"] == "WARNING"
        for issue in issues
    )

    status = "PASS" if error_count == 0 else "NEEDS REVIEW"

    report_lines = [
        "========== NIH ChestX-ray14 Data Quality Report ==========",
        "",
        f"Status: {status}",
        "",
        f"Metadata records: {len(df):,}",
        f"Image files: {image_count:,}",
        f"Train/Validation list: {train_val_count:,}",
        f"Test list: {test_count:,}",
        "",
        f"Total errors: {error_count}",
        f"Total warnings: {warning_count}",
        "",
        "Disease positive counts:",
    ]

    for disease in DISEASES:
        report_lines.append(
            f"{disease}: {df[disease].sum():,}"
        )

    report_lines.extend([
        "",
        "Issue categories:",
    ])

    if issue_df.empty:
        report_lines.append("No issues found.")
    else:
        category_counts = (
            issue_df.groupby(["Severity", "Category"])
            .size()
            .reset_index(name="Count")
        )

        for row in category_counts.itertuples(index=False):
            report_lines.append(
                f"{row.Severity} | {row.Category}: {row.Count}"
            )

    report_text = "\n".join(report_lines)

    report_path = REPORT_DIR / "data_quality_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)

    print(f"\nSaved report: {report_path}")
    print(f"Saved issues: {issue_path}")

    return error_count


# ==========================================
# 8. Main
# ==========================================

def main():
    """Run all metadata, split, and image-integrity checks."""
    check_required_files()

    df = check_metadata()

    train_val_images, test_images = check_official_splits(df)

    image_count = check_images(
        set(df["Image Index"])
    )

    error_count = save_report(
        df,
        image_count,
        len(train_val_images),
        len(test_images),
    )

    if error_count > 0:
        print("\nData quality check finished with errors.")
        raise SystemExit(1)

    print("\nData quality check completed successfully!")


if __name__ == "__main__":
    main()

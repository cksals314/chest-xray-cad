"""Dataset and image preprocessing for 14-label NIH ChestX-ray14 training.

Images are loaded as grayscale, optionally CLAHE-enhanced, converted to RGB,
then converted to tensors and normalized with ImageNet statistics. No data
augmentation is applied in the project training pipeline.
"""


from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ==========================================
# 1. Configuration
# ==========================================

try:
    from .constants import DISEASES
except ImportError:  # Support running scripts as: python src/<script>.py
    from constants import DISEASES

# ImageNet normalization values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==========================================
# 2. NIH ChestX-ray14 Dataset
# ==========================================

class CXR14Dataset(Dataset):
    """Load NIH CXR14 images and their 14-dimensional multi-label targets."""

    def __init__(
        self,
        csv_path,
        image_dir,
        use_clahe=False,
        clahe_clip_limit=2.0,
        clahe_grid_size=(8, 8),
    ):
        """
        Args:
            csv_path:
                Path to train.csv, val.csv, or test.csv.

            image_dir:
                Path to the images-224 directory.

            use_clahe:
                True: Apply CLAHE.
                False: Use original grayscale image.

            clahe_clip_limit:
                CLAHE contrast limiting parameter.

            clahe_grid_size:
                CLAHE tile grid size.
        """

        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)

        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

        # Load metadata.
        self.df = pd.read_csv(self.csv_path)

        required_columns = [
            "Image Index",
            "Patient ID",
            *DISEASES,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in CSV: {missing_columns}"
            )

        # Store image filenames and labels.
        self.image_names = (
            self.df["Image Index"].astype(str).tolist()
        )

        self.labels = self.df[DISEASES].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(self.labels).all():
            raise ValueError("Invalid values found in labels.")

        if not np.isin(self.labels, [0.0, 1.0]).all():
            raise ValueError("Labels must contain only 0 and 1.")

        # Create CLAHE lazily inside each DataLoader worker.
        self.clahe = None

        # Tensor conversion and ImageNet normalization.
        # No data augmentation is applied.
        self.transform = transforms.Compose([
            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ])

    def __len__(self):
        return len(self.image_names)

    def _get_clahe(self):
        if self.clahe is None:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size,
            )

        return self.clahe

    def __getitem__(self, index):

        # Get image path.
        image_name = self.image_names[index]
        image_path = self.image_dir / image_name

        # Load image and convert to grayscale.
        # LA images are safe to convert because their alpha
        # channels were verified to be fully opaque.
        with Image.open(image_path) as img:
            grayscale = img.convert("L")

        # Convert PIL image to NumPy array.
        image_array = np.array(grayscale, dtype=np.uint8)

        # Apply CLAHE if enabled.
        if self.use_clahe:
            image_array = self._get_clahe().apply(
                image_array
            )

        # Convert grayscale image to RGB.
        image = Image.fromarray(
            image_array,
            mode="L",
        ).convert("RGB")

        # Convert to Tensor and normalize.
        image_tensor = self.transform(image)

        # Load 14 binary labels.
        label_tensor = torch.from_numpy(
            self.labels[index].copy()
        )

        return image_tensor, label_tensor

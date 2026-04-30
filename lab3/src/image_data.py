"""Utilities for loading the prepared image splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from lab3_utils import project_lab_root


DEFAULT_IMAGE_SIZE = 96
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)


def default_image_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Return a simple image transform for the baseline CNN."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )


class ImageSplitDataset(Dataset):
    """Dataset backed by the prepared image split CSV file."""

    def __init__(self, csv_path: Path, transform: transforms.Compose | None = None) -> None:
        self.dataframe = pd.read_csv(csv_path)
        self.transform = transform or default_image_transform()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[object, int]:
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image_tensor = self.transform(image)
        label = int(row["label"])
        return image_tensor, label


def prepared_image_paths() -> dict[str, Path]:
    """Return the processed image split file paths."""
    lab_root = project_lab_root(__file__)
    processed_dir = lab_root / "data" / "processed"
    return {
        "train": processed_dir / "image_train.csv",
        "validation": processed_dir / "image_validation.csv",
        "test": processed_dir / "image_test.csv",
    }


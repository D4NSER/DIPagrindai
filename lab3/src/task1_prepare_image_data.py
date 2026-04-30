"""Prepare rock-paper-scissors image metadata and train/validation/test splits."""

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from lab3_utils import ensure_directory, project_lab_root


RANDOM_SEED = 2026
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10
IMAGE_SIZE = {"width": 300, "height": 200}
CLASS_TO_LABEL = {"paper": 0, "rock": 1, "scissors": 2}


def collect_image_rows(images_dir: Path) -> list[dict[str, object]]:
    """Collect one metadata row for every image file."""
    rows: list[dict[str, object]] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = images_dir / class_name
        image_paths = sorted(class_dir.glob("*.png"))
        for image_path in image_paths:
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "file_name": image_path.name,
                    "class_name": class_name,
                    "label": label,
                }
            )
    return rows


def split_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split metadata into train/validation/test with stratification."""
    train_df, temp_df = train_test_split(
        dataframe,
        test_size=VALIDATION_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=dataframe["label"],
    )
    validation_share_of_temp = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO)
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - validation_share_of_temp,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    validation_df["split"] = "validation"
    test_df["split"] = "test"
    return train_df, validation_df, test_df


def class_distribution(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return count per class name."""
    counts = dataframe["class_name"].value_counts().sort_index()
    return {str(class_name): int(count) for class_name, count in counts.items()}


def main() -> None:
    """Prepare image metadata and save the split files."""
    random.seed(RANDOM_SEED)
    lab_root = project_lab_root(__file__)
    images_dir = lab_root / "images"
    processed_dir = ensure_directory(lab_root / "data" / "processed")

    rows = collect_image_rows(images_dir)
    if not rows:
        raise FileNotFoundError(f"No PNG images found in {images_dir}")

    all_df = pd.DataFrame(rows)
    all_df = all_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    train_df, validation_df, test_df = split_dataframe(all_df)
    train_df.to_csv(processed_dir / "image_train.csv", index=False, encoding="utf-8")
    validation_df.to_csv(
        processed_dir / "image_validation.csv", index=False, encoding="utf-8"
    )
    test_df.to_csv(processed_dir / "image_test.csv", index=False, encoding="utf-8")

    print("Task 1 completed: image dataset prepared.")
    print(f"Total samples: {len(train_df) + len(validation_df) + len(test_df)}")
    print(f"Train: {len(train_df)}, validation: {len(validation_df)}, test: {len(test_df)}")
    print(f"Classes: {class_distribution(pd.concat([train_df, validation_df, test_df], ignore_index=True))}")


if __name__ == "__main__":
    main()

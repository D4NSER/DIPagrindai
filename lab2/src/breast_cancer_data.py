"""Shared data-loading and preparation utilities for Lab 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np


RAW_COLUMNS = (
    "sample_code_number",
    "clump_thickness",
    "uniformity_of_cell_size",
    "uniformity_of_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class",
)

FEATURE_COLUMNS = RAW_COLUMNS[1:-1]
LABEL_MAP = {"2": 0, "4": 1}


@dataclass(frozen=True)
class PreparedDataset:
    """Prepared dataset without ID column and with labels mapped to 0/1."""

    features: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def size(self) -> int:
        """Return number of rows in the dataset."""
        return int(self.labels.shape[0])

    @property
    def feature_count(self) -> int:
        """Return number of usable input features."""
        return int(self.features.shape[1])


@dataclass(frozen=True)
class DatasetSplit:
    """One named subset of the prepared dataset."""

    name: str
    features: np.ndarray
    labels: np.ndarray

    @property
    def size(self) -> int:
        """Return number of rows in the split."""
        return int(self.labels.shape[0])


def project_lab_root(current_file: Path) -> Path:
    """Resolve the `lab2` root directory from a script path."""
    return current_file.resolve().parents[1]


def load_raw_rows(raw_path: Path) -> list[dict[str, str]]:
    """Load the raw UCI CSV rows into a list of dictionaries."""
    rows: list[dict[str, str]] = []
    with raw_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for raw_row in reader:
            if not raw_row:
                continue
            rows.append(dict(zip(RAW_COLUMNS, raw_row)))
    return rows


def prepare_dataset(raw_path: Path, seed: int) -> tuple[PreparedDataset, dict[str, object]]:
    """Clean, convert, and shuffle the raw dataset."""
    raw_rows = load_raw_rows(raw_path)
    cleaned_features: list[list[float]] = []
    cleaned_labels: list[int] = []

    removed_missing = 0
    raw_class_counts = {"2": 0, "4": 0}

    for row in raw_rows:
        raw_label = row["class"]
        if raw_label in raw_class_counts:
            raw_class_counts[raw_label] += 1

        if "?" in row.values():
            removed_missing += 1
            continue

        cleaned_features.append([float(row[column]) for column in FEATURE_COLUMNS])
        cleaned_labels.append(LABEL_MAP[raw_label])

    feature_array = np.asarray(cleaned_features, dtype=float)
    label_array = np.asarray(cleaned_labels, dtype=np.int64)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(label_array.shape[0])
    feature_array = feature_array[permutation]
    label_array = label_array[permutation]

    prepared = PreparedDataset(
        features=feature_array,
        labels=label_array,
        feature_names=FEATURE_COLUMNS,
    )
    metadata = {
        "source_file": str(raw_path),
        "seed": seed,
        "raw_row_count": len(raw_rows),
        "clean_row_count": prepared.size,
        "feature_count": prepared.feature_count,
        "removed_missing_rows": removed_missing,
        "dropped_columns": ["sample_code_number"],
        "label_mapping": {"2": 0, "4": 1},
        "raw_class_counts": raw_class_counts,
        "clean_class_counts": {
            "0": int(np.sum(label_array == 0)),
            "1": int(np.sum(label_array == 1)),
        },
        "feature_names": list(FEATURE_COLUMNS),
    }
    return prepared, metadata


def split_dataset(
    dataset: PreparedDataset,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Split the prepared dataset into train, validation, and test subsets."""
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Ratios must be positive and leave a non-empty test split.")

    total = dataset.size
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    return (
        DatasetSplit(
            name="train",
            features=dataset.features[:train_end],
            labels=dataset.labels[:train_end],
        ),
        DatasetSplit(
            name="validation",
            features=dataset.features[train_end:validation_end],
            labels=dataset.labels[train_end:validation_end],
        ),
        DatasetSplit(
            name="test",
            features=dataset.features[validation_end:],
            labels=dataset.labels[validation_end:],
        ),
    )


def save_prepared_dataset_csv(dataset: PreparedDataset, destination: Path) -> None:
    """Save the cleaned and shuffled dataset to CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([*dataset.feature_names, "label"])
        for feature_row, label in zip(dataset.features, dataset.labels):
            writer.writerow([*(int(value) for value in feature_row), int(label)])


def save_json(payload: dict[str, object], destination: Path) -> None:
    """Serialize metadata JSON with UTF-8 encoding."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

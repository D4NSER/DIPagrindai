"""Task 1: download-independent preparation of the Wisconsin breast cancer dataset."""

from __future__ import annotations

from pathlib import Path

from breast_cancer_data import (
    prepare_dataset,
    project_lab_root,
    save_json,
    save_prepared_dataset_csv,
    split_dataset,
)


RANDOM_SEED = 2026


def main() -> None:
    """Clean the raw dataset, shuffle rows, and save processed outputs."""
    lab_root = project_lab_root(Path(__file__))
    raw_path = lab_root / "data" / "raw" / "breast-cancer-wisconsin.data"
    processed_csv_path = lab_root / "data" / "processed" / "breast-cancer-wisconsin-clean.csv"
    metadata_path = lab_root / "data" / "processed" / "task1_data_preparation_metadata.json"

    dataset, metadata = prepare_dataset(raw_path=raw_path, seed=RANDOM_SEED)
    train_split, validation_split, test_split = split_dataset(dataset)

    metadata["default_split_sizes"] = {
        "train": train_split.size,
        "validation": validation_split.size,
        "test": test_split.size,
    }

    save_prepared_dataset_csv(dataset, processed_csv_path)
    save_json(metadata, metadata_path)

    print("Task 1 completed: dataset prepared successfully.")
    print(f"Processed rows: {metadata['clean_row_count']} of {metadata['raw_row_count']}")
    print(f"Removed rows with missing values: {metadata['removed_missing_rows']}")
    print(f"Class counts after mapping: {metadata['clean_class_counts']}")
    print(f"Default 80:10:10 split sizes: {metadata['default_split_sizes']}")
    print(f"Processed CSV: {processed_csv_path}")
    print(f"Metadata JSON: {metadata_path}")


if __name__ == "__main__":
    main()

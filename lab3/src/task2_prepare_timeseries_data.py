"""Prepare CMU keystroke data for a 30-class Conv1D task."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lab3_utils import ensure_directory, project_lab_root


RANDOM_SEED = 2026
SELECTED_SESSION = 1
SELECTED_SUBJECT_COUNT = 30
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10


def select_subjects(session_df: pd.DataFrame) -> list[str]:
    """Pick 30 subjects with stable ordering for repeatability."""
    subjects = sorted(session_df["subject"].unique())
    if len(subjects) < SELECTED_SUBJECT_COUNT:
        raise ValueError(
            f"Need at least {SELECTED_SUBJECT_COUNT} subjects, found only {len(subjects)}."
        )
    return subjects[:SELECTED_SUBJECT_COUNT]


def split_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split records into train/validation/test while keeping classes balanced."""
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


def main() -> None:
    """Prepare one-session, 30-user classification data."""
    lab_root = project_lab_root(__file__)
    source_path = lab_root / "DSL-StrongPasswordData.csv"
    processed_dir = ensure_directory(lab_root / "data" / "processed")

    dataframe = pd.read_csv(source_path)
    feature_columns = list(dataframe.columns[3:])

    session_df = dataframe[dataframe["sessionIndex"] == SELECTED_SESSION].copy()
    selected_subjects = select_subjects(session_df)
    filtered_df = session_df[session_df["subject"].isin(selected_subjects)].copy()

    label_map = {subject: index for index, subject in enumerate(selected_subjects)}
    filtered_df["label"] = filtered_df["subject"].map(label_map)

    ordered_columns = ["subject", "sessionIndex", "rep", "label", *feature_columns]
    filtered_df = filtered_df[ordered_columns]
    filtered_df = filtered_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    train_df, validation_df, test_df = split_dataframe(filtered_df)
    train_df.to_csv(processed_dir / "timeseries_train.csv", index=False, encoding="utf-8")
    validation_df.to_csv(
        processed_dir / "timeseries_validation.csv", index=False, encoding="utf-8"
    )
    test_df.to_csv(processed_dir / "timeseries_test.csv", index=False, encoding="utf-8")

    print("Task 2 completed: timeseries dataset prepared.")
    print(f"Selected session: {SELECTED_SESSION}")
    print(f"Selected subjects: {len(selected_subjects)}")
    print(f"Total samples: {len(train_df) + len(validation_df) + len(test_df)}")
    print(f"Train: {len(train_df)}, validation: {len(validation_df)}, test: {len(test_df)}")
    print(f"Feature count: {len(feature_columns)}")


if __name__ == "__main__":
    main()

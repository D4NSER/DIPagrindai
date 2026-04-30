"""Utilities for loading prepared CMU keystroke splits for Conv1D models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from lab3_utils import project_lab_root


def prepared_timeseries_paths() -> dict[str, Path]:
    """Return processed timeseries split file paths."""
    lab_root = project_lab_root(__file__)
    processed_dir = lab_root / "data" / "processed"
    return {
        "train": processed_dir / "timeseries_train.csv",
        "validation": processed_dir / "timeseries_validation.csv",
        "test": processed_dir / "timeseries_test.csv",
    }


def load_timeseries_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load split CSV files as dataframes."""
    paths = prepared_timeseries_paths()
    return (
        pd.read_csv(paths["train"]),
        pd.read_csv(paths["validation"]),
        pd.read_csv(paths["test"]),
    )


def standardize_features(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize splits using only train statistics."""
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    return (
        (train_x - mean) / std,
        (validation_x - mean) / std,
        (test_x - mean) / std,
    )


def load_timeseries_tensors() -> dict[str, TensorDataset]:
    """Load standardized splits as TensorDataset objects for Conv1D."""
    train_df, validation_df, test_df = load_timeseries_frames()
    excluded_columns = {"subject", "sessionIndex", "rep", "label", "split"}
    feature_columns = [column for column in train_df.columns if column not in excluded_columns]

    train_x = train_df[feature_columns].to_numpy(dtype=np.float32)
    validation_x = validation_df[feature_columns].to_numpy(dtype=np.float32)
    test_x = test_df[feature_columns].to_numpy(dtype=np.float32)
    train_x, validation_x, test_x = standardize_features(train_x, validation_x, test_x)

    train_y = train_df["label"].to_numpy(dtype=np.int64)
    validation_y = validation_df["label"].to_numpy(dtype=np.int64)
    test_y = test_df["label"].to_numpy(dtype=np.int64)

    return {
        "train": TensorDataset(
            torch.tensor(train_x).unsqueeze(1),
            torch.tensor(train_y),
        ),
        "validation": TensorDataset(
            torch.tensor(validation_x).unsqueeze(1),
            torch.tensor(validation_y),
        ),
        "test": TensorDataset(
            torch.tensor(test_x).unsqueeze(1),
            torch.tensor(test_y),
        ),
    }

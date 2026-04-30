"""Run the required time-series study with simpler result outputs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from cnn1d_model import ConvNet1D
from lab3_utils import ensure_directory, project_lab_root
from timeseries_data import load_timeseries_frames, load_timeseries_tensors


RANDOM_SEED = 2026
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
INPUT_LENGTH = 31
ARCHITECTURE_EPOCHS = 15
STUDY_EPOCHS = 10
LEARNING_RATES = [0.0005, 0.001, 0.005]

TRAIN_FRAME, VALIDATION_FRAME, TEST_FRAME = load_timeseries_frames()
SUBJECT_NAMES = sorted(TRAIN_FRAME["subject"].unique())
NUM_CLASSES = len(SUBJECT_NAMES)

ARCHITECTURES = {
    "t1_small": {
        "conv_channels": [16],
        "dense_units": 64,
        "dropout_rate": 0.0,
        "use_batch_norm": False,
        "activation_name": "relu",
    },
    "t2_medium": {
        "conv_channels": [16, 32],
        "dense_units": 128,
        "dropout_rate": 0.0,
        "use_batch_norm": False,
        "activation_name": "relu",
    },
    "t3_dropout_bn": {
        "conv_channels": [16, 32, 64],
        "dense_units": 128,
        "dropout_rate": 0.25,
        "use_batch_norm": True,
        "activation_name": "relu",
    },
}

BASE_STUDY_CONFIG = {
    "conv_channels": [16, 32],
    "dense_units": 128,
    "dropout_rate": 0.0,
    "use_batch_norm": False,
    "activation_name": "relu",
}

STUDIES = {
    "dropout": [
        {"name": "dropout_0_0", "config_updates": {"dropout_rate": 0.0}, "optimizer": "adam"},
        {"name": "dropout_0_25", "config_updates": {"dropout_rate": 0.25}, "optimizer": "adam"},
        {"name": "dropout_0_5", "config_updates": {"dropout_rate": 0.5}, "optimizer": "adam"},
    ],
    "batch_norm": [
        {"name": "batch_norm_off", "config_updates": {"use_batch_norm": False}, "optimizer": "adam"},
        {"name": "batch_norm_on", "config_updates": {"use_batch_norm": True}, "optimizer": "adam"},
    ],
    "activation": [
        {"name": "relu", "config_updates": {"activation_name": "relu"}, "optimizer": "adam"},
        {"name": "tanh", "config_updates": {"activation_name": "tanh"}, "optimizer": "adam"},
        {
            "name": "leaky_relu",
            "config_updates": {"activation_name": "leaky_relu"},
            "optimizer": "adam",
        },
    ],
    "optimizer": [
        {"name": "adam", "config_updates": {}, "optimizer": "adam"},
        {"name": "sgd", "config_updates": {}, "optimizer": "sgd"},
    ],
    "learning_rate": [
        {"name": "lr_0_0005", "config_updates": {}, "optimizer": "adam", "learning_rate": 0.0005},
        {"name": "lr_0_001", "config_updates": {}, "optimizer": "adam", "learning_rate": 0.001},
        {"name": "lr_0_005", "config_updates": {}, "optimizer": "adam", "learning_rate": 0.005},
    ],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(batch_size: int) -> dict[str, DataLoader]:
    datasets = load_timeseries_tensors()
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }


def create_optimizer(name: str, model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    normalized = name.lower()
    if normalized == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if normalized == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if is_training:
            optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        if is_training:
            loss.backward()
            optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def evaluate_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_size
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def train_run(
    run_name: str,
    model_config: dict[str, object],
    loaders: dict[str, DataLoader],
    device: torch.device,
    learning_rate: float,
    epochs: int,
    optimizer_name: str,
    evaluate_test: bool,
) -> dict[str, object]:
    model = ConvNet1D(
        input_length=INPUT_LENGTH,
        num_classes=NUM_CLASSES,
        conv_channels=model_config["conv_channels"],
        dense_units=model_config["dense_units"],
        dropout_rate=model_config["dropout_rate"],
        use_batch_norm=model_config["use_batch_norm"],
        activation_name=model_config["activation_name"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(optimizer_name, model, learning_rate)

    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }
    best_state = None
    best_epoch = 1
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")

    start_time = time.perf_counter()
    for epoch in range(epochs):
        train_loss, train_accuracy = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation_loss, validation_accuracy = run_epoch(
            model, loaders["validation"], criterion, device
        )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)

        is_better = (
            validation_accuracy > best_validation_accuracy
            or (
                validation_accuracy == best_validation_accuracy
                and validation_loss < best_validation_loss
            )
        )
        if is_better:
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss

    training_time_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)

    result = {
        "run_name": run_name,
        "model_config": model_config,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_loss": best_validation_loss,
        "best_train_accuracy": max(history["train_accuracy"]),
        "best_train_loss": min(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_validation_loss": history["validation_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_validation_accuracy": history["validation_accuracy"][-1],
        "training_time_seconds": training_time_seconds,
        "history": history,
    }

    if evaluate_test:
        test_result = evaluate_with_predictions(model, loaders["test"], criterion, device)
        result["test_accuracy"] = test_result["accuracy"]
        result["test_loss"] = test_result["loss"]
        result["test_result"] = test_result

    return result


def plot_training_curves(destination: Path, history: dict[str, list[float]], title: str) -> None:
    ensure_directory(destination.parent)
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["validation_loss"], label="Validation loss")
    axes[0].set_title("Loss by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="Train accuracy")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation accuracy")
    axes[1].set_title("Accuracy by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def plot_study_summary(destination: Path, title: str, rows: list[dict[str, object]]) -> None:
    ensure_directory(destination.parent)
    labels = [str(row["run_name"]) for row in rows]
    x_positions = list(range(len(labels)))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x_positions, [row["best_validation_accuracy"] for row in rows], color="#31a354")
    axes[0].set_title("Best validation accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x_positions, labels, rotation=20)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[1].bar(x_positions, [row["best_validation_loss"] for row in rows], color="#756bb1")
    axes[1].set_title("Best validation loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_xticks(x_positions, labels, rotation=20)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def plot_confusion_matrix(destination: Path, labels: list[int], predictions: list[int]) -> None:
    ensure_directory(destination.parent)
    matrix = confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES)))
    figure, axis = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=SUBJECT_NAMES,
        yticklabels=SUBJECT_NAMES,
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Confusion matrix")
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def save_summary_csv(destination: Path, rows: list[dict[str, object]], include_test: bool) -> None:
    ensure_directory(destination.parent)
    columns = [
        "run_name",
        "best_epoch",
        "best_validation_accuracy",
        "best_validation_loss",
        "best_train_accuracy",
        "best_train_loss",
        "final_train_loss",
        "final_validation_loss",
        "final_train_accuracy",
        "final_validation_accuracy",
        "training_time_seconds",
    ]
    if include_test:
        columns.extend(["test_accuracy", "test_loss"])
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        for row in rows:
            values = [
                row["run_name"],
                row["best_epoch"],
                f"{row['best_validation_accuracy']:.8f}",
                f"{row['best_validation_loss']:.8f}",
                f"{row['best_train_accuracy']:.8f}",
                f"{row['best_train_loss']:.8f}",
                f"{row['final_train_loss']:.8f}",
                f"{row['final_validation_loss']:.8f}",
                f"{row['final_train_accuracy']:.8f}",
                f"{row['final_validation_accuracy']:.8f}",
                f"{row['training_time_seconds']:.8f}",
            ]
            if include_test:
                values.extend([f"{row['test_accuracy']:.8f}", f"{row['test_loss']:.8f}"])
            writer.writerow(values)


def save_history_csv(destination: Path, history: dict[str, list[float]]) -> None:
    ensure_directory(destination.parent)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["epoch", "train_loss", "validation_loss", "train_accuracy", "validation_accuracy"]
        )
        for epoch_index in range(len(history["train_loss"])):
            writer.writerow(
                [
                    epoch_index + 1,
                    f"{history['train_loss'][epoch_index]:.8f}",
                    f"{history['validation_loss'][epoch_index]:.8f}",
                    f"{history['train_accuracy'][epoch_index]:.8f}",
                    f"{history['validation_accuracy'][epoch_index]:.8f}",
                ]
            )


def save_test_predictions(destination: Path, predictions: list[int], labels: list[int]) -> None:
    ensure_directory(destination.parent)
    test_metadata = TEST_FRAME.reset_index(drop=True)
    rows = []
    for index, (prediction, label) in enumerate(zip(predictions, labels), start=1):
        metadata_row = test_metadata.iloc[index - 1]
        rows.append(
            {
                "index": index,
                "subject": metadata_row["subject"],
                "rep": int(metadata_row["rep"]),
                "predicted_label": prediction,
                "predicted_class": SUBJECT_NAMES[prediction],
                "true_label": label,
                "true_class": SUBJECT_NAMES[label],
            }
        )
    pd.DataFrame(rows).to_csv(destination, index=False, encoding="utf-8")


def save_selected_examples(source_csv: Path, destination: Path) -> None:
    ensure_directory(destination.parent)
    dataframe = pd.read_csv(source_csv)
    selected_parts = []
    for subject_name in SUBJECT_NAMES:
        selected_parts.append(dataframe[dataframe["true_class"] == subject_name].head(1))
    pd.concat(selected_parts, ignore_index=True).to_csv(destination, index=False, encoding="utf-8")


def save_best_run_outputs(results_dir: Path, best_run: dict[str, object]) -> None:
    best_dir = ensure_directory(results_dir / "final_best_timeseries_model")
    save_history_csv(best_dir / "epoch_metrics.csv", best_run["history"])
    plot_training_curves(
        best_dir / "training_curves.png",
        best_run["history"],
        title=f"Best timeseries model: {best_run['run_name']}",
    )
    plot_confusion_matrix(
        best_dir / "confusion_matrix.png",
        best_run["test_result"]["labels"],
        best_run["test_result"]["predictions"],
    )
    save_test_predictions(
        best_dir / "test_predictions.csv",
        best_run["test_result"]["predictions"],
        best_run["test_result"]["labels"],
    )
    save_selected_examples(
        best_dir / "test_predictions.csv",
        best_dir / "selected_30_examples.csv",
    )
    with (best_dir / "summary.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["field", "value"])
        writer.writerow(["run_name", best_run["run_name"]])
        writer.writerow(["optimizer", best_run["optimizer"]])
        writer.writerow(["learning_rate", best_run["learning_rate"]])
        writer.writerow(["conv_channels", str(best_run["model_config"]["conv_channels"])])
        writer.writerow(["dense_units", best_run["model_config"]["dense_units"]])
        writer.writerow(["dropout_rate", best_run["model_config"]["dropout_rate"]])
        writer.writerow(["use_batch_norm", best_run["model_config"]["use_batch_norm"]])
        writer.writerow(["activation_name", best_run["model_config"]["activation_name"]])
        writer.writerow(["best_epoch", best_run["best_epoch"]])
        writer.writerow(["best_train_accuracy", f"{best_run['best_train_accuracy']:.8f}"])
        writer.writerow(["best_train_loss", f"{best_run['best_train_loss']:.8f}"])
        writer.writerow(["best_validation_accuracy", f"{best_run['best_validation_accuracy']:.8f}"])
        writer.writerow(["best_validation_loss", f"{best_run['best_validation_loss']:.8f}"])
        writer.writerow(["test_accuracy", f"{best_run['test_accuracy']:.8f}"])
        writer.writerow(["test_loss", f"{best_run['test_loss']:.8f}"])


def is_better_training_run(candidate: dict[str, object], current_best: dict[str, object] | None) -> bool:
    """Pick the best run by training metrics, then use validation as tie-breaker."""
    if current_best is None:
        return True
    candidate_key = (
        candidate["best_train_accuracy"],
        -candidate["best_train_loss"],
        candidate["best_validation_accuracy"],
        -candidate["best_validation_loss"],
    )
    current_key = (
        current_best["best_train_accuracy"],
        -current_best["best_train_loss"],
        current_best["best_validation_accuracy"],
        -current_best["best_validation_loss"],
    )
    return candidate_key > current_key


def main() -> None:
    set_seed(RANDOM_SEED)
    lab_root = project_lab_root(__file__)
    results_dir = ensure_directory(lab_root / "results" / "task4_timeseries")
    loaders = build_loaders(DEFAULT_BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    architecture_rows = []
    all_experiment_rows = []
    for run_name, config in ARCHITECTURES.items():
        print(f"Training timeseries architecture: {run_name}")
        result = train_run(
            run_name=run_name,
            model_config=config,
            loaders=loaders,
            device=device,
            learning_rate=DEFAULT_LEARNING_RATE,
            epochs=ARCHITECTURE_EPOCHS,
            optimizer_name="adam",
            evaluate_test=False,
        )
        architecture_rows.append(result)
        all_experiment_rows.append(result)

    save_summary_csv(results_dir / "architectures.csv", architecture_rows, include_test=False)
    plot_study_summary(results_dir / "architectures.png", "Timeseries architectures", architecture_rows)

    for study_name, variants in STUDIES.items():
        study_rows = []
        for variant in variants:
            run_name = str(variant["name"])
            config = dict(BASE_STUDY_CONFIG)
            config.update(variant["config_updates"])
            print(f"Training timeseries study: {study_name} -> {run_name}")
            result = train_run(
                run_name=run_name,
                model_config=config,
                loaders=loaders,
                device=device,
                learning_rate=float(variant.get("learning_rate", DEFAULT_LEARNING_RATE)),
                epochs=STUDY_EPOCHS,
                optimizer_name=str(variant["optimizer"]),
                evaluate_test=False,
            )
            study_rows.append(result)
            all_experiment_rows.append(result)
        save_summary_csv(results_dir / f"{study_name}_summary.csv", study_rows, include_test=False)
        plot_study_summary(
            results_dir / f"{study_name}_summary.png",
            f"Timeseries study: {study_name}",
            study_rows,
        )

    best_overall_run = None
    for result in all_experiment_rows:
        if is_better_training_run(result, best_overall_run):
            best_overall_run = result

    best_with_test = train_run(
        run_name=str(best_overall_run["run_name"]),
        model_config=dict(best_overall_run["model_config"]),
        loaders=loaders,
        device=device,
        learning_rate=float(best_overall_run["learning_rate"]),
        epochs=int(best_overall_run["epochs"]),
        optimizer_name=str(best_overall_run["optimizer"]),
        evaluate_test=True,
    )
    save_best_run_outputs(results_dir, best_with_test)

    print("Task 4 completed: timeseries study finished.")
    print(f"Best timeseries run: {best_with_test['run_name']}")
    print(f"Results directory: {results_dir}")


if __name__ == "__main__":
    main()

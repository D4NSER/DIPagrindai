"""Task 2: train, validate, and test a single sigmoid neuron."""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from breast_cancer_data import prepare_dataset, project_lab_root, save_json, split_dataset
from single_neuron import evaluate, predict_classes, predict_proba, train_sigmoid_neuron


RANDOM_SEED = 2026
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_EPOCHS = 200


def standardize_splits(
    train_x: np.ndarray, validation_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Standardize features using only training statistics."""
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0
    return (
        (train_x - mean) / std,
        (validation_x - mean) / std,
        (test_x - mean) / std,
        {"mean": mean.tolist(), "std": std.tolist()},
    )


def save_history_csv(
    destination: Path,
    train_loss: list[float],
    validation_loss: list[float],
    train_accuracy: list[float],
    validation_accuracy: list[float],
) -> None:
    """Save per-epoch metrics for later plotting."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "validation_loss",
                "train_accuracy",
                "validation_accuracy",
            ]
        )
        for epoch in range(len(train_loss)):
            writer.writerow(
                [
                    epoch + 1,
                    f"{train_loss[epoch]:.8f}",
                    f"{validation_loss[epoch]:.8f}",
                    f"{train_accuracy[epoch]:.8f}",
                    f"{validation_accuracy[epoch]:.8f}",
                ]
            )


def save_test_predictions_csv(
    destination: Path,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    true_labels: np.ndarray,
) -> None:
    """Save detailed test predictions required for the report."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["index", "predicted_probability", "predicted_class", "true_class"])
        for index, (probability, predicted, true_label) in enumerate(
            zip(probabilities, predictions, true_labels), start=1
        ):
            writer.writerow(
                [index, f"{float(probability):.8f}", int(predicted), int(true_label)]
            )


def run_training(
    method: str,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epochs: int = DEFAULT_EPOCHS,
) -> dict[str, object]:
    """Prepare data, train the model, and return all task outputs."""
    lab_root = project_lab_root(Path(__file__))
    raw_path = lab_root / "data" / "raw" / "breast-cancer-wisconsin.data"

    dataset, _ = prepare_dataset(raw_path=raw_path, seed=RANDOM_SEED)
    train_split, validation_split, test_split = split_dataset(dataset)

    train_x, validation_x, test_x, scaling = standardize_splits(
        train_split.features,
        validation_split.features,
        test_split.features,
    )

    result = train_sigmoid_neuron(
        train_x=train_x,
        train_y=train_split.labels,
        validation_x=validation_x,
        validation_y=validation_split.labels,
        learning_rate=learning_rate,
        epochs=epochs,
        method=method,
        seed=RANDOM_SEED,
    )

    selected_train_loss = result.train_loss_history[result.best_epoch - 1]
    selected_validation_loss = result.validation_loss_history[result.best_epoch - 1]
    selected_train_accuracy = result.train_accuracy_history[result.best_epoch - 1]
    selected_validation_accuracy = result.validation_accuracy_history[result.best_epoch - 1]

    test_loss, test_accuracy = evaluate(
        test_x, test_split.labels, result.best_weights, result.best_bias
    )
    test_probabilities = predict_proba(test_x, result.best_weights, result.best_bias)
    test_predictions = predict_classes(test_x, result.best_weights, result.best_bias)

    return {
        "method": method,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": RANDOM_SEED,
        "selected_model_rule": "best_validation_epoch",
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "best_validation_accuracy": result.best_validation_accuracy,
        "selected_weights": [float(value) for value in result.best_weights],
        "selected_bias": result.best_bias,
        "final_weights": [float(value) for value in result.final_weights],
        "final_bias": result.final_bias,
        "train_loss_history": result.train_loss_history,
        "validation_loss_history": result.validation_loss_history,
        "train_accuracy_history": result.train_accuracy_history,
        "validation_accuracy_history": result.validation_accuracy_history,
        "selected_train_loss": selected_train_loss,
        "selected_validation_loss": selected_validation_loss,
        "selected_train_accuracy": selected_train_accuracy,
        "selected_validation_accuracy": selected_validation_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "training_time_seconds": result.training_time_seconds,
        "split_sizes": {
            "train": train_split.size,
            "validation": validation_split.size,
            "test": test_split.size,
        },
        "scaling": scaling,
        "test_probabilities": test_probabilities,
        "test_predictions": test_predictions,
        "test_true_labels": test_split.labels,
    }


def main() -> None:
    """Train both required gradient descent variants and save results."""
    lab_root = project_lab_root(Path(__file__))
    results_dir = lab_root / "results" / "task2"

    for method in ("batch", "sgd"):
        output = run_training(method)
        method_dir = results_dir / method

        save_history_csv(
            method_dir / "epoch_metrics.csv",
            output["train_loss_history"],
            output["validation_loss_history"],
            output["train_accuracy_history"],
            output["validation_accuracy_history"],
        )
        save_test_predictions_csv(
            method_dir / "test_predictions.csv",
            output["test_probabilities"],
            output["test_predictions"],
            output["test_true_labels"],
        )

        summary = {
            key: value
            for key, value in output.items()
            if key not in {"test_probabilities", "test_predictions", "test_true_labels"}
        }
        save_json(summary, method_dir / "summary.json")

        print(method.upper())
        print(f"  Selected epoch: {output['best_epoch']}")
        print(f"  Test loss: {output['test_loss']:.6f}")
        print(f"  Test accuracy: {output['test_accuracy']:.6f}")
        print(f"  Training time (s): {output['training_time_seconds']:.6f}")
        print(f"  Results: {method_dir}")


if __name__ == "__main__":
    main()

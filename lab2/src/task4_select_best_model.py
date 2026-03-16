"""Task 4: choose the best Lab 2 model using validation results and save it."""

from __future__ import annotations

from pathlib import Path

from breast_cancer_data import project_lab_root, save_json
from task2_train_and_evaluate import run_training, save_history_csv, save_test_predictions_csv
from task3_experiments import EPOCHS, LEARNING_RATES, plot_epoch_curves


def main() -> None:
    """Find the best configuration by validation metrics and save final artifacts."""
    lab_root = project_lab_root(Path(__file__))
    results_dir = lab_root / "results" / "final_best_model"

    best_output: dict[str, object] | None = None

    for method in ("batch", "sgd"):
        for learning_rate in LEARNING_RATES:
            output = run_training(method=method, learning_rate=learning_rate, epochs=EPOCHS)
            if best_output is None:
                best_output = output
                continue

            current_key = (output["best_validation_accuracy"], -output["best_validation_loss"])
            best_key = (
                best_output["best_validation_accuracy"],
                -best_output["best_validation_loss"],
            )
            if current_key > best_key:
                best_output = output

    assert best_output is not None

    save_history_csv(
        results_dir / "epoch_metrics.csv",
        best_output["train_loss_history"],
        best_output["validation_loss_history"],
        best_output["train_accuracy_history"],
        best_output["validation_accuracy_history"],
    )
    save_test_predictions_csv(
        results_dir / "test_predictions.csv",
        best_output["test_probabilities"],
        best_output["test_predictions"],
        best_output["test_true_labels"],
    )
    plot_epoch_curves(results_dir / "best_model_curves.png", best_output)

    summary = {
        key: value
        for key, value in best_output.items()
        if key not in {"test_probabilities", "test_predictions", "test_true_labels"}
    }
    save_json(summary, results_dir / "summary.json")

    print("Task 4 completed: best model selected by validation metrics.")
    print(f"Method: {best_output['method']}")
    print(f"Learning rate: {best_output['learning_rate']}")
    print(f"Selected epoch: {best_output['best_epoch']}")
    print(f"Validation loss: {best_output['best_validation_loss']:.6f}")
    print(f"Validation accuracy: {best_output['best_validation_accuracy']:.6f}")
    print(f"Test loss: {best_output['test_loss']:.6f}")
    print(f"Test accuracy: {best_output['test_accuracy']:.6f}")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()

"""Task 3: compare learning rates and gradient descent methods."""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt

from breast_cancer_data import project_lab_root, save_json
from task2_train_and_evaluate import run_training


LEARNING_RATES = [0.01, 0.05, 0.1]
EPOCHS = 200


def save_experiment_table(destination: Path, rows: list[dict[str, float | str | int]]) -> None:
    """Save experiment summary rows to CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "method",
                "learning_rate",
                "epochs",
                "best_epoch",
                "final_train_loss",
                "final_validation_loss",
                "final_train_accuracy",
                "final_validation_accuracy",
                "best_validation_accuracy",
                "best_validation_loss",
                "selected_train_loss",
                "selected_validation_loss",
                "selected_train_accuracy",
                "selected_validation_accuracy",
                "test_loss",
                "test_accuracy",
                "training_time_seconds",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["method"],
                    row["learning_rate"],
                    row["epochs"],
                    row["best_epoch"],
                    f"{row['final_train_loss']:.8f}",
                    f"{row['final_validation_loss']:.8f}",
                    f"{row['final_train_accuracy']:.8f}",
                    f"{row['final_validation_accuracy']:.8f}",
                    f"{row['best_validation_accuracy']:.8f}",
                    f"{row['best_validation_loss']:.8f}",
                    f"{row['selected_train_loss']:.8f}",
                    f"{row['selected_validation_loss']:.8f}",
                    f"{row['selected_train_accuracy']:.8f}",
                    f"{row['selected_validation_accuracy']:.8f}",
                    f"{row['test_loss']:.8f}",
                    f"{row['test_accuracy']:.8f}",
                    f"{row['training_time_seconds']:.8f}",
                ]
            )


def plot_epoch_curves(destination: Path, output: dict[str, object]) -> None:
    """Plot train and validation loss/accuracy against epochs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(output["train_loss_history"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, output["train_loss_history"], label="Mokymo paklaida")
    axes[0].plot(epochs, output["validation_loss_history"], label="Validavimo paklaida")
    axes[0].set_title("Paklaida pagal epochas")
    axes[0].set_xlabel("Epocha")
    axes[0].set_ylabel("Paklaida")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(epochs, output["train_accuracy_history"], label="Mokymo tikslumas")
    axes[1].plot(
        epochs, output["validation_accuracy_history"], label="Validavimo tikslumas"
    )
    axes[1].set_title("Tikslumas pagal epochas")
    axes[1].set_xlabel("Epocha")
    axes[1].set_ylabel("Tikslumas")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    figure.suptitle(
        f"{output['method'].upper()} | learning rate = {output['learning_rate']}"
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def plot_learning_rate_bars(destination: Path, rows: list[dict[str, float | str | int]]) -> None:
    """Plot how validation/test accuracy depends on learning rate."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    batch_rows = [row for row in rows if row["method"] == "batch"]
    sgd_rows = [row for row in rows if row["method"] == "sgd"]
    labels = [str(row["learning_rate"]) for row in batch_rows]
    x_positions = list(range(len(labels)))
    width = 0.18

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(
        [x - 1.5 * width for x in x_positions],
        [row["selected_validation_accuracy"] for row in batch_rows],
        width=width,
        label="Batch val",
    )
    axes[0].bar(
        [x - 0.5 * width for x in x_positions],
        [row["test_accuracy"] for row in batch_rows],
        width=width,
        label="Batch test",
    )
    axes[0].bar(
        [x + 0.5 * width for x in x_positions],
        [row["selected_validation_accuracy"] for row in sgd_rows],
        width=width,
        label="SGD val",
    )
    axes[0].bar(
        [x + 1.5 * width for x in x_positions],
        [row["test_accuracy"] for row in sgd_rows],
        width=width,
        label="SGD test",
    )
    axes[0].set_title("Tikslumas pagal learning rate")
    axes[0].set_xlabel("Learning rate")
    axes[0].set_ylabel("Tikslumas")
    axes[0].set_xticks(x_positions, labels)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].bar(
        [x - width / 2 for x in x_positions],
        [row["training_time_seconds"] for row in batch_rows],
        width=width,
        label="Batch",
    )
    axes[1].bar(
        [x + width / 2 for x in x_positions],
        [row["training_time_seconds"] for row in sgd_rows],
        width=width,
        label="SGD",
    )
    axes[1].set_title("Mokymo laikas pagal learning rate")
    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylabel("Laikas, s")
    axes[1].set_xticks(x_positions, labels)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def main() -> None:
    """Run the required experiments and save plots plus summaries."""
    lab_root = project_lab_root(Path(__file__))
    results_dir = lab_root / "results" / "task3"

    summary_rows: list[dict[str, float | str | int]] = []
    for method in ("batch", "sgd"):
        for learning_rate in LEARNING_RATES:
            output = run_training(method=method, learning_rate=learning_rate, epochs=EPOCHS)
            summary_rows.append(
                {
                    "method": method,
                    "learning_rate": learning_rate,
                    "epochs": EPOCHS,
                    "best_epoch": output["best_epoch"],
                    "final_train_loss": output["train_loss_history"][-1],
                    "final_validation_loss": output["validation_loss_history"][-1],
                    "final_train_accuracy": output["train_accuracy_history"][-1],
                    "final_validation_accuracy": output["validation_accuracy_history"][-1],
                    "best_validation_accuracy": output["best_validation_accuracy"],
                    "best_validation_loss": output["best_validation_loss"],
                    "selected_train_loss": output["selected_train_loss"],
                    "selected_validation_loss": output["selected_validation_loss"],
                    "selected_train_accuracy": output["selected_train_accuracy"],
                    "selected_validation_accuracy": output["selected_validation_accuracy"],
                    "test_loss": output["test_loss"],
                    "test_accuracy": output["test_accuracy"],
                    "training_time_seconds": output["training_time_seconds"],
                }
            )

            plot_epoch_curves(
                results_dir / f"{method}_lr_{str(learning_rate).replace('.', '_')}_curves.png",
                output,
            )

    save_experiment_table(results_dir / "experiment_summary.csv", summary_rows)
    save_json({"learning_rates": LEARNING_RATES, "epochs": EPOCHS, "results": summary_rows}, results_dir / "experiment_summary.json")
    plot_learning_rate_bars(results_dir / "learning_rate_comparison.png", summary_rows)

    best_by_validation = max(
        summary_rows,
        key=lambda row: (row["best_validation_accuracy"], -row["best_validation_loss"]),
    )

    print("Task 3 completed: experiments finished.")
    print(f"Best setup by validation accuracy: {best_by_validation}")
    print(f"Summary CSV: {results_dir / 'experiment_summary.csv'}")
    print(f"Comparison plot: {results_dir / 'learning_rate_comparison.png'}")


if __name__ == "__main__":
    main()

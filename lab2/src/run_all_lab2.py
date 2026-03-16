"""Run all Lab 2 steps in order."""

from __future__ import annotations

from task1_prepare_data import main as task1_main
from task2_train_and_evaluate import main as task2_main
from task3_experiments import main as task3_main
from task4_select_best_model import main as task4_main


def main() -> None:
    """Execute all Lab 2 scripts in the required order."""
    print("1. Data preparation")
    task1_main()
    print()

    print("2. Neuron training, validation, and testing")
    task2_main()
    print()

    print("3. Experiments")
    task3_main()
    print()

    print("4. Best model selection")
    task4_main()


if __name__ == "__main__":
    main()

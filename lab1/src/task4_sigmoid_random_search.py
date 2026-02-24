from __future__ import annotations

from pathlib import Path

from artificial_neuron import ActivationType, load_dataset, project_lab_root
from random_parameter_search import (
    RandomParameterSearch,
    SearchConfig,
    save_search_results,
)


def main() -> None:
    lab_root = project_lab_root(Path(__file__))
    dataset_path = lab_root / "data" / "task1_points.csv"
    output_path = lab_root / "data" / "task4_sigmoid_search_results.json"

    records = load_dataset(dataset_path)
    config = SearchConfig(
        activation=ActivationType.SIGMOID,
        weight_bias_min=-10.0,
        weight_bias_max=10.0,
        target_solutions=3,
        max_attempts=1000000,
        seed=2027,
    )

    searcher = RandomParameterSearch(config)
    results = searcher.find_solutions(records)
    save_search_results(output_path, config, results)

    print("Task 4 completed: sigmoid random search found 3 solutions.")
    print(f"Results saved to: {output_path}")
    for idx, result in enumerate(results, start=1):
        p = result.parameters
        print(
            f"  {idx}. w1={p.w1:.6f}, w2={p.w2:.6f}, b={p.b:.6f} "
            f"(attempt {result.attempts_needed}, {result.correct_count}/{result.total_count})"
        )


if __name__ == "__main__":
    main()

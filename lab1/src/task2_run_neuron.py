"""Run Task 2 neuron evaluation on the generated dataset with a chosen activation function."""

from __future__ import annotations

import argparse
from pathlib import Path

from artificial_neuron import (
    ActivationType,
    ArtificialNeuron,
    load_dataset,
    load_parameters_from_metadata,
    project_lab_root,
    save_evaluation_results,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Task 2 neuron evaluation."""
    parser = argparse.ArgumentParser(
        description="Task 2: evaluate generated data with an artificial neuron."
    )
    parser.add_argument(
        "--activation",
        choices=[activation.value for activation in ActivationType],
        default=ActivationType.THRESHOLD.value,
        help="Activation function to use.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate Task 1 data with the artificial neuron and save Task 2 CSV outputs."""
    # pajungia skripta su parinkta funkcija pagal --activation argumenta
    args = parse_args()
    activation = ActivationType(args.activation)

    lab_root = project_lab_root(Path(__file__))
    dataset_path = lab_root / "data" / "task1_points.csv"
    metadata_path = lab_root / "data" / "task1_generation_metadata.json"
    output_path = lab_root / "data" / f"task2_outputs_{activation.value}.csv"

    records = load_dataset(dataset_path)
    # naudojame 1 punkto metadata, kaip gerus parametrus
    parameters = load_parameters_from_metadata(metadata_path)
    neuron = ArtificialNeuron(parameters=parameters, activation=activation)

    # vertinu irasa ir issaugau
    outputs = [neuron.evaluate(record.x1, record.x2) for record in records]
    save_evaluation_results(output_path, records, outputs)

    correct = sum(
        1
        for record, output in zip(records, outputs)
        if record.target_class == output.predicted_class
    )

    print(f"Activation: {activation.value}")
    print("Neuron parameters:")
    print(f"  w1={parameters.w1:.6f}, w2={parameters.w2:.6f}, b={parameters.b:.6f}")
    print(f"Records evaluated: {len(records)}")
    print(f"Correct classifications: {correct}/{len(records)}")
    print(f"Results saved to: {output_path}")
    print("Sample rows (first 5):")
    for index, (record, output) in enumerate(zip(records, outputs), start=1):
        if index > 5:
            break
        print(
            f"  {index}. x=({record.x1:.3f}, {record.x2:.3f}) -> "
            f"a={output.a_value:.3f}, act={output.activation_value:.3f}, "
            f"pred={output.predicted_class}, true={record.target_class}"
        )


if __name__ == "__main__":
    main()

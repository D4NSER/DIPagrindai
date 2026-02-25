"""Core data structures and logic for evaluating a single artificial neuron."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp
from pathlib import Path
import csv
import json


class ActivationType(str, Enum):
    """Supported activation function types for the artificial neuron."""

    # reiksmes argumentams

    THRESHOLD = "threshold"
    SIGMOID = "sigmoid"


@dataclass(frozen=True)
class DataRecord:
    """One dataset row used as neuron input and ground-truth target label."""

    x1: float
    x2: float
    target_class: int

    @property
    def inputs(self) -> tuple[float, float]:
        """Return the feature values as an `(x1, x2)` tuple."""
        return (self.x1, self.x2)


@dataclass(frozen=True)
class NeuronParameters:
    """Container for neuron weights and bias `(w1, w2, b)`."""

    w1: float
    w2: float
    b: float


@dataclass(frozen=True)
class NeuronOutput:
    """Stores the linear output, activation output, and predicted class."""

    a_value: float
    activation_value: float
    predicted_class: int


class ArtificialNeuron:
    """Artificial neuron that evaluates a 2D input with a selected activation."""

    def __init__(
        self, parameters: NeuronParameters, activation: ActivationType
    ) -> None:
        """Initialize the neuron with fixed parameters and activation type."""
        self.parameters = parameters
        self.activation = activation

    def compute_a(self, x1: float, x2: float) -> float:
        """Compute the linear combination `a = x1*w1 + x2*w2 + b`."""
        p = self.parameters
        return x1 * p.w1 + x2 * p.w2 + p.b

    def activation_value(self, a_value: float) -> float:
        """Apply the configured activation function to the linear output `a`."""
        if self.activation == ActivationType.THRESHOLD:
            # slenkstine is karto grazina 1 arba 0
            return 1.0 if a_value >= 0.0 else 0.0
        if self.activation == ActivationType.SIGMOID:
            # del skaitinio stabilumo naudoji dvi sakas
            if a_value >= 0:
                z = exp(-a_value)
                return 1.0 / (1.0 + z)
            z = exp(a_value)
            return z / (1.0 + z)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def predict_class(self, activation_value: float) -> int:
        """Convert activation output to a binary class label."""
        if self.activation == ActivationType.THRESHOLD:
            return int(activation_value)
        # sigmoidinei gaunam klase apvalindami iki 1 arba 0
        return int(round(activation_value))

    def evaluate(self, x1: float, x2: float) -> NeuronOutput:
        """Evaluate one input point and return both intermediate and final outputs."""
        a_value = self.compute_a(x1, x2)
        act_value = self.activation_value(a_value)
        predicted = self.predict_class(act_value)
        return NeuronOutput(
            a_value=a_value, activation_value=act_value, predicted_class=predicted
        )


def load_dataset(csv_path: Path) -> list[DataRecord]:
    """Load Task 1 dataset from CSV into typed `DataRecord` objects."""
    records: list[DataRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        # privalo buti csv
        for row in reader:
            records.append(
                DataRecord(
                    x1=float(row["x1"]),
                    x2=float(row["x2"]),
                    target_class=int(row["class"]),
                )
            )
    return records


def load_parameters_from_metadata(metadata_path: Path) -> NeuronParameters:
    """Load separator parameters from Task 1 metadata JSON."""
    # is 1 metadata naudoju duomenis, kad po to galima butu viska atkurt
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    separator = payload["separator"]
    return NeuronParameters(
        w1=float(separator["w1"]),
        w2=float(separator["w2"]),
        b=float(separator["b"]),
    )


def save_evaluation_results(
    destination: Path,
    records: list[DataRecord],
    outputs: list[NeuronOutput],
) -> None:
    """Save Task 2 neuron evaluation outputs to a CSV file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["x1", "x2", "true_class", "a", "activation_value", "predicted_class"]
        )
        for record, output in zip(records, outputs):
            writer.writerow(
                [
                    f"{record.x1:.6f}",
                    f"{record.x2:.6f}",
                    record.target_class,
                    f"{output.a_value:.6f}",
                    f"{output.activation_value:.6f}",
                    output.predicted_class,
                ]
            )


def project_lab_root(current_file: Path) -> Path:
    """Resolve the `lab1` root directory from a script path."""
    return current_file.resolve().parents[1]

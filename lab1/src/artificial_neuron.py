from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp
from pathlib import Path
import csv
import json


class ActivationType(str, Enum):
    THRESHOLD = "threshold"
    SIGMOID = "sigmoid"


@dataclass(frozen=True)
class DataRecord:
    x1: float
    x2: float
    target_class: int

    @property
    def inputs(self) -> tuple[float, float]:
        return (self.x1, self.x2)


@dataclass(frozen=True)
class NeuronParameters:
    w1: float
    w2: float
    b: float


@dataclass(frozen=True)
class NeuronOutput:
    a_value: float
    activation_value: float
    predicted_class: int


class ArtificialNeuron:
    def __init__(self, parameters: NeuronParameters, activation: ActivationType) -> None:
        self.parameters = parameters
        self.activation = activation

    def compute_a(self, x1: float, x2: float) -> float:
        p = self.parameters
        return x1 * p.w1 + x2 * p.w2 + p.b

    def activation_value(self, a_value: float) -> float:
        if self.activation == ActivationType.THRESHOLD:
            return 1.0 if a_value >= 0.0 else 0.0
        if self.activation == ActivationType.SIGMOID:
            if a_value >= 0:
                z = exp(-a_value)
                return 1.0 / (1.0 + z)
            z = exp(a_value)
            return z / (1.0 + z)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def predict_class(self, activation_value: float) -> int:
        if self.activation == ActivationType.THRESHOLD:
            return int(activation_value)
        return int(round(activation_value))

    def evaluate(self, x1: float, x2: float) -> NeuronOutput:
        a_value = self.compute_a(x1, x2)
        act_value = self.activation_value(a_value)
        predicted = self.predict_class(act_value)
        return NeuronOutput(a_value=a_value, activation_value=act_value, predicted_class=predicted)


def load_dataset(csv_path: Path) -> list[DataRecord]:
    records: list[DataRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["x1", "x2", "true_class", "a", "activation_value", "predicted_class"])
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
    return current_file.resolve().parents[1]

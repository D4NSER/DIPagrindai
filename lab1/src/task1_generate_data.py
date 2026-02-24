from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POINTS_PER_CLASS = 15
RANDOM_SEED = 42
SAMPLING_BOUNDS = (-10.0, 10.0)
SEPARATION_MARGIN = 1.25
MAX_SAMPLING_ATTEMPTS = 1000


@dataclass(frozen=True)
class DatasetPoint:
    x1: float
    x2: float
    label: int


@dataclass(frozen=True)
class SeparatingHyperplane:
    weights: np.ndarray
    bias: float
    margin: float

    def score(self, point: np.ndarray) -> float:
        return float(np.dot(self.weights, point) + self.bias)

    def classify_with_margin(self, point: np.ndarray) -> int | None:
        score = self.score(point)
        if score >= self.margin:
            return 1
        if score <= -self.margin:
            return 0
        return None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_random_separator(rng: np.random.Generator) -> SeparatingHyperplane:
    raw_weights = rng.normal(loc=0.0, scale=1.0, size=2)
    norm = np.linalg.norm(raw_weights)
    while norm == 0.0:
        raw_weights = rng.normal(loc=0.0, scale=1.0, size=2)
        norm = np.linalg.norm(raw_weights)

    weights = raw_weights / norm
    bias = float(rng.uniform(-2.5, 2.5))
    return SeparatingHyperplane(weights=weights, bias=bias, margin=SEPARATION_MARGIN)


def sample_linearly_separable_points(
    rng: np.random.Generator,
    separator: SeparatingHyperplane,
    points_per_class: int,
    bounds: tuple[float, float],
) -> list[DatasetPoint]:
    lower, upper = bounds
    class_counts = {0: 0, 1: 0}
    sampled_points: list[DatasetPoint] = []

    for _ in range(MAX_SAMPLING_ATTEMPTS):
        if all(count >= points_per_class for count in class_counts.values()):
            break

        candidate = rng.uniform(lower, upper, size=2)
        label = separator.classify_with_margin(candidate)
        if label is None or class_counts[label] >= points_per_class:
            continue

        sampled_points.append(
            DatasetPoint(x1=float(candidate[0]), x2=float(candidate[1]), label=label)
        )
        class_counts[label] += 1
    else:
        raise RuntimeError(
            "Could not generate enough linearly separable points. Increase attempts or reduce margin."
        )

    return sampled_points


def validate_linear_separability(
    points: list[DatasetPoint],
    separator: SeparatingHyperplane,
) -> None:
    for point in points:
        point_vector = np.array([point.x1, point.x2], dtype=float)
        predicted_label = separator.classify_with_margin(point_vector)
        if predicted_label != point.label:
            raise ValueError(
                f"Point ({point.x1:.3f}, {point.x2:.3f}) with label {point.label} is not separated."
            )


def save_points_to_csv(points: list[DatasetPoint], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["x1", "x2", "class"])
        for point in points:
            writer.writerow([f"{point.x1:.6f}", f"{point.x2:.6f}", point.label])


def save_generation_metadata(
    separator: SeparatingHyperplane,
    destination: Path,
    seed: int,
    bounds: tuple[float, float],
) -> None:
    payload = {
        "seed": seed,
        "sampling_bounds": [bounds[0], bounds[1]],
        "margin": separator.margin,
        "separator": {
            "w1": float(separator.weights[0]),
            "w2": float(separator.weights[1]),
            "b": float(separator.bias),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_points(points: list[DatasetPoint], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    class_zero = [p for p in points if p.label == 0]
    class_one = [p for p in points if p.label == 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        [p.x1 for p in class_zero],
        [p.x2 for p in class_zero],
        c="#1f77b4",
        label="Klase 0",
        edgecolors="black",
        linewidths=0.6,
        s=60,
    )
    plt.scatter(
        [p.x1 for p in class_one],
        [p.x2 for p in class_one],
        c="#d62728",
        label="Klase 1",
        edgecolors="black",
        linewidths=0.6,
        s=60,
        marker="s",
    )

    plt.title("1 uzduotis: sugeneruoti tiesiskai atskiriami duomenys")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    separator = build_random_separator(rng)
    points = sample_linearly_separable_points(
        rng=rng,
        separator=separator,
        points_per_class=POINTS_PER_CLASS,
        bounds=SAMPLING_BOUNDS,
    )
    validate_linear_separability(points, separator)

    root = project_root()
    csv_path = root / "data" / "task1_points.csv"
    metadata_path = root / "data" / "task1_generation_metadata.json"
    figure_path = root / "figures" / "task1_points.png"

    save_points_to_csv(points, csv_path)
    save_generation_metadata(
        separator=separator,
        destination=metadata_path,
        seed=RANDOM_SEED,
        bounds=SAMPLING_BOUNDS,
    )
    plot_points(points, figure_path)

    counts = {
        0: sum(p.label == 0 for p in points),
        1: sum(p.label == 1 for p in points),
    }
    print("Task 1 dataset generated successfully.")
    print(f"Class counts: {counts}")
    print(f"CSV: {csv_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()

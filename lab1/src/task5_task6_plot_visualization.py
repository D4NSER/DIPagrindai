"""Plot Task 5-6 outputs: data points, separating lines, and weight vectors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from artificial_neuron import NeuronParameters, load_dataset, project_lab_root


@dataclass(frozen=True)
class ThresholdSolution:
    """One threshold-search solution used to plot a line and its weight vector."""

    index: int
    parameters: NeuronParameters


def load_threshold_solutions(json_path: Path) -> list[ThresholdSolution]:
    """Load Task 3 threshold solutions from JSON into typed objects."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    solutions: list[ThresholdSolution] = []
    for item in payload["results"]:
        p = item["parameters"]
        solutions.append(
            ThresholdSolution(
                index=int(item["index"]),
                parameters=NeuronParameters(
                    w1=float(p["w1"]), w2=float(p["w2"]), b=float(p["b"])
                ),
            )
        )
    return solutions


# vektoriaus pradziai pasirenkame taska ant tieses (negali but (0,0)), kad atsispindetu poslinkis.
def closest_point_on_line_to_origin(parameters: NeuronParameters) -> np.ndarray:
    """Compute the point on the separating line that is closest to the origin."""
    w = np.array([parameters.w1, parameters.w2], dtype=float)
    norm_sq = float(np.dot(w, w))
    if norm_sq == 0.0:
        raise ValueError("Weights cannot both be zero when drawing a separating line.")
    return (-parameters.b / norm_sq) * w


def line_points_for_plot(
    parameters: NeuronParameters, x_limits: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Generate XY coordinates for plotting a separating line."""
    w1, w2, b = parameters.w1, parameters.w2, parameters.b
    x_values = np.linspace(x_limits[0], x_limits[1], 200)

    # istraukiu y is a lygties
    if abs(w2) > 1e-9:
        y_values = -(w1 * x_values + b) / w2
        return x_values, y_values

    x_const = -b / w1
    y_values = np.linspace(x_limits[0], x_limits[1], 200)
    x_values = np.full_like(y_values, x_const)
    return x_values, y_values


def choose_vector_scale(
    solutions: list[ThresholdSolution],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> float:
    """Choose a readable vector length based on current plot dimensions."""
    diagonal = float(np.hypot(x_limits[1] - x_limits[0], y_limits[1] - y_limits[0]))
    return 0.12 * diagonal


def plot_task5_task6(
    dataset_csv: Path,
    threshold_results_json: Path,
    output_figure: Path,
) -> None:
    """Plot Task 1 points together with Task 3 lines and weight vectors."""
    records = load_dataset(dataset_csv)
    solutions = load_threshold_solutions(threshold_results_json)

    points_class0 = [record for record in records if record.target_class == 0]
    points_class1 = [record for record in records if record.target_class == 1]

    all_x = [record.x1 for record in records]
    all_y = [record.x2 for record in records]
    padding = 1.5
    x_limits = (min(all_x) - padding, max(all_x) + padding)
    y_limits = (min(all_y) - padding, max(all_y) + padding)

    colors = ["#d95f02", "#1b9e77", "#7570b3"]

    plt.figure(figsize=(9, 7))
    plt.scatter(
        [p.x1 for p in points_class0],
        [p.x2 for p in points_class0],
        c="#1f77b4",
        edgecolors="black",
        linewidths=0.5,
        s=60,
        label="Klase 0",
    )
    plt.scatter(
        [p.x1 for p in points_class1],
        [p.x2 for p in points_class1],
        c="#e41a1c",
        marker="s",
        edgecolors="black",
        linewidths=0.5,
        s=60,
        label="Klase 1",
    )

    # pasakau vektoriu ilgi
    vector_scale = choose_vector_scale(solutions, x_limits, y_limits)

    for color, solution in zip(colors, solutions):
        p = solution.parameters
        x_line, y_line = line_points_for_plot(p, x_limits)
        plt.plot(
            x_line,
            y_line,
            color=color,
            linewidth=2.0,
            label=f"Skyrimo tiese #{solution.index}",
        )

        # svoriu vektorius statmenas, nes yra normale
        start_point = closest_point_on_line_to_origin(p)
        weight_vector = np.array([p.w1, p.w2], dtype=float)
        norm = float(np.linalg.norm(weight_vector))
        if norm == 0.0:
            continue
        direction = (weight_vector / norm) * vector_scale

        plt.quiver(
            start_point[0],
            start_point[1],
            direction[0],
            direction[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.005,
            headwidth=4.5,
            headlength=6,
            label=f"Svoriu vektorius #{solution.index}",
        )

    plt.title("1 uzduotis: taskai, skyrimo tieses ir svoriu vektoriai")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(*x_limits)
    plt.ylim(*y_limits)
    plt.grid(True, linestyle="--", alpha=0.4)
    # suivienodinu asiu masteli
    plt.axis("equal")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    output_figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_figure, dpi=220)
    plt.close()


def main() -> None:
    """Load default inputs and generate the combined Tasks 5-6 visualization figure."""
    lab_root = project_lab_root(Path(__file__))
    dataset_csv = lab_root / "data" / "task1_points.csv"
    threshold_results_json = lab_root / "data" / "task3_threshold_search_results.json"
    output_figure = lab_root / "figures" / "task5_task6_points_lines_vectors.png"

    plot_task5_task6(dataset_csv, threshold_results_json, output_figure)

    print("Task 5-6 figure generated successfully.")
    print(f"Figure: {output_figure}")


if __name__ == "__main__":
    main()

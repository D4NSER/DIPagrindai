from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class ActivationRow:
    x1: float
    x2: float
    true_class: int
    a: float
    activation_value: float
    predicted_class: int

    @property
    def key(self) -> tuple[float, float, int]:
        return (self.x1, self.x2, self.true_class)


@dataclass(frozen=True)
class RowComparison:
    index: int
    threshold: ActivationRow
    sigmoid: ActivationRow

    @property
    def a_diff(self) -> float:
        return self.sigmoid.a - self.threshold.a

    @property
    def activation_diff(self) -> float:
        return self.sigmoid.activation_value - self.threshold.activation_value

    @property
    def predicted_match(self) -> bool:
        return self.threshold.predicted_class == self.sigmoid.predicted_class

    @property
    def both_correct(self) -> bool:
        return (
            self.threshold.predicted_class == self.threshold.true_class
            and self.sigmoid.predicted_class == self.sigmoid.true_class
        )

    @property
    def abs_a(self) -> float:
        return abs(self.threshold.a)

    @property
    def sigmoid_distance_from_half(self) -> float:
        return abs(self.sigmoid.activation_value - 0.5)


def lab_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = lab_root_from_script()
    parser = argparse.ArgumentParser(
        description="Detailed comparison of Task 2 threshold vs sigmoid CSV outputs."
    )
    parser.add_argument(
        "--threshold-csv",
        type=Path,
        default=root / "data" / "task2_outputs_threshold.csv",
    )
    parser.add_argument(
        "--sigmoid-csv",
        type=Path,
        default=root / "data" / "task2_outputs_sigmoid.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=root / "report" / "task2_activation_comparison_detailed.csv",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=root / "report" / "task2_activation_comparison_summary.md",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Rows closest to decision boundary (smallest |a|) to show in Markdown.",
    )
    return parser.parse_args()


def read_activation_csv(path: Path) -> list[ActivationRow]:
    rows: list[ActivationRow] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "x1",
            "x2",
            "true_class",
            "a",
            "activation_value",
            "predicted_class",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        for raw in reader:
            rows.append(
                ActivationRow(
                    x1=float(raw["x1"]),
                    x2=float(raw["x2"]),
                    true_class=int(raw["true_class"]),
                    a=float(raw["a"]),
                    activation_value=float(raw["activation_value"]),
                    predicted_class=int(raw["predicted_class"]),
                )
            )
    return rows


def build_comparisons(
    threshold_rows: list[ActivationRow],
    sigmoid_rows: list[ActivationRow],
) -> list[RowComparison]:
    if len(threshold_rows) != len(sigmoid_rows):
        raise ValueError(
            f"Row count mismatch: threshold={len(threshold_rows)}, sigmoid={len(sigmoid_rows)}"
        )

    comparisons: list[RowComparison] = []
    for index, (threshold, sigmoid) in enumerate(
        zip(threshold_rows, sigmoid_rows), start=1
    ):
        if threshold.key != sigmoid.key:
            raise ValueError(
                f"Row mismatch at index {index}: threshold={threshold.key}, sigmoid={sigmoid.key}"
            )
        comparisons.append(
            RowComparison(index=index, threshold=threshold, sigmoid=sigmoid)
        )
    return comparisons


def save_detailed_csv(path: Path, comparisons: list[RowComparison]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "row_index",
                "x1",
                "x2",
                "true_class",
                "a_threshold",
                "a_sigmoid",
                "a_diff",
                "threshold_activation_value",
                "sigmoid_activation_value",
                "activation_diff_sigmoid_minus_threshold",
                "threshold_predicted_class",
                "sigmoid_predicted_class",
                "predicted_match",
                "both_correct",
                "abs_a",
                "sigmoid_distance_from_0_5",
            ]
        )

        for r in comparisons:
            t, s = r.threshold, r.sigmoid
            writer.writerow(
                [
                    r.index,
                    f"{t.x1:.6f}",
                    f"{t.x2:.6f}",
                    t.true_class,
                    f"{t.a:.6f}",
                    f"{s.a:.6f}",
                    f"{r.a_diff:.12f}",
                    f"{t.activation_value:.6f}",
                    f"{s.activation_value:.6f}",
                    f"{r.activation_diff:.6f}",
                    t.predicted_class,
                    s.predicted_class,
                    int(r.predicted_match),
                    int(r.both_correct),
                    f"{r.abs_a:.6f}",
                    f"{r.sigmoid_distance_from_half:.6f}",
                ]
            )


def markdown_table(rows: list[RowComparison]) -> str:
    lines = [
        "| # | x1 | x2 | true | a | thr_act | sig_act | thr_pred | sig_pred | |a| | |sig-0.5| |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        t, s = r.threshold, r.sigmoid
        lines.append(
            f"| {r.index} | {t.x1:.3f} | {t.x2:.3f} | {t.true_class} | {t.a:.3f} | "
            f"{t.activation_value:.3f} | {s.activation_value:.3f} | "
            f"{t.predicted_class} | {s.predicted_class} | {r.abs_a:.3f} | {r.sigmoid_distance_from_half:.3f} |"
        )
    return "\n".join(lines)


def save_markdown_summary(
    path: Path, comparisons: list[RowComparison], top_n: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    predicted_matches = sum(1 for r in comparisons if r.predicted_match)
    predicted_mismatches = len(comparisons) - predicted_matches
    both_correct = sum(1 for r in comparisons if r.both_correct)

    a_diffs = [abs(r.a_diff) for r in comparisons]
    activation_diffs = [r.activation_diff for r in comparisons]
    abs_activation_diffs = [abs(v) for v in activation_diffs]
    threshold_acts = [r.threshold.activation_value for r in comparisons]
    sigmoid_acts = [r.sigmoid.activation_value for r in comparisons]
    abs_a_values = [r.abs_a for r in comparisons]
    sigmoid_distances = [r.sigmoid_distance_from_half for r in comparisons]

    near_boundary_rows = sorted(comparisons, key=lambda r: (r.abs_a, r.index))[
        : max(0, top_n)
    ]
    mismatch_rows = [r for r in comparisons if not r.predicted_match]
    threshold_unique = sorted({round(v, 12) for v in threshold_acts})

    lines: list[str] = []
    lines.append("# Task 2 aktyvacijų palyginimas (slenkstinė vs sigmoidinė)")
    lines.append("")
    lines.append("## Santrauka")
    lines.append(f"- Eilučių skaičius: **{len(comparisons)}**")
    lines.append(
        f"- Prognozių sutapimas tarp aktyvacijų: **{predicted_matches}/{len(comparisons)}**"
    )
    lines.append(f"- Prognozių nesutapimai: **{predicted_mismatches}**")
    lines.append(
        f"- Abi aktyvacijos teisingai klasifikavo: **{both_correct}/{len(comparisons)}**"
    )
    lines.append(f"- Maks. `|a_sigmoid - a_threshold|`: **{max(a_diffs):.12f}**")
    lines.append(
        f"- Vid. aktyvacijų skirtumas `(sig - thr)`: **{mean(activation_diffs):.6f}**"
    )
    lines.append(f"- Maks. `|sig - thr|`: **{max(abs_activation_diffs):.6f}**")
    lines.append(f"- Slenkstinės unikalios reikšmės: **{threshold_unique}**")
    lines.append(
        f"- Sigmoidinės intervalas: **[{min(sigmoid_acts):.6f}, {max(sigmoid_acts):.6f}]**"
    )
    lines.append(
        f"- `|a|` intervalas: **[{min(abs_a_values):.6f}, {max(abs_a_values):.6f}]**"
    )
    lines.append(
        f"- Sigmoidinės atstumo iki `0.5` intervalas: **[{min(sigmoid_distances):.6f}, {max(sigmoid_distances):.6f}]**"
    )
    lines.append("")
    lines.append("## Interpretacija")
    lines.append(
        "- `a` reikšmės turi sutapti abiejuose CSV, nes aktyvacija taikoma po to paties tiesinio skaičiavimo."
    )
    lines.append(
        "- Slenkstinė grąžina tik `0`/`1`, o sigmoidinė grąžina reikšmes intervale `(0,1)`."
    )
    lines.append(
        "- Eilutės su mažiausiu `|a|` yra arčiausiai sprendimo ribos ir yra svarbiausios lyginimui."
    )
    lines.append("")
    lines.append(f"## {top_n} eilutės arčiausiai sprendimo ribos (mažiausias `|a|`)")
    lines.append(markdown_table(near_boundary_rows))
    lines.append("")
    lines.append("## Prognozių nesutapimai")
    if mismatch_rows:
        lines.append(markdown_table(mismatch_rows))
    else:
        lines.append("Nesutapimų nerasta - galutinės klasės sutampa visoms eilutėms.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_console_summary(
    comparisons: list[RowComparison], out_csv: Path, out_md: Path
) -> None:
    predicted_mismatches = sum(1 for r in comparisons if not r.predicted_match)
    max_abs_a_diff = max(abs(r.a_diff) for r in comparisons) if comparisons else 0.0
    max_abs_act_diff = (
        max(abs(r.activation_diff) for r in comparisons) if comparisons else 0.0
    )

    print("Detailed comparison completed.")
    print(f"Rows compared: {len(comparisons)}")
    print(f"Predicted class mismatches (threshold vs sigmoid): {predicted_mismatches}")
    print(f"Max |a difference|: {max_abs_a_diff:.12f}")
    print(f"Max |activation difference|: {max_abs_act_diff:.6f}")
    print(f"Detailed CSV: {out_csv}")
    print(f"Markdown summary: {out_md}")


def main() -> None:
    args = parse_args()
    threshold_rows = read_activation_csv(args.threshold_csv)
    sigmoid_rows = read_activation_csv(args.sigmoid_csv)
    comparisons = build_comparisons(threshold_rows, sigmoid_rows)

    save_detailed_csv(args.out_csv, comparisons)
    save_markdown_summary(args.out_md, comparisons, args.top_n)
    print_console_summary(comparisons, args.out_csv, args.out_md)


if __name__ == "__main__":
    main()

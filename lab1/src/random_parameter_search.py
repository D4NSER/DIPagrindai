from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from artificial_neuron import (
    ActivationType,
    ArtificialNeuron,
    DataRecord,
    NeuronParameters,
)


@dataclass(frozen=True)
class SearchConfig:
    activation: ActivationType
    weight_bias_min: float
    weight_bias_max: float
    target_solutions: int
    max_attempts: int
    seed: int


@dataclass(frozen=True)
class SearchResult:
    parameters: NeuronParameters
    attempts_needed: int
    correct_count: int
    total_count: int


class RandomParameterSearch:
    def __init__(self, config: SearchConfig) -> None:
        if config.weight_bias_min >= config.weight_bias_max:
            raise ValueError("weight_bias_min must be smaller than weight_bias_max")
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _random_parameters(self) -> NeuronParameters:
        low = self.config.weight_bias_min
        high = self.config.weight_bias_max
        return NeuronParameters(
            w1=float(self.rng.uniform(low, high)),
            w2=float(self.rng.uniform(low, high)),
            b=float(self.rng.uniform(low, high)),
        )

    @staticmethod
    def _is_distinct(
        parameters: NeuronParameters,
        existing: list[SearchResult],
        tolerance: float = 1e-6,
    ) -> bool:
        for result in existing:
            p = result.parameters
            if (
                abs(parameters.w1 - p.w1) < tolerance
                and abs(parameters.w2 - p.w2) < tolerance
                and abs(parameters.b - p.b) < tolerance
            ):
                return False
        return True

    def _evaluate_candidate(
        self, records: list[DataRecord], parameters: NeuronParameters
    ) -> tuple[int, int]:
        neuron = ArtificialNeuron(
            parameters=parameters, activation=self.config.activation
        )
        total = len(records)
        correct = 0
        for record in records:
            output = neuron.evaluate(record.x1, record.x2)
            if output.predicted_class == record.target_class:
                correct += 1
        return correct, total

    def find_solutions(self, records: list[DataRecord]) -> list[SearchResult]:
        solutions: list[SearchResult] = []

        for attempt in range(1, self.config.max_attempts + 1):
            candidate = self._random_parameters()
            correct, total = self._evaluate_candidate(records, candidate)
            if correct != total:
                continue
            if not self._is_distinct(candidate, solutions):
                continue

            solutions.append(
                SearchResult(
                    parameters=candidate,
                    attempts_needed=attempt,
                    correct_count=correct,
                    total_count=total,
                )
            )
            if len(solutions) >= self.config.target_solutions:
                break

        if len(solutions) < self.config.target_solutions:
            raise RuntimeError(
                f"Found only {len(solutions)} solutions out of {self.config.target_solutions} "
                f"within {self.config.max_attempts} attempts."
            )

        return solutions


def save_search_results(
    destination: Path, config: SearchConfig, results: list[SearchResult]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "activation": config.activation.value,
        "search_interval": [config.weight_bias_min, config.weight_bias_max],
        "target_solutions": config.target_solutions,
        "max_attempts": config.max_attempts,
        "seed": config.seed,
        "results": [
            {
                "index": idx,
                "attempts_needed": result.attempts_needed,
                "correct_count": result.correct_count,
                "total_count": result.total_count,
                "parameters": {
                    "w1": result.parameters.w1,
                    "w2": result.parameters.w2,
                    "b": result.parameters.b,
                },
            }
            for idx, result in enumerate(results, start=1)
        ],
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

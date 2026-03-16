"""Simple sigmoid neuron training utilities for Lab 2."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class TrainingResult:
    """Final neuron parameters and collected metrics."""

    final_weights: np.ndarray
    final_bias: float
    best_weights: np.ndarray
    best_bias: float
    best_epoch: int
    best_validation_loss: float
    best_validation_accuracy: float
    train_loss_history: list[float]
    validation_loss_history: list[float]
    train_accuracy_history: list[float]
    validation_accuracy_history: list[float]
    training_time_seconds: float


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply a numerically stable sigmoid."""
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def predict_proba(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Return sigmoid outputs for all rows."""
    return sigmoid(features @ weights + bias)


def predict_classes(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Round sigmoid outputs to 0 or 1."""
    return np.rint(predict_proba(features, weights, bias)).astype(int)


def binary_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute average binary cross-entropy."""
    p = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float(np.mean(labels == predictions))


def evaluate(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: float) -> tuple[float, float]:
    """Return loss and accuracy for a dataset."""
    probabilities = predict_proba(features, weights, bias)
    predictions = np.rint(probabilities).astype(int)
    return binary_cross_entropy(labels, probabilities), accuracy(labels, predictions)


def train_sigmoid_neuron(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    learning_rate: float,
    epochs: int,
    method: str,
    seed: int,
) -> TrainingResult:
    """Train a single sigmoid neuron using batch GD or SGD."""
    if method not in {"batch", "sgd"}:
        raise ValueError("method must be 'batch' or 'sgd'")

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.1, size=train_x.shape[1])
    bias = 0.0

    train_loss_history: list[float] = []
    validation_loss_history: list[float] = []
    train_accuracy_history: list[float] = []
    validation_accuracy_history: list[float] = []
    best_weights = weights.copy()
    best_bias = float(bias)
    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_accuracy = float("-inf")

    start_time = time.perf_counter()

    for epoch_index in range(epochs):
        if method == "batch":
            probabilities = predict_proba(train_x, weights, bias)
            errors = probabilities - train_y
            weights -= learning_rate * (train_x.T @ errors) / len(train_x)
            bias -= learning_rate * float(np.mean(errors))
        else:
            order = rng.permutation(len(train_x))
            for index in order:
                row = train_x[index]
                target = train_y[index]
                probability = float(sigmoid(np.array([row @ weights + bias]))[0])
                error = probability - float(target)
                weights -= learning_rate * error * row
                bias -= learning_rate * error

        train_loss, train_accuracy = evaluate(train_x, train_y, weights, bias)
        validation_loss, validation_accuracy = evaluate(
            validation_x, validation_y, weights, bias
        )
        train_loss_history.append(train_loss)
        validation_loss_history.append(validation_loss)
        train_accuracy_history.append(train_accuracy)
        validation_accuracy_history.append(validation_accuracy)

        if (
            validation_accuracy > best_validation_accuracy
            or (
                validation_accuracy == best_validation_accuracy
                and validation_loss < best_validation_loss
            )
        ):
            best_weights = weights.copy()
            best_bias = float(bias)
            best_epoch = epoch_index + 1
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy

    elapsed = time.perf_counter() - start_time
    return TrainingResult(
        final_weights=weights.copy(),
        final_bias=float(bias),
        best_weights=best_weights,
        best_bias=best_bias,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_validation_accuracy=best_validation_accuracy,
        train_loss_history=train_loss_history,
        validation_loss_history=validation_loss_history,
        train_accuracy_history=train_accuracy_history,
        validation_accuracy_history=validation_accuracy_history,
        training_time_seconds=elapsed,
    )

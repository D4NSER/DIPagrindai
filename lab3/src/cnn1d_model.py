"""Simple configurable Conv1D classifier for keystroke time series."""

from __future__ import annotations

import torch
from torch import nn


class ConvNet1D(nn.Module):
    """Student-friendly configurable 1D CNN."""

    def __init__(
        self,
        input_length: int,
        num_classes: int,
        conv_channels: list[int],
        dense_units: int,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        activation_name: str = "relu",
    ) -> None:
        super().__init__()
        activation_factory = build_activation_factory(activation_name)

        layers: list[nn.Module] = []
        in_channels = 1
        current_length = input_length
        for out_channels in conv_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(activation_factory())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
            current_length //= 2

        self.features = nn.Sequential(*layers)
        flattened_size = conv_channels[-1] * current_length

        classifier_layers: list[nn.Module] = [nn.Flatten(), nn.Linear(flattened_size, dense_units)]
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(dense_units))
        classifier_layers.append(activation_factory())
        if dropout_rate > 0.0:
            classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(dense_units, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_activation_factory(name: str) -> type[nn.Module]:
    """Map activation name to a PyTorch module factory."""
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU
    if normalized == "tanh":
        return nn.Tanh
    if normalized == "leaky_relu":
        return nn.LeakyReLU
    raise ValueError(f"Unsupported activation: {name}")


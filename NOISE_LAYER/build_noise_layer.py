"""Factory and composition utilities for degradation layers."""

import torch
import torch.nn as nn

from .PIMoG_Layer import PIMoGLayer
from .Projector_Layer import ProjectorSimulator


class MixedNoiseLayer(nn.Module):
    """Select one degradation layer for the whole batch on each forward call."""

    def __init__(self, layers, probs=None):
        super().__init__()
        if not layers:
            raise ValueError("layers must not be empty")
        self.layers = nn.ModuleList(layers)
        if probs is None:
            probs = [1.0 / len(layers)] * len(layers)
        if len(probs) != len(layers) or any(float(p) < 0 for p in probs):
            raise ValueError("probs must be non-negative and match layers")
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        if probs_tensor.sum() <= 0:
            raise ValueError("at least one probability must be positive")
        self.register_buffer("probs", probs_tensor / probs_tensor.sum())

    def forward(self, x):
        index = torch.multinomial(self.probs, 1).item()
        return self.layers[index](x)


def get_noise_layer_type(config):
    """Return the degradation type selected by ``noise_layer.type``."""
    return str(config.get("noise_layer", {}).get("type", "none")).lower()


def build_noise_layer(config):
    """Build ``none``, ``pimog``, ``projector``, or ``mixed`` from a config dict."""
    noise_cfg = config.get("noise_layer", {})
    noise_type = get_noise_layer_type(config)

    if noise_type == "none":
        return nn.Identity()
    if noise_type == "pimog":
        return PIMoGLayer(**noise_cfg.get("pimog", {}))
    if noise_type == "projector":
        return ProjectorSimulator(**noise_cfg.get("projector", {}))
    if noise_type == "mixed":
        return MixedNoiseLayer(
            layers=[
                PIMoGLayer(**noise_cfg.get("pimog", {})),
                ProjectorSimulator(**noise_cfg.get("projector", {})),
            ],
            probs=noise_cfg.get("mixed_probs", [0.5, 0.5]),
        )
    raise ValueError(
        f"Unknown noise layer type: {noise_type}. "
        "Expected one of: none, pimog, projector, mixed"
    )

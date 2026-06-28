"""Factory and composition utilities for degradation layers."""

import torch
import torch.nn as nn

from .PIMoG_Layer import PIMoGLayer
from .Projector_Layer import ProjectorSimulator
from .OLED_Layer import OLEDNoiseLayer
from .LED_Layer import LEDNoiseLayer


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


def _build_single_noise_layer(noise_type, noise_cfg):
    """Build one concrete degradation layer from the ``noise_layer`` section."""
    if noise_type == "pimog":
        return PIMoGLayer(**noise_cfg.get("pimog", {}))
    if noise_type == "oled":
        return OLEDNoiseLayer(**noise_cfg.get("oled", {}))
    if noise_type == "led":
        return LEDNoiseLayer(**noise_cfg.get("led", {}))
    if noise_type == "projector":
        return ProjectorSimulator(**noise_cfg.get("projector", {}))
    raise ValueError(f"Unsupported mixed noise layer candidate: {noise_type}")


def build_noise_layer(config):
    """Build ``none``, concrete screen layers, or ``mixed`` from a config dict."""
    noise_cfg = config.get("noise_layer", {})
    noise_type = get_noise_layer_type(config)

    if noise_type == "none":
        return nn.Identity()
    if noise_type in {"pimog", "oled", "led", "projector"}:
        return _build_single_noise_layer(noise_type, noise_cfg)
    if noise_type == "mixed":
        mixed_cfg = noise_cfg.get("mixed", {})
        candidates = mixed_cfg.get("candidates", None)
        if candidates is None:
            # Backward compatible default for existing configs with
            # mixed_probs: [pimog_prob, projector_prob].
            candidates = ["pimog", "projector"]
            probs = noise_cfg.get("mixed_probs", [0.5, 0.5])
        else:
            candidates = [str(candidate).lower() for candidate in candidates]
            probs = mixed_cfg.get("probs", noise_cfg.get("mixed_probs", None))
        return MixedNoiseLayer(
            layers=[_build_single_noise_layer(candidate, noise_cfg) for candidate in candidates],
            probs=probs,
        )
    raise ValueError(
        f"Unsupported noise layer type: {noise_type}. "
        "Expected one of: none, pimog, oled, led, projector, mixed"
    )

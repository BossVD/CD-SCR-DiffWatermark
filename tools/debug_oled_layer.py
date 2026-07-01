import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from NOISE_LAYER.OLED_Layer import OLED_Layer


def check(name, x):
    finite = torch.isfinite(x).all().item()
    finite_x = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    print(
        f"{name} finite={finite} "
        f"nan={torch.isnan(x).sum().item()} "
        f"inf={torch.isinf(x).sum().item()} "
        f"min={finite_x.min().item():.6f} "
        f"max={finite_x.max().item():.6f}"
    )
    return finite


def run_case(name, cfg, device, batch_size, image_size):
    print(f"--- {name} ---")
    layer = OLED_Layer(cfg).to(device)
    x = torch.rand(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
        requires_grad=True,
    )

    y = layer(x)
    output_finite = check("output", y)

    loss = y.mean()
    loss.backward()
    grad_finite = check("input_grad", x.grad)

    if not output_finite or not grad_finite:
        raise FloatingPointError(f"{name} produced non-finite output or input grad")


def main():
    parser = argparse.ArgumentParser(description="Debug OLED layer forward/backward")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()

    base_cfg = {
        "p": 1.0,
        "train_safe": True,
        "debug_finite": True,
        "sensor_noise_eps": 1e-4,
        "final_nan_to_num": True,
        "enable_banding": False,
        "enable_reflection": False,
        "enable_resample": False,
        "use_jpeg": False,
        "noise_std_range": [0.001, 0.010],
    }

    cases = [
        ("train_safe_noise_off", {**base_cfg, "enable_noise": False}),
        ("train_safe_noise_on", {**base_cfg, "enable_noise": True}),
        (
            "train_safe_perspective_off",
            {**base_cfg, "enable_noise": True, "enable_perspective": False},
        ),
        (
            "train_safe_perspective_on",
            {
                **base_cfg,
                "enable_noise": True,
                "enable_perspective": True,
                "perspective_strength_range": [0.0, 0.008],
            },
        ),
        (
            "manual_safe",
            {
                **base_cfg,
                "train_safe": False,
                "enable_noise": True,
                "enable_perspective": False,
                "enable_banding": False,
                "enable_reflection": False,
            },
        ),
    ]

    for name, cfg in cases:
        run_case(name, cfg, args.device, args.batch_size, args.image_size)

    print("OLED debug completed without non-finite values.")


if __name__ == "__main__":
    main()

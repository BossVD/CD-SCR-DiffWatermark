"""Visual and numerical smoke test for each concrete degradation layer."""

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from NOISE_LAYER.build_noise_layer import build_noise_layer


def main():
    parser = argparse.ArgumentParser(description="Test degradation layers")
    parser.add_argument("--input", default=None, help="Optional sample image path")
    parser.add_argument("--config", default="configs/watermark_diffusion.yaml")
    parser.add_argument("--output_dir", default="outputs/noise_layer_debug")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--projector_samples", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8-sig") as handle:
        cfg = yaml.safe_load(handle)
    image_size = args.image_size or cfg.get("data", {}).get("image_size", 128)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    if args.input:
        x = transform(Image.open(args.input).convert("RGB")).unsqueeze(0).to(device)
    else:
        # A deterministic RGB gradient keeps the script usable in a fresh repo.
        axis = torch.linspace(0.0, 1.0, image_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        x = torch.stack((xx, yy, (xx + yy) / 2.0)).unsqueeze(0).to(device)
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = {"original": x}

    # Mixed only selects one concrete layer per call. Showing it here would
    # duplicate a randomly re-sampled PIMoG or Projector result and make the
    # visual comparison ambiguous.
    for noise_type in ("pimog", "oled", "led", "projector"):
        test_cfg = dict(cfg)
        test_cfg["noise_layer"] = dict(cfg.get("noise_layer", {}), type=noise_type)
        test_cfg["noise_layer"]["pimog"] = dict(
            cfg.get("noise_layer", {}).get("pimog", {}), p=1.0
        )
        test_cfg["noise_layer"]["oled"] = dict(
            cfg.get("noise_layer", {}).get("oled", {}), p=1.0
        )
        test_cfg["noise_layer"]["led"] = dict(
            cfg.get("noise_layer", {}).get("led", {}), p=1.0
        )
        test_cfg["noise_layer"]["projector"] = dict(
            cfg.get("noise_layer", {}).get("projector", {}), p=1.0
        )
        layer = build_noise_layer(test_cfg).to(device)
        y = layer(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        assert y.min() >= 0 and y.max() <= 1
        outputs[noise_type] = y
        save_image(y[0], os.path.join(args.output_dir, f"{noise_type}_deg.png"))
        if noise_type == "projector" and args.projector_samples > 1:
            samples = [y]
            for _ in range(args.projector_samples - 1):
                samples.append(layer(x))
            save_image(
                torch.cat(samples, dim=0),
                os.path.join(args.output_dir, "projector_samples.png"),
                nrow=args.projector_samples,
            )

    save_image(x[0], os.path.join(args.output_dir, "original.png"))
    comparison = torch.cat(list(outputs.values()), dim=0)
    save_image(
        comparison,
        os.path.join(args.output_dir, "compare.png"),
        nrow=len(outputs),
    )
    print(f"Noise-layer visualization saved to {args.output_dir}")


if __name__ == "__main__":
    main()

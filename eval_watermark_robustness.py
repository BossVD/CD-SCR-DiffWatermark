"""
Evaluate watermark robustness on a validation set.

Tests multiple degradation levels and outputs CSV results.

Usage:
    D:\Anaconda_envs\envs\wadiff\python.exe eval_watermark_robustness.py \
        --checkpoint checkpoints/best.pt \
        --data_dir ./data/val \
        --output ./outputs/eval_results.csv
"""
import os
import sys
import argparse
import csv
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image


from guided_diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType

from dataset.watermark_image_dataset import WatermarkImageDataset
from models.watermark_unet import WatermarkConditionedUNet
from models.watermark_decoder import (
    build_watermark_decoder,
    load_watermark_decoder_state,
)
from NOISE_LAYER import build_noise_layer


def compute_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return (20 * math.log10(max_val) - 10 * math.log10(mse.item()))


def load_yaml_config(config_path):
    if not config_path:
        return None
    import yaml
    with open(config_path, 'r', encoding='utf-8-sig') as handle:
        return yaml.safe_load(handle)


def parse_noise_layers(value):
    layers = [item.strip().lower() for item in value.split(',') if item.strip()]
    valid = {'clean', 'pimog', 'oled', 'led', 'projector', 'mixed'}
    invalid = [layer for layer in layers if layer not in valid]
    if invalid:
        raise ValueError(f"Unsupported noise layer(s): {invalid}")
    return layers


def build_eval_noise_layer(cfg, noise_type, device):
    eval_cfg = dict(cfg)
    eval_cfg['noise_layer'] = dict(cfg.get('noise_layer', {}), type=noise_type)
    if noise_type in {'pimog', 'oled', 'led', 'projector'}:
        eval_cfg['noise_layer'][noise_type] = dict(
            cfg.get('noise_layer', {}).get(noise_type, {}),
            p=1.0,
        )
    simulator = build_noise_layer(eval_cfg).to(device)
    simulator.eval()
    return simulator


def embed_watermark_eval(diffusion, model, cover_img, wm_bits, t_start=300):
    """Same as embed_watermark in train script, but for evaluation."""
    device = cover_img.device
    B = cover_img.size(0)

    t = torch.full((B,), t_start - 1, device=device, dtype=torch.long)
    noise = torch.randn_like(cover_img)
    x_t = diffusion.q_sample(cover_img, t, noise=noise)

    for step in reversed(range(t_start)):
        t_batch = torch.full((B,), step, device=device, dtype=torch.long)
        t_scaled = t_batch.float() * (1000.0 / diffusion.num_timesteps)

        pred_noise = model(
            x_t=x_t,
            t=t_scaled,
            cover_img=cover_img,
            wm_bits=wm_bits,
        )

        out = diffusion.p_mean_variance(
            model=lambda *a, **kw: pred_noise,
            x=x_t,
            t=t_batch,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs={},
        )
        mean = out['mean']
        log_variance = out['log_variance']
        noise_term = torch.randn_like(x_t) if step > 0 else torch.zeros_like(x_t)
        x_t = mean + torch.exp(0.5 * log_variance) * noise_term

    return x_t.clamp(-1, 1)


def main():
    parser = argparse.ArgumentParser(description='Evaluate watermark robustness')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.pt)')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML config path; overrides checkpoint config')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to validation data directory; defaults to config data.val_dir')
    parser.add_argument('--output', type=str, default='./outputs/eval_results.csv',
                        help='Output CSV path')
    parser.add_argument('--noise_layers', type=str,
                        default='clean,pimog,oled,led,projector,mixed',
                        help='Comma-separated layers to evaluate')
    parser.add_argument('--t_start', type=int, default=300,
                        help='Timestep to start reverse from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed; defaults to the checkpoint training seed')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Using device: {device}")

    # --- Load checkpoint ---
    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = checkpoint.get('config', {})
    config_override = load_yaml_config(args.config)
    if config_override is not None:
        cfg = config_override

    seed = args.seed if args.seed is not None else cfg.get('train', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Eval] Random seed: {seed}")

    image_size = cfg.get('data', {}).get('image_size', 128)
    watermark_length = cfg.get('data', {}).get('watermark_length', 64)
    base_channels = cfg.get('model', {}).get('base_channels', 64)
    cond_dim = cfg.get('model', {}).get('cond_dim', 256)
    timesteps = cfg.get('diffusion', {}).get('timesteps', 1000)
    beta_schedule = cfg.get('diffusion', {}).get('beta_schedule', 'linear')

    print(f"[Eval] image_size={image_size}, watermark_length={watermark_length}")

    # --- Create diffusion ---
    betas = get_named_beta_schedule(beta_schedule, timesteps)
    diffusion = GaussianDiffusion(
        betas=torch.tensor(betas, dtype=torch.float32),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )

    # --- Create model and decoder ---
    model = WatermarkConditionedUNet(
        image_size=image_size,
        base_channels=base_channels,
        cond_dim=cond_dim,
        watermark_length=watermark_length,
    ).to(device)

    if 'diffusion_model' in checkpoint:
        model.load_state_dict(checkpoint['diffusion_model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    decoder = build_watermark_decoder(
        cfg,
        watermark_length=watermark_length,
    ).to(device)
    if 'decoder' in checkpoint:
        missing, unexpected, mismatched = load_watermark_decoder_state(
            decoder, checkpoint['decoder']
        )
        if missing or unexpected or mismatched:
            print(
                "[Eval] Decoder checkpoint partially loaded "
                "(architecture may have changed)."
            )
    decoder.eval()

    # --- Dataset ---
    data_dir = args.data_dir or cfg.get('data', {}).get('val_dir')
    if not data_dir:
        raise ValueError("Validation data directory is required via --data_dir or config data.val_dir")
    dataset = WatermarkImageDataset(
        data_dir=data_dir,
        image_size=image_size,
        watermark_length=watermark_length,
        watermark_seed=cfg.get('data', {}).get('watermark_seed', 42),
        watermark_mode='fixed',
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True,
    )
    print(f"[Eval] Dataset size: {len(dataset)}")

    requested_layers = parse_noise_layers(args.noise_layers)
    simulators = {}
    for noise_type in requested_layers:
        if noise_type == 'clean':
            simulators[noise_type] = None
        else:
            simulators[noise_type] = build_eval_noise_layer(cfg, noise_type, device)
    print(f"[Eval] Noise layers: {', '.join(requested_layers)}")

    # --- Evaluate ---
    results = defaultdict(list)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sample_dir = os.path.join(output_dir, 'eval_samples') if output_dir else './eval_samples'
    os.makedirs(sample_dir, exist_ok=True)
    sample_count = 0
    max_samples = 20  # Save up to N sample comparisons

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            cover_img = batch['image'].to(device)
            wm_bits = batch['wm_bits'].to(device)

            # Generate watermarked images
            watermarked = embed_watermark_eval(
                diffusion, model, cover_img, wm_bits, t_start=args.t_start
            )

            watermarked_01 = (watermarked + 1.0) / 2.0
            cover_01 = (cover_img + 1.0) / 2.0

            # Test each degradation level
            for level_name, simulator in simulators.items():
                if simulator is not None:
                    degraded_01 = simulator(watermarked_01).float()
                    degraded = degraded_01.mul(2.0).sub(1.0)
                else:
                    degraded_01 = watermarked_01
                    degraded = watermarked

                logits = decoder(degraded)
                pred_bits = (torch.sigmoid(logits) > 0.5).float()
                psnr_val = compute_psnr(degraded_01, cover_01, max_val=1.0)
                l1_val = F.l1_loss(degraded, cover_img).item()

                # Per-sample accuracy
                for i in range(cover_img.size(0)):
                    acc = (pred_bits[i] == wm_bits[i]).float().mean().item()
                    results[f'bit_acc_{level_name}'].append(acc)
                    results[f'ber_{level_name}'].append(1.0 - acc)
                    results[f'psnr_{level_name}'].append(psnr_val)
                    results[f'l1_{level_name}'].append(l1_val)

            # Save some samples
            if sample_count < max_samples:
                for i in range(min(2, cover_img.size(0))):
                    idx = sample_count + i
                    if idx >= max_samples:
                        break
                    save_image(cover_01[i], os.path.join(sample_dir, f'{idx:04d}_cover.png'))
                    save_image(watermarked_01[i], os.path.join(sample_dir, f'{idx:04d}_watermarked.png'))

                    # Save degraded versions too
                    for level_name, simulator in simulators.items():
                        if simulator is not None:
                            degraded_01 = simulator(watermarked_01[i:i+1]).float()
                            save_image(degraded_01[0], os.path.join(
                                sample_dir, f'{idx:04d}_degraded_{level_name}.png'))

                sample_count += min(2, cover_img.size(0))

            if (batch_idx + 1) % 10 == 0:
                print(f"[Eval] Processed {(batch_idx + 1) * args.batch_size} images...")

    # --- Compute aggregate statistics ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    summary = {}
    for level_name in requested_layers:
        bit_key = f'bit_acc_{level_name}'
        ber_key = f'ber_{level_name}'
        psnr_key = f'psnr_{level_name}'
        if not results[bit_key]:
            continue
        bit_acc = sum(results[bit_key]) / len(results[bit_key])
        ber = sum(results[ber_key]) / len(results[ber_key])
        psnr = sum(results[psnr_key]) / len(results[psnr_key])
        summary[bit_key] = bit_acc
        summary[ber_key] = ber
        summary[psnr_key] = psnr
        print(
            f"[Eval] {level_name:9s}: "
            f"bit_acc={bit_acc:.4f}, BER={ber:.4f}, PSNR={psnr:.2f}"
        )

    # --- Save CSV ---
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, val in summary.items():
            writer.writerow([key, val])

    print(f"\n[Eval] Results saved to: {args.output}")
    print(f"[Eval] Sample images saved to: {sample_dir}")

    # --- Also save per-image results ---
    per_image_path = args.output.replace('.csv', '_per_image.csv')
    with open(per_image_path, 'w', newline='') as f:
        n_images = len(next(iter(results.values())))
        fieldnames = list(results.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_images):
            row = {k: results[k][i] for k in fieldnames}
            writer.writerow(row)

    print(f"[Eval] Per-image results saved to: {per_image_path}")


if __name__ == '__main__':
    main()



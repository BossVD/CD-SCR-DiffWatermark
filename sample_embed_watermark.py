"""
Sample watermarked images using the trained WatermarkConditionedUNet.

Given a cover image and watermark bits, produces a watermarked image
via image-to-image diffusion (partial forward + full reverse).

Usage:
    D:\Anaconda_envs\envs\wadiff\python.exe sample_embed_watermark.py \
        --checkpoint checkpoints/best.pt \
        --input ./test_images/cover.png \
        --watermark "1010101011001010" \
        --output ./outputs/watermarked.png \
        --t_start 300
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


from guided_diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType

from models.watermark_unet import WatermarkConditionedUNet
from models.watermark_decoder import WatermarkDecoder


def embed_watermark_sample(diffusion, model, cover_img, wm_bits, t_start=300):
    """
    Image-to-image watermark embedding via partial DDPM reverse sampling.

    Args:
        diffusion: GaussianDiffusion instance
        model: WatermarkConditionedUNet
        cover_img: [1, 3, H, W] in [-1, 1]
        wm_bits:  [1, wm_len] 0/1 float
        t_start:  timestep to start reverse from

    Returns:
        watermarked: [1, 3, H, W] in [-1, 1]
    """
    device = cover_img.device
    B = cover_img.size(0)

    # Forward diffuse to t_start
    t = torch.full((B,), t_start - 1, device=device, dtype=torch.long)
    noise = torch.randn_like(cover_img)
    x_t = diffusion.q_sample(cover_img, t, noise=noise)

    # Reverse denoise
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
    parser = argparse.ArgumentParser(description='Embed watermark into cover image')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.pt)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input cover image')
    parser.add_argument('--watermark', type=str, default=None,
                        help='Watermark bits as binary string, e.g. "10101010". If None, random.')
    parser.add_argument('--watermark_length', type=int, default=64,
                        help='Number of watermark bits (used if --watermark not provided)')
    parser.add_argument('--output', type=str, default='./outputs/watermarked.png',
                        help='Output path for watermarked image')
    parser.add_argument('--t_start', type=int, default=300,
                        help='Timestep to start reverse from (controls edit strength)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed; defaults to the checkpoint training seed')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Sample] Using device: {device}")

    # --- Load checkpoint ---
    print(f"[Sample] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = checkpoint.get('config', {})

    seed = args.seed if args.seed is not None else cfg.get('train', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Sample] Random seed: {seed}")

    # --- Determine watermark length (ALWAYS from checkpoint config) ---
    if cfg and 'data' in cfg and 'watermark_length' in cfg['data']:
        watermark_length = cfg['data']['watermark_length']
    elif args.watermark is not None:
        watermark_length = len(args.watermark)
    else:
        watermark_length = args.watermark_length
    print(f"[Sample] Model trained with watermark_length={watermark_length}")

    image_size = cfg.get('data', {}).get('image_size', 128)
    base_channels = cfg.get('model', {}).get('base_channels', 64)
    cond_dim = cfg.get('model', {}).get('cond_dim', 256)
    timesteps = cfg.get('diffusion', {}).get('timesteps', 1000)
    beta_schedule = cfg.get('diffusion', {}).get('beta_schedule', 'linear')

    print(f"[Sample] image_size={image_size}, watermark_length={watermark_length}")

    # --- Create diffusion ---
    betas = get_named_beta_schedule(beta_schedule, timesteps)
    diffusion = GaussianDiffusion(
        betas=torch.tensor(betas, dtype=torch.float32),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )

    # --- Create model ---
    model = WatermarkConditionedUNet(
        image_size=image_size,
        base_channels=base_channels,
        cond_dim=cond_dim,
        watermark_length=watermark_length,
        use_pretrained_unet=False,
        pretrained_path=None,
    ).to(device)

    # Load weights
    if 'diffusion_model' in checkpoint:
        model.load_state_dict(checkpoint['diffusion_model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("[Sample] Model loaded.")

    # --- Create decoder for verification ---
    decoder = WatermarkDecoder(watermark_length=watermark_length).to(device)
    if 'decoder' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder'], strict=False)
    decoder.eval()

    # --- Load and preprocess cover image ---
    print(f"[Sample] Loading cover image: {args.input}")
    transform = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    cover_img = Image.open(args.input).convert("RGB")
    cover_tensor = transform(cover_img).unsqueeze(0).to(device)  # [1, 3, H, W]

    # --- Create watermark bits ---
    if args.watermark is not None:
        wm_bits = torch.tensor([float(b) for b in args.watermark], device=device).unsqueeze(0)
        # Pad or truncate to match expected length
        if wm_bits.size(1) < watermark_length:
            pad = torch.zeros(1, watermark_length - wm_bits.size(1), device=device)
            wm_bits = torch.cat([wm_bits, pad], dim=1)
        elif wm_bits.size(1) > watermark_length:
            wm_bits = wm_bits[:, :watermark_length]
        print(f"[Sample] Watermark bits: {args.watermark}")
    else:
        wm_bits = torch.randint(0, 2, (1, watermark_length), device=device, dtype=torch.float32)
        print(f"[Sample] Random watermark: {''.join(str(int(b)) for b in wm_bits[0].tolist())}")

    # --- Embed watermark ---
    print(f"[Sample] Embedding watermark (t_start={args.t_start})...")
    with torch.no_grad():
        watermarked = embed_watermark_sample(
            diffusion, model, cover_tensor, wm_bits, t_start=args.t_start
        )

    # --- Verify ---
    watermarked_01 = (watermarked + 1.0) / 2.0
    logits = decoder(watermarked)
    pred_bits = (torch.sigmoid(logits) > 0.5).float()
    bit_acc = (pred_bits == wm_bits).float().mean().item()
    print(f"[Sample] Recovered bit accuracy: {bit_acc:.4f}")
    print(f"[Sample] Recovered bits: {''.join(str(int(b)) for b in pred_bits[0].tolist())}")

    # --- Save ---
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    save_image(watermarked_01[0], args.output)

    # Also save cover + watermarked comparison
    comparison_dir = os.path.join(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    cover_01 = (cover_tensor + 1.0) / 2.0
    comparison = torch.cat([cover_01, watermarked_01], dim=0)
    base_name = os.path.splitext(os.path.basename(args.output))[0]
    save_image(comparison, os.path.join(comparison_dir, f'{base_name}_comparison.png'), nrow=1)

    print(f"[Sample] Watermarked image saved to: {args.output}")
    print(f"[Sample] Comparison saved to: {comparison_dir}")


if __name__ == '__main__':
    main()



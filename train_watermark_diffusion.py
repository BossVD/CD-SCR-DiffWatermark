"""
Train Watermark-Conditioned Image-to-Image Diffusion Model.

Core training logic:
  1. Two timestep ranges: t_diff (full) for noise prediction, t_wm (small) for watermark loss
  2. loss_wm backpropagates through the ENTIRE computation graph (no .detach())
  3. Image range discipline: [-1,1] for diffusion/decoder, [0,1] for degradations
  4. Unified none/PIMoG/projector/mixed degradation construction

KEY DEBUG POINTS if bit_acc ~ 0.5:
  1. Check wm_bits are actually fed into U-Net (watermark_mlp)
  2. Check watermark_mlp parameters have requires_grad=True
  3. Check loss_wm backprop reaches diffusion_model (no .detach() on pred_x0)
  4. Check decoder input range is [-1, 1]
  5. Check lambda_wm is not too small
  6. Check wm_t_max is not too large

Usage:
    D:\Anaconda_envs\envs\wadiff\python.exe train_watermark_diffusion.py --config configs/watermark_stage1.yaml
r"""
import os
import sys
import argparse
import csv
import glob
import math
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.utils import save_image

from guided_diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.nn import mean_flat

from dataset.watermark_image_dataset import WatermarkImageDataset
from models.watermark_unet import WatermarkConditionedUNet
from models.watermark_decoder import (
    build_watermark_decoder,
    load_watermark_decoder_state,
)
from NOISE_LAYER import build_noise_layer, get_noise_layer_type

# ============================================================
# Helper: PSNR computation
# ============================================================
def compute_psnr(pred, target, max_val=1.0):
    """Compute PSNR in [0, max_val] range."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return (20 * math.log10(max_val) - 10 * math.log10(mse.item()))

# ============================================================
# Helper: predict x0 from noise prediction
# ============================================================
def predict_start_from_noise(diffusion, x_t, t, noise_pred):
    """Wrapper around GaussianDiffusion._predict_xstart_from_eps."""
    return diffusion._predict_xstart_from_eps(x_t, t, noise_pred)


def set_random_seed(seed, deterministic=True):
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """Give every DataLoader worker a reproducible Python/NumPy RNG state."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_random_state(train_generator):
    numpy_state = np.random.get_state()
    state = {
        'python': random.getstate(),
        'numpy': {
            'bit_generator': numpy_state[0],
            'state': torch.from_numpy(numpy_state[1].copy()),
            'pos': numpy_state[2],
            'has_gauss': numpy_state[3],
            'cached_gaussian': numpy_state[4],
        },
        'torch': torch.get_rng_state(),
        'train_generator': train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state, train_generator):
    if not state:
        return
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((
        numpy_state['bit_generator'],
        numpy_state['state'].cpu().numpy(),
        numpy_state['pos'],
        numpy_state['has_gauss'],
        numpy_state['cached_gaussian'],
    ))
    # Checkpoints are loaded with map_location=device, which can move these
    # CPU generator states onto CUDA. PyTorch RNG restore APIs require CPU
    # ByteTensors even when restoring CUDA generator states.
    torch.set_rng_state(state['torch'].detach().cpu().to(torch.uint8))
    train_generator.set_state(
        state['train_generator'].detach().cpu().to(torch.uint8)
    )
    if torch.cuda.is_available() and 'cuda' in state:
        cuda_states = [
            rng_state.detach().cpu().to(torch.uint8)
            for rng_state in state['cuda']
        ]
        torch.cuda.set_rng_state_all(cuda_states)


# ============================================================
# Configuration loading
# ============================================================
def load_config(config_path):
    """Load YAML config or return defaults."""
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                return yaml.safe_load(f)
        except ImportError:
            print("[WARNING] PyYAML not installed; using default config.")
    return default_config()

def default_config():
    return {
        'data': {
            'train_dir': './data/train',
            'val_dir': './data/val',
            'image_size': 128,
            'watermark_length': 64,
            'watermark_seed': 42,
            'train_watermark_mode': 'per_epoch',
            'val_watermark_mode': 'fixed',
            'max_train_images': 10000,
            'max_val_images': 1000,
        },
        'model': {
            'base_channels': 64,
            'cond_dim': 256,
            'use_pretrained_unet': False,
            'pretrained_path': None,
        },
        'decoder': {
            'type': 'residual_multiscale',
            'base_channels': 32,
            'hidden_dim': 512,
            'dropout': 0.1,
            'norm_groups': 8,
            'use_multiscale': True,
        },
        'diffusion': {
            'timesteps': 1000,
            'beta_schedule': 'linear',
            'wm_t_min': 0,
            'wm_t_max': 200,
            'train_t_start': 200,
            'sample_steps': 100,
        },
        'train': {
            'seed': 42,
            'deterministic': True,
            'batch_size': 32,
            'num_workers': 8,
            'lr': 1e-4,
            'epochs': 10,
            'device': 'cuda',
            'use_amp': True,
            'lambda_diff': 1.0,
            'lambda_img': 1.0,
            'lambda_wm': 5.0,
            'save_interval': 2,
            'sample_interval': 5000,
            'log_interval': 100,
            'reset_decoder': False,
        },
        'noise_layer': {'type': 'none'},
        'output': {
            'checkpoint_dir': './checkpoints',
            'sample_dir': './outputs/samples',
            'log_dir': './outputs/logs',
        },
    }

# ============================================================
# embed_watermark: Full DDPM reverse sampling for image-to-image
# ============================================================
@torch.no_grad()
def embed_watermark(diffusion, model, cover_img, wm_bits, t_start=300):
    """
    Image-to-image watermark embedding via partial DDPM reverse sampling.

    Strategy:
      1. Add noise to cover_img up to t_start
      2. Reverse-denoise with cover_img + wm_bits as conditions
      3. Return watermarked image in [-1, 1]

    Args:
        diffusion: GaussianDiffusion instance (for schedules)
        model: WatermarkConditionedUNet
        cover_img: [B, 3, H, W] in [-1, 1]
        wm_bits:  [B, wm_len] 0/1 float
        t_start:  timestep to start reverse from (controls edit strength)

    Returns:
        watermarked: [B, 3, H, W] in [-1, 1]
    """
    device = cover_img.device
    B = cover_img.size(0)

    # 1. Forward diffuse to t_start
    t = torch.full((B,), t_start - 1, device=device, dtype=torch.long)
    noise = torch.randn_like(cover_img)
    x_t = diffusion.q_sample(cover_img, t, noise=noise)

    # 2. Reverse denoise step by step
    for step in reversed(range(t_start)):
        t_batch = torch.full((B,), step, device=device, dtype=torch.long)

        # Scale timesteps for model
        t_scaled = t_batch.float() * (1000.0 / diffusion.num_timesteps)

        pred_noise = model(
            x_t=x_t,
            t=t_scaled,
            cover_img=cover_img,
            wm_bits=wm_bits,
        )

        # Use DDPM sampling step
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

        # Sample x_{t-1}
        noise_term = torch.randn_like(x_t) if step > 0 else torch.zeros_like(x_t)
        x_t = mean + torch.exp(0.5 * log_variance) * noise_term

    watermarked = x_t.clamp(-1, 1)
    return watermarked

# ============================================================
# Main training function
# ============================================================
def train(config):
    cfg = config

    seed = cfg['train'].get('seed', 42)
    deterministic = cfg['train'].get('deterministic', True)
    set_random_seed(seed, deterministic=deterministic)
    print(f"[Train] Random seed: {seed}, deterministic={deterministic}")

    # --- Device ---
    device_str = cfg['train'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Using device: {device}")

    # --- Create output directories ---
    checkpoint_dir = cfg['output']['checkpoint_dir']
    sample_dir = cfg['output']['sample_dir']
    log_dir = cfg['output']['log_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- Dataset ---
    image_size = cfg['data']['image_size']
    watermark_length = cfg['data']['watermark_length']
    watermark_seed = cfg['data'].get('watermark_seed', seed)
    batch_size = cfg['train']['batch_size']
    num_workers = cfg['train']['num_workers']

    train_dataset = WatermarkImageDataset(
        data_dir=cfg['data']['train_dir'],
        image_size=image_size,
        watermark_length=watermark_length,
        watermark_seed=watermark_seed,
        watermark_mode=cfg['data'].get('train_watermark_mode', 'per_epoch'),
        is_train=True,
        max_images=cfg['data'].get('max_train_images', None),
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

    val_dataset = WatermarkImageDataset(
        data_dir=cfg['data']['val_dir'],
        image_size=image_size,
        watermark_length=watermark_length,
        watermark_seed=watermark_seed,
        watermark_mode=cfg['data'].get('val_watermark_mode', 'fixed'),
        is_train=False,
        max_images=cfg['data'].get('max_val_images', None),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    print(f"[Train] Train set: {len(train_dataset)} images, Val set: {len(val_dataset)} images")

    # --- Training scale summary ---
    total_train_available = len(glob.glob(os.path.join(cfg['data']['train_dir'], '*.jpg'))) + \
                           len(glob.glob(os.path.join(cfg['data']['train_dir'], '*.png'))) + \
                           len(glob.glob(os.path.join(cfg['data']['train_dir'], '*.jpeg')))
    total_val_available = len(glob.glob(os.path.join(cfg['data']['val_dir'], '*.jpg'))) + \
                         len(glob.glob(os.path.join(cfg['data']['val_dir'], '*.png'))) + \
                         len(glob.glob(os.path.join(cfg['data']['val_dir'], '*.jpeg')))
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * cfg['train']['epochs']
    noise_type = get_noise_layer_type(cfg)
    print(f"[Scale] Train images: {len(train_dataset)} / {total_train_available}")
    print(f"[Scale] Val images:   {len(val_dataset)} / {total_val_available}")
    print(f"[Scale] Batch size: {batch_size}")
    print(f"[Scale] Steps per epoch: {steps_per_epoch}")
    print(f"[Scale] Epochs: {cfg['train']['epochs']}")
    print(f"[Scale] Total training steps: {total_steps}")
    print(f"[Scale] Noise layer: {noise_type}")

    # --- Diffusion ---
    timesteps = cfg['diffusion']['timesteps']
    betas = get_named_beta_schedule(cfg['diffusion']['beta_schedule'], timesteps)
    diffusion = GaussianDiffusion(
        betas=torch.tensor(betas, dtype=torch.float32),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )

    # --- Model ---
    model_cfg = cfg['model']
    model = WatermarkConditionedUNet(
        image_size=image_size,
        base_channels=model_cfg['base_channels'],
        cond_dim=model_cfg['cond_dim'],
        watermark_length=watermark_length,
        use_pretrained_unet=model_cfg['use_pretrained_unet'],
        pretrained_path=model_cfg['pretrained_path'],
    ).to(device)

    # --- Watermark Decoder ---
    decoder = build_watermark_decoder(
        cfg,
        watermark_length=watermark_length,
    ).to(device)
    decoder_cfg = cfg.get('decoder', {})
    print(
        "[Decoder] type={}, base_channels={}, hidden_dim={}, multiscale={}".format(
            decoder_cfg.get('type', 'residual_multiscale'),
            decoder_cfg.get('base_channels', 32),
            decoder_cfg.get('hidden_dim', 512),
            decoder_cfg.get('use_multiscale', True),
        )
    )

    # Unified layers consume and return [0, 1]. The diffusion model and
    # decoder retain their existing [-1, 1] contracts around this boundary.
    noise_layer = build_noise_layer(cfg).to(device)
    use_noise_layer = noise_type != 'none'
    print(f"[NoiseLayer] type: {noise_type}")
    if noise_type == 'mixed':
        noise_cfg = cfg.get('noise_layer', {})
        mixed_cfg = noise_cfg.get('mixed', {})
        mixed_candidates = mixed_cfg.get('candidates', ['pimog', 'projector'])
        mixed_probs = mixed_cfg.get('probs', noise_cfg.get('mixed_probs', [0.5, 0.5]))
        print(f"[NoiseLayer] layers: {', '.join(mixed_candidates)}")
        print(f"[NoiseLayer] probs: {mixed_probs}")
    elif use_noise_layer:
        print(f"[NoiseLayer] {noise_layer.__class__.__name__} enabled")

    # --- Optimizer ---
    # KEY: Both model and decoder are optimized together
    optimizer = AdamW(
        list(model.parameters()) + list(decoder.parameters()),
        lr=cfg['train']['lr'],
    )

    # --- AMP ---
    use_amp = cfg['train'].get('use_amp', False)
    amp_enabled = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if amp_enabled else None
    print(f"[Train] AMP autocast: {'enabled' if amp_enabled else 'disabled'}")

    # --- Resume from checkpoint (for Stage 2 continuation) ---
    resume_path = cfg.get('_resume_path', None)
    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        print(f"[Resume] Loading checkpoint from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['diffusion_model'])
        reset_decoder = cfg['train'].get('reset_decoder', False)
        if reset_decoder:
            print("[Resume] reset_decoder=true, skip loading old decoder weights.")
        elif 'decoder' in ckpt:
            missing, unexpected, mismatched = load_watermark_decoder_state(
                decoder, ckpt['decoder']
            )
            if missing or unexpected or mismatched:
                print(
                    "[Resume] Decoder structure changed, loaded compatible "
                    "weights with strict=False."
                )
                print(f"[Resume] Missing decoder keys: {missing}")
                print(f"[Resume] Unexpected decoder keys: {unexpected}")
                print(f"[Resume] Mismatched decoder keys: {mismatched}")
            else:
                print("[Resume] Decoder weights loaded.")

        if not reset_decoder and 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except ValueError as exc:
                print(
                    "[Resume] Optimizer state is incompatible with the current "
                    f"decoder; starting optimizer from scratch. Reason: {exc}"
                )
        elif reset_decoder:
            print("[Resume] reset_decoder=true, start optimizer from scratch.")
        if scaler is not None and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        print(f"[Resume] Restored epoch={ckpt['epoch']}, global_step={global_step}")
        if 'best_bit_acc' in ckpt:
            best_name = ckpt.get('best_metric_name', 'bit_acc_clean')
            print(
                f"[Resume] Previous best {best_name}="
                f"{ckpt['best_bit_acc']:.4f}"
            )
        restore_random_state(ckpt.get('random_state'), train_generator)
    else:
        if resume_path:
            print(f"[WARNING] Resume checkpoint not found: {resume_path}")
            print("[WARNING] Training from scratch.")
        global_step = 0

    # --- Loss weights ---
    lambda_diff = cfg['train']['lambda_diff']
    lambda_img = cfg['train']['lambda_img']
    lambda_wm = cfg['train']['lambda_wm']

    # --- Timestep config ---
    wm_t_min = cfg['diffusion']['wm_t_min']
    wm_t_max = cfg['diffusion']['wm_t_max']
    train_t_start = cfg['diffusion']['train_t_start']

    # --- Training state ---
    epochs = cfg['train']['epochs']
    save_interval = cfg['train']['save_interval']
    sample_interval = cfg['train']['sample_interval']
    log_interval = cfg['train']['log_interval']
    # --- CSV loggers ---
    train_log_path = os.path.join(log_dir, 'train_log.csv')
    val_log_path = os.path.join(log_dir, 'val_log.csv')

    csv_mode = 'a' if resume_path else 'w'
    train_csv = open(train_log_path, csv_mode, newline='')
    train_writer = csv.DictWriter(train_csv, fieldnames=[
        'epoch', 'global_step', 'loss_total', 'loss_diff', 'loss_img', 'loss_wm',
        'bit_acc', 'psnr', 'lr', 'noise_layer_type',
    ])
    if not os.path.exists(train_log_path) or os.path.getsize(train_log_path) == 0:
        train_writer.writeheader()

    val_csv = open(val_log_path, csv_mode, newline='')
    val_writer = csv.DictWriter(val_csv, fieldnames=[
        'epoch', 'global_step',
        'bit_acc_clean', 'bit_acc_degraded',
        'psnr', 'loss_wm_clean', 'loss_wm_degraded',
    ])
    if not os.path.exists(val_log_path) or os.path.getsize(val_log_path) == 0:
        val_writer.writeheader()

    # --- Training loop ---
    print(f"[Train] Starting training: {epochs} epochs, log_interval={log_interval}")
    print(f"[Train] lambda_diff={lambda_diff}, lambda_img={lambda_img}, lambda_wm={lambda_wm}")
    print(f"[Train] wm_t range: [{wm_t_min}, {wm_t_max}), noise_layer={noise_type}")

    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        decoder.train()
        noise_layer.train()

        for batch in train_loader:
            cover_img = batch['image'].to(device)    # [B, 3, H, W], [-1, 1]
            wm_bits = batch['wm_bits'].to(device)    # [B, wm_len], 0/1 float
            B = cover_img.size(0)
            optimizer.zero_grad(set_to_none=True)

            # ========================================================
            # Official PIMoG is either fully enabled or disabled.
            # ========================================================
            active_noise_type = noise_type

            # ========================================================
            # 1. Diffusion noise prediction loss (full timestep range)
            # ========================================================
            t_diff = torch.randint(0, timesteps, (B,), device=device).long()
            noise = torch.randn_like(cover_img)
            x_t_diff = diffusion.q_sample(cover_img, t_diff, noise=noise)

            t_diff_scaled = t_diff.float() * (1000.0 / timesteps)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred_noise = model(
                    x_t=x_t_diff,
                    t=t_diff_scaled,
                    cover_img=cover_img,
                    wm_bits=wm_bits,
                )
                loss_diff = F.mse_loss(pred_noise, noise)

            # Backpropagate this branch immediately so its large U-Net
            # activation graph is released before the watermark branch.
            diffusion_objective = lambda_diff * loss_diff
            if scaler is not None:
                scaler.scale(diffusion_objective).backward()
            else:
                diffusion_objective.backward()
            loss_diff = loss_diff.detach()
            del pred_noise, diffusion_objective, x_t_diff

            # ========================================================
            # 2. Watermark + image fidelity loss (small timestep range)
            # ========================================================
            # KEY: t_wm from a SMALL range so pred_x0 is meaningful
            t_wm = torch.randint(wm_t_min, wm_t_max, (B,), device=device).long()
            noise_wm = torch.randn_like(cover_img)
            x_t_wm = diffusion.q_sample(cover_img, t_wm, noise=noise_wm)

            t_wm_scaled = t_wm.float() * (1000.0 / timesteps)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred_noise_wm = model(
                    x_t=x_t_wm,
                    t=t_wm_scaled,
                    cover_img=cover_img,
                    wm_bits=wm_bits,
                )

                # KEY: NO .detach() — loss_wm backpropagates through the U-Net.
                pred_x0 = predict_start_from_noise(diffusion, x_t_wm, t_wm, pred_noise_wm)
                pred_x0 = pred_x0.clamp(-1, 1)

                # Image fidelity loss in [-1, 1]
                loss_img = F.l1_loss(pred_x0, cover_img)

                # Unified degradation layers use [0, 1].
                pred_x0_01 = (pred_x0 + 1.0) / 2.0

                attacked_01 = noise_layer(pred_x0_01).float()
                if noise_type == 'mixed':
                    active_noise_type = f"mixed:{noise_layer.get_last_name()}"
                decoder_input = attacked_01.mul(2.0).sub(1.0)

                pred_logits = decoder(decoder_input)
                loss_wm = F.binary_cross_entropy_with_logits(
                    pred_logits, wm_bits.float()
                )

            # ========================================================
            # 3. Total loss
            # ========================================================
            watermark_objective = lambda_img * loss_img + lambda_wm * loss_wm

            # ========================================================
            # 4. Backward
            # ========================================================
            if scaler is not None:
                scaler.scale(watermark_objective).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                watermark_objective.backward()
                optimizer.step()

            loss_total = (
                lambda_diff * loss_diff
                + lambda_img * loss_img.detach()
                + lambda_wm * loss_wm.detach()
            )

            # ========================================================
            # 5. Metrics (no_grad for logging)
            # ========================================================
            with torch.no_grad():
                pred_bits = (torch.sigmoid(pred_logits) > 0.5).float()
                bit_acc = (pred_bits == wm_bits).float().mean().item()

                cover_01 = (cover_img + 1.0) / 2.0
                psnr_val = compute_psnr(pred_x0_01, cover_01, max_val=1.0)

            # ========================================================
            # 6. Logging
            # ========================================================
            if global_step % log_interval == 0:
                log_data = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'loss_total': loss_total.item(),
                    'loss_diff': loss_diff.item(),
                    'loss_img': loss_img.item(),
                    'loss_wm': loss_wm.item(),
                    'bit_acc': bit_acc,
                    'psnr': psnr_val,
                    'lr': optimizer.param_groups[0]['lr'],
                    'noise_layer_type': active_noise_type,
                }
                train_writer.writerow(log_data)
                train_csv.flush()

                print(
                    f"[E{epoch:03d}|S{global_step:06d}] "
                    f"L={loss_total.item():.4f} "
                    f"(diff={loss_diff.item():.4f} img={loss_img.item():.4f} wm={loss_wm.item():.4f}) "
                    f"bit_acc={bit_acc:.3f} PSNR={psnr_val:.1f} "
                    f"noise_layer={noise_type}"
                )

            # ========================================================
            # 7. Periodic sampling
            # ========================================================
            if global_step % sample_interval == 0 and global_step > 0:
                model.eval()
                with torch.no_grad():
                    # Take a batch for sampling
                    sample_batch = next(iter(train_loader))
                    s_cover = sample_batch['image'][:4].to(device)
                    s_wm = sample_batch['wm_bits'][:4].to(device)

                    # Generate watermarked images via full reverse sampling
                    s_watermarked = embed_watermark(
                        diffusion, model, s_cover, s_wm,
                        t_start=train_t_start,
                    )

                    # Convert to [0, 1] for saving
                    s_cover_01 = (s_cover + 1.0) / 2.0
                    s_wm_01 = (s_watermarked + 1.0) / 2.0

                    s_degraded_01 = noise_layer(s_wm_01).float()
                    sample_noise_type = (
                        noise_layer.get_last_name() if noise_type == 'mixed' else noise_type
                    )
                    s_decoder_input = s_degraded_01.mul(2.0).sub(1.0)
                    s_logits = decoder(s_decoder_input)
                    s_bits = (torch.sigmoid(s_logits) > 0.5).float()
                    s_acc = (s_bits == s_wm).float().mean().item()

                    # Save comparison grid
                    comparison = torch.cat([s_cover_01, s_wm_01, s_degraded_01], dim=0)
                    save_path = os.path.join(
                        sample_dir,
                        f'step_{global_step:06d}_{sample_noise_type}_acc_{s_acc:.3f}.png',
                    )
                    save_image(comparison, save_path, nrow=4)
                    print(
                        f"[Sample] Saved {save_path} "
                        f"(noise={sample_noise_type}, bit_acc={s_acc:.3f})"
                    )

                    # Also save individual images
                    for i in range(min(4, s_cover.size(0))):
                        save_image(s_cover_01[i], os.path.join(
                            sample_dir, f'step_{global_step:06d}_cover_{i}.png'))
                        save_image(s_wm_01[i], os.path.join(
                            sample_dir, f'step_{global_step:06d}_watermarked_{i}.png'))
                        save_image(s_degraded_01[i], os.path.join(
                            sample_dir,
                            f'step_{global_step:06d}_degraded_{sample_noise_type}_{i}.png'))

                model.train()

            global_step += 1

        # ============================================================
        # End of epoch: Validation
        # ============================================================
        print("[Val] Using pred_x0 validation, not full embed_watermark sampling.")
        model.eval()
        decoder.eval()
        val_bit_acc_clean = []
        val_bit_acc_degraded = []
        val_psnr_list = []
        val_loss_wm_clean = []
        val_loss_wm_degraded = []

        with torch.no_grad():
            for v_batch in val_loader:
                v_cover = v_batch['image'].to(device)
                v_wm = v_batch['wm_bits'].to(device)
                B_v = v_cover.size(0)

                # ---- Single-step pred_x0 validation ----
                t_eval = torch.randint(wm_t_min, wm_t_max, (B_v,), device=device).long()
                noise_eval = torch.randn_like(v_cover)
                x_t_eval = diffusion.q_sample(v_cover, t_eval, noise=noise_eval)

                t_eval_scaled = t_eval.float() * (1000.0 / timesteps)
                pred_noise_eval = model(
                    x_t=x_t_eval,
                    t=t_eval_scaled,
                    cover_img=v_cover,
                    wm_bits=v_wm,
                )

                v_watermarked = predict_start_from_noise(
                    diffusion, x_t_eval, t_eval, pred_noise_eval
                )
                v_watermarked = v_watermarked.clamp(-1, 1)

                v_wm_01 = (v_watermarked + 1.0) / 2.0
                v_cover_01 = (v_cover + 1.0) / 2.0

                # Clean accuracy
                v_logits_clean = decoder(v_watermarked)
                v_loss_clean = F.binary_cross_entropy_with_logits(v_logits_clean, v_wm.float())
                v_bits_clean = (torch.sigmoid(v_logits_clean) > 0.5).float()
                v_acc_clean = (v_bits_clean == v_wm).float().mean().item()

                # Degraded accuracy
                if use_noise_layer:
                    v_degraded_01 = noise_layer(v_wm_01).float()
                    v_logits_deg = decoder(v_degraded_01.mul(2.0).sub(1.0))
                    v_loss_deg = F.binary_cross_entropy_with_logits(v_logits_deg, v_wm.float())
                    v_bits_deg = (torch.sigmoid(v_logits_deg) > 0.5).float()
                    v_acc_deg = (v_bits_deg == v_wm).float().mean().item()
                else:
                    v_acc_deg = v_acc_clean
                    v_loss_deg = v_loss_clean

                v_psnr = compute_psnr(v_wm_01, v_cover_01, max_val=1.0)

                val_bit_acc_clean.append(v_acc_clean)
                val_bit_acc_degraded.append(v_acc_deg)
                val_psnr_list.append(v_psnr)
                val_loss_wm_clean.append(v_loss_clean.item())
                val_loss_wm_degraded.append(v_loss_deg.item() if isinstance(v_loss_deg, torch.Tensor) else v_loss_deg)

        # Average validation metrics
        avg_acc_clean = sum(val_bit_acc_clean) / len(val_bit_acc_clean)
        avg_acc_deg = sum(val_bit_acc_degraded) / len(val_bit_acc_degraded)
        avg_psnr = sum(val_psnr_list) / len(val_psnr_list)
        avg_loss_clean = sum(val_loss_wm_clean) / len(val_loss_wm_clean)
        avg_loss_deg = sum(val_loss_wm_degraded) / len(val_loss_wm_degraded)

        val_log_data = {
            'epoch': epoch,
            'global_step': global_step,
            'bit_acc_clean': avg_acc_clean,
            'bit_acc_degraded': avg_acc_deg,
            'psnr': avg_psnr,
            'loss_wm_clean': avg_loss_clean,
            'loss_wm_degraded': avg_loss_deg,
        }
        val_writer.writerow(val_log_data)
        val_csv.flush()

        print(
            f"[Val E{epoch:03d}] "
            f"bit_acc_clean={avg_acc_clean:.3f} "
            f"bit_acc_deg={avg_acc_deg:.3f} "
            f"PSNR={avg_psnr:.1f}"
        )

        # ============================================================
        # Save checkpoint. Stage 1 selects clean accuracy; Stage 2 selects
        # accuracy after the complete official PIMoG degradation.
        # ============================================================
        best_metric_name = (
            'bit_acc_degraded' if use_noise_layer else 'bit_acc_clean'
        )
        current_best_metric = (
            avg_acc_deg if use_noise_layer else avg_acc_clean
        )
        checkpoint = {
            'diffusion_model': model.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'config': cfg,
            'random_state': capture_random_state(train_generator),
            'bit_acc_clean': avg_acc_clean,
            'bit_acc_degraded': avg_acc_deg,
            'best_metric_name': best_metric_name,
        }

        # latest.pt follows save_interval.
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'latest.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"[Checkpoint] Saved {checkpoint_path}")

        # best.pt is evaluated every epoch, independently of save_interval.
        best_path = os.path.join(checkpoint_dir, 'best.pt')
        best_acc = 0.0
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location='cpu')
            saved_metric_name = best_ckpt.get('best_metric_name')
            if saved_metric_name == best_metric_name:
                best_acc = best_ckpt.get(
                    'best_metric_value', best_ckpt.get('best_bit_acc', 0.0)
                )
            elif saved_metric_name is None and not use_noise_layer:
                # Backward compatibility for legacy Stage 1 checkpoints,
                # whose best_bit_acc always meant clean accuracy.
                best_acc = best_ckpt.get('best_bit_acc', 0.0)

        if current_best_metric > best_acc:
            checkpoint['best_bit_acc'] = current_best_metric
            checkpoint['best_metric_value'] = current_best_metric
            torch.save(checkpoint, best_path)
            print(
                f"[Checkpoint] New best! {best_metric_name}="
                f"{current_best_metric:.3f}"
            )

        # ============================================================
        # DIAGNOSTIC: If bit_acc stays near 0.5, print warning
        # ============================================================
        if avg_acc_clean < 0.55 and epoch > 10:
            print(
                "[DIAGNOSTIC] bit_acc_clean is near 0.5. Possible causes:\n"
                "  1. wm_bits not actually fed to U-Net? Check watermark_mlp.\n"
                "  2. watermark_mlp requires_grad=True? Check parameters.\n"
                "  3. loss_wm backprop to diffusion_model? Check no .detach().\n"
                "  4. decoder input range correct [-1, 1]?\n"
                "  5. lambda_wm too small? Current: {:.1f}\n"
                "  6. wm_t_max too large? Current: {}".format(lambda_wm, wm_t_max)
            )

    # --- End training ---
    train_csv.close()
    val_csv.close()
    print(f"[Train] Done! Logs saved to {log_dir}")

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, 'final.pt')
    torch.save({
        'diffusion_model': model.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epochs,
        'global_step': global_step,
        'config': cfg,
        'random_state': capture_random_state(train_generator),
    }, final_path)
    print(f"[Checkpoint] Final model saved to {final_path}")

# ============================================================
# CLI entry point
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Watermark-Conditioned Diffusion')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g. checkpoints/best.pt)')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume:
        config['_resume_path'] = args.resume
    train(config)



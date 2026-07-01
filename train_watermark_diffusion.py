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


def residual_tv_loss(delta):
    """Penalize short, high-frequency residual streaks."""
    loss_h = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
    loss_w = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
    return loss_h + loss_w


def residual_topk_loss(delta, fraction=0.01):
    """Penalize sparse, visually obvious residual spikes."""
    flat = delta.abs().flatten(1)
    k = max(1, int(flat.size(1) * fraction))
    return flat.topk(k, dim=1).values.mean()


def residual_channel_balance_loss(delta):
    """Discourage hiding most residual energy in one color channel."""
    channel_energy = delta.abs().mean(dim=(0, 2, 3))
    return channel_energy.std(unbiased=False)


def get_loss_weights(cfg, global_step):
    train_cfg = cfg.get('train', {})
    stage = str(train_cfg.get('stage', '')).lower()
    stages_cfg = train_cfg.get('stages', {})
    if stage and stage in stages_cfg:
        stage_cfg = stages_cfg[stage] or {}
        schedule = stage_cfg.get('loss_schedule')
        if schedule:
            for item in schedule:
                until_step = int(item.get('until_step', -1))
                if until_step < 0 or global_step < until_step:
                    return _loss_weight_dict(item, stage_cfg)
        return _loss_weight_dict(stage_cfg, train_cfg)

    if train_cfg.get('use_loss_schedule', False):
        for item in train_cfg.get('loss_schedule', []):
            until_step = int(item.get('until_step', -1))
            if until_step < 0 or global_step < until_step:
                return {
                    'lambda_diff': float(item.get('lambda_diff', train_cfg['lambda_diff'])),
                    'lambda_img': float(item.get('lambda_img', train_cfg['lambda_img'])),
                    'lambda_wm': float(item.get('lambda_wm', train_cfg['lambda_wm'])),
                    'lambda_delta': float(item.get('lambda_delta', train_cfg.get('lambda_delta', 0.0))),
                    'lambda_tv': float(item.get('lambda_tv', train_cfg.get('lambda_tv', 0.0))),
                    'lambda_topk': float(item.get('lambda_topk', train_cfg.get('lambda_topk', 0.0))),
                    'lambda_channel': float(item.get('lambda_channel', train_cfg.get('lambda_channel', 0.0))),
                }
    return {
        'lambda_diff': float(train_cfg['lambda_diff']),
        'lambda_img': float(train_cfg['lambda_img']),
        'lambda_wm': float(train_cfg['lambda_wm']),
        'lambda_delta': float(train_cfg.get('lambda_delta', 0.0)),
        'lambda_tv': float(train_cfg.get('lambda_tv', 0.0)),
        'lambda_topk': float(train_cfg.get('lambda_topk', 0.0)),
        'lambda_channel': float(train_cfg.get('lambda_channel', 0.0)),
    }


def _loss_weight_dict(source, fallback):
    return {
        'lambda_diff': float(source.get('lambda_diff', fallback.get('lambda_diff', 0.0))),
        'lambda_img': float(source.get('lambda_img', fallback.get('lambda_img', 0.0))),
        'lambda_wm': float(source.get('lambda_wm', fallback.get('lambda_wm', 0.0))),
        'lambda_delta': float(source.get('lambda_delta', fallback.get('lambda_delta', 0.0))),
        'lambda_tv': float(source.get('lambda_tv', fallback.get('lambda_tv', 0.0))),
        'lambda_topk': float(source.get('lambda_topk', fallback.get('lambda_topk', 0.0))),
        'lambda_channel': float(source.get('lambda_channel', fallback.get('lambda_channel', 0.0))),
    }


def generate_train_watermark(batch_size, length, device):
    return torch.randint(0, 2, (batch_size, length), device=device).float()


def generate_val_watermark(batch_size, length, seed, device, offset=0):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed) + int(offset))
    bits = torch.randint(0, 2, (batch_size, length), generator=generator).float()
    return bits.to(device)


def grad_norm(module):
    total = 0.0
    has_grad = False
    for param in module.parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.detach().float().norm(2).item()
        total += param_norm * param_norm
        has_grad = True
    return math.sqrt(total) if has_grad else float('nan')


def tensor_is_finite(value):
    """Return True when a scalar/tensor contains only finite values."""
    if torch.is_tensor(value):
        return torch.isfinite(value.detach()).all().item()
    return math.isfinite(float(value))


def first_nonfinite_tensor(named_tensors):
    """Return the first non-finite tensor name, or None if all are finite."""
    for name, value in named_tensors:
        if not tensor_is_finite(value):
            return name
    return None


def gradients_are_finite(named_parameters):
    """Check all existing gradients for trainable named parameters."""
    for name, param in named_parameters:
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach()
        finite = torch.isfinite(grad)
        if not finite.all().item():
            nan_count = torch.isnan(grad).sum().item()
            inf_count = torch.isinf(grad).sum().item()
            finite_abs = grad[finite].abs()
            grad_abs_max = (
                finite_abs.max().item() if finite_abs.numel() > 0 else float('nan')
            )
            reason = (
                f'grad_nonfinite:{name}:nan_count={nan_count}:'
                f'inf_count={inf_count}:grad_abs_max={grad_abs_max:.6g}'
            )
            return False, reason
    return True, None


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


def _get_checkpoint_model_state(ckpt):
    if 'diffusion_model' in ckpt:
        return ckpt['diffusion_model']
    if 'model' in ckpt:
        return ckpt['model']
    return ckpt


def load_decoder_checkpoint(decoder, decoder_state, log_prefix):
    try:
        decoder.load_state_dict(decoder_state, strict=True)
        print(f"{log_prefix} Loaded watermark decoder weights.")
        return
    except RuntimeError as exc:
        print(f"{log_prefix} Decoder strict=True load failed: {exc}")

    missing, unexpected, mismatched = load_watermark_decoder_state(
        decoder, decoder_state
    )
    print(f"{log_prefix} Decoder weights are partially loaded with strict=False.")
    print(f"{log_prefix} Missing keys: {missing}")
    print(f"{log_prefix} Unexpected keys: {unexpected}")
    print(f"{log_prefix} Mismatched keys: {mismatched}")


def load_model_state_for_init(model, checkpoint_state):
    current_state = model.state_dict()
    load_state = dict(current_state)
    missing_keys = []
    unexpected_keys = []
    shape_mismatch_keys = []
    copied_first_conv = False

    for key, value in checkpoint_state.items():
        if key not in current_state:
            unexpected_keys.append(key)
            continue

        current_value = current_state[key]
        if current_value.shape == value.shape:
            load_state[key] = value
            continue

        shape_mismatch_keys.append(
            f"{key}: checkpoint={tuple(value.shape)}, current={tuple(current_value.shape)}"
        )
        if (
            key.endswith('input_blocks.0.0.weight')
            and value.ndim == 4
            and current_value.ndim == 4
            and value.shape[0] == current_value.shape[0]
            and value.shape[2:] == current_value.shape[2:]
            and value.shape[1] < current_value.shape[1]
        ):
            new_weight = current_value.clone()
            new_weight[:, :value.shape[1], :, :] = value
            new_weight[:, value.shape[1]:, :, :] = 0.0
            load_state[key] = new_weight
            copied_first_conv = True
            print(
                "[Init] Detected input channel mismatch in first conv: "
                f"checkpoint={value.shape[1]}, current={current_value.shape[1]}."
            )
            print("[Init] Copied old x_t and cover_img channels.")
            print("[Init] Initialized new wm_map channels.")

    for key in current_state:
        if key not in checkpoint_state:
            missing_keys.append(key)

    model.load_state_dict(load_state, strict=True)
    print("[Init] Loaded compatible diffusion model weights.")
    print(f"[Init] Missing model keys: {missing_keys}")
    print(f"[Init] Unexpected model keys: {unexpected_keys}")
    print(f"[Init] Shape mismatch model keys: {shape_mismatch_keys}")
    if not copied_first_conv and shape_mismatch_keys:
        print("[Init] Shape-mismatched tensors kept at current initialization.")


def resume_training(checkpoint_path, model, decoder, optimizer, scaler,
                    train_generator, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    print(f"[Resume] Resume training from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(_get_checkpoint_model_state(ckpt), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "[Resume Error] Checkpoint model structure does not match current model.\n"
            "Use --init_from if you want to initialize a new stage with changed architecture."
        ) from exc
    print("[Resume] Loaded diffusion model.")

    if 'decoder' in ckpt:
        load_decoder_checkpoint(decoder, ckpt['decoder'], "[Resume]")
    else:
        print("[Resume] No decoder weights found in checkpoint.")

    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("[Resume] Loaded optimizer.")
    else:
        print("[Resume] No optimizer state found in checkpoint.")

    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
        print("[Resume] Loaded AMP scaler.")

    start_epoch = ckpt.get('epoch', 0) + 1
    global_step = ckpt.get('global_step', 0)
    restore_random_state(ckpt.get('random_state'), train_generator)
    print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}")
    if 'best_bit_acc' in ckpt:
        best_name = ckpt.get('best_metric_name', 'bit_acc_clean')
        print(f"[Resume] Previous best {best_name}={ckpt['best_bit_acc']:.4f}")
    return start_epoch, global_step


def init_from_checkpoint(checkpoint_path, model, decoder, reset_decoder, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Init checkpoint not found: {checkpoint_path}")

    print(f"[Init] Initialize new training stage from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    load_model_state_for_init(model, _get_checkpoint_model_state(ckpt))
    print("[Init] Loaded diffusion model weights.")

    if reset_decoder:
        print("[Init] reset_decoder=true, skip loading old decoder weights.")
    elif 'decoder' in ckpt:
        load_decoder_checkpoint(decoder, ckpt['decoder'], "[Init]")
    else:
        print("[Init] No decoder weights found in checkpoint.")

    print("[Init] Skip optimizer state.")
    print("[Init] Reset start_epoch=1, global_step=0.")
    return 1, 0

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
            'use_watermark_time_emb': True,
            'use_watermark_spatial_map': True,
            'wm_map_channels': 4,
            'wm_map_size': 16,
            'wm_time_scale': 1.0,
            'wm_map_scale': 1.0,
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
            'stage': 'warmup',
            'lambda_diff': 1.0,
            'lambda_img': 1.0,
            'lambda_wm': 5.0,
            'lambda_delta': 0.0,
            'use_loss_schedule': False,
            'loss_schedule': [],
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
        use_watermark_time_emb=model_cfg.get('use_watermark_time_emb', True),
        use_watermark_spatial_map=model_cfg.get('use_watermark_spatial_map', True),
        wm_map_channels=model_cfg.get('wm_map_channels', 4),
        wm_map_size=model_cfg.get('wm_map_size', 16),
        wm_time_scale=model_cfg.get('wm_time_scale', 1.0),
        wm_map_scale=model_cfg.get('wm_map_scale', 1.0),
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
    scaler = (
        torch.amp.GradScaler(
            'cuda',
            init_scale=float(cfg['train'].get('amp_init_scale', 256)),
            growth_interval=int(cfg['train'].get('amp_growth_interval', 1000)),
        )
        if amp_enabled
        else None
    )
    print(
        f"[Train] AMP autocast: {'enabled' if amp_enabled else 'disabled'} "
        f"init_scale={cfg['train'].get('amp_init_scale', 256)} "
        f"growth_interval={cfg['train'].get('amp_growth_interval', 1000)}"
    )
    detect_anomaly = bool(cfg['train'].get('detect_anomaly', False))
    torch.autograd.set_detect_anomaly(detect_anomaly)
    if detect_anomaly:
        print("[Train] torch.autograd anomaly detection enabled")

    # --- Checkpoint loading ---
    resume_path = cfg.get('_resume_path', None)
    init_from_path = cfg.get('_init_from_path', None)
    start_epoch = 1
    global_step = 0
    if resume_path:
        start_epoch, global_step = resume_training(
            resume_path, model, decoder, optimizer, scaler,
            train_generator, device,
        )
    elif init_from_path:
        start_epoch, global_step = init_from_checkpoint(
            init_from_path,
            model,
            decoder,
            cfg['train'].get('reset_decoder', False),
            device,
        )

    # --- Loss weights ---
    initial_loss_weights = get_loss_weights(cfg, 0)

    # --- Timestep config ---
    wm_t_min = cfg['diffusion']['wm_t_min']
    wm_t_max = cfg['diffusion']['wm_t_max']
    train_t_start = cfg['diffusion']['train_t_start']

    # --- Training state ---
    epochs = cfg['train']['epochs']
    save_interval = cfg['train']['save_interval']
    sample_interval = cfg['train']['sample_interval']
    log_interval = cfg['train']['log_interval']
    debug_interval = cfg['train'].get('debug_interval', log_interval * 5)
    max_grad_norm = float(cfg['train'].get('max_grad_norm', 1.0))
    skip_nonfinite = bool(cfg['train'].get('skip_nonfinite', True))
    named_trainable_parameters = [
        (f'model.{name}', param) for name, param in model.named_parameters()
    ] + [
        (f'decoder.{name}', param) for name, param in decoder.named_parameters()
    ]
    trainable_parameters = [param for _, param in named_trainable_parameters]
    # --- CSV loggers ---
    train_log_path = os.path.join(log_dir, 'train_log.csv')
    val_log_path = os.path.join(log_dir, 'val_log.csv')
    sample_log_path = os.path.join(log_dir, 'sample_log.csv')

    csv_mode = 'a' if resume_path else 'w'
    train_csv = open(train_log_path, csv_mode, newline='')
    train_writer = csv.DictWriter(train_csv, fieldnames=[
        'epoch', 'batch_step', 'global_step',
        'loss_total', 'loss_diff', 'loss_img', 'loss_wm',
        'loss_delta', 'loss_tv', 'loss_topk', 'loss_channel',
        'bit_acc', 'psnr', 'logits_std', 'sigmoid_mean',
        'bit_flip_image_delta', 'bit_flip_logit_delta', 'lr', 'noise_layer_type',
        'skipped', 'skip_reason', 'grad_norm_clipped',
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

    sample_csv = open(sample_log_path, csv_mode, newline='')
    sample_writer = csv.DictWriter(sample_csv, fieldnames=[
        'epoch', 'global_step', 'noise_layer_type', 'bit_acc',
        'psnr_watermarked', 'psnr_degraded',
        'mae_watermarked', 'mae_degraded',
        'max_abs_delta_watermarked', 'max_abs_delta_degraded',
        'topk_abs_delta_watermarked', 'tv_watermarked',
        'channel_delta_r', 'channel_delta_g', 'channel_delta_b',
        'channel_delta_std',
        'logits_std', 'sigmoid_mean',
    ])
    if not os.path.exists(sample_log_path) or os.path.getsize(sample_log_path) == 0:
        sample_writer.writeheader()

    skipped_steps = 0
    batch_step = 0

    def log_skipped_step(epoch, batch_step, global_step, active_noise_type, reason,
                         grad_norm_clipped=float('nan')):
        nonlocal skipped_steps
        skipped_steps += 1
        optimizer.zero_grad(set_to_none=True)
        train_writer.writerow({
            'epoch': epoch,
            'batch_step': batch_step,
            'global_step': global_step,
            'loss_total': float('nan'),
            'loss_diff': float('nan'),
            'loss_img': float('nan'),
            'loss_wm': float('nan'),
            'loss_delta': float('nan'),
            'loss_tv': float('nan'),
            'loss_topk': float('nan'),
            'loss_channel': float('nan'),
            'bit_acc': float('nan'),
            'psnr': float('nan'),
            'logits_std': float('nan'),
            'sigmoid_mean': float('nan'),
            'bit_flip_image_delta': float('nan'),
            'bit_flip_logit_delta': float('nan'),
            'lr': optimizer.param_groups[0]['lr'],
            'noise_layer_type': active_noise_type,
            'skipped': 1,
            'skip_reason': reason,
            'grad_norm_clipped': grad_norm_clipped,
        })
        train_csv.flush()
        print(
            f"[Skip NonFinite] batch_step={batch_step} global_step={global_step} "
            f"noise_layer={active_noise_type} "
            f"reason={reason} skipped_steps={skipped_steps} "
            f"grad_norm={grad_norm_clipped}"
        )

    # --- Training loop ---
    print(
        f"[Train] Starting training: {epochs} epochs, "
        f"log_interval={log_interval}, debug_interval={debug_interval}"
    )
    print(
        f"[Train] stage={cfg['train'].get('stage', 'legacy')} "
        f"initial lambda_diff={initial_loss_weights['lambda_diff']}, "
        f"lambda_img={initial_loss_weights['lambda_img']}, "
        f"lambda_wm={initial_loss_weights['lambda_wm']}, "
        f"lambda_delta={initial_loss_weights['lambda_delta']}, "
        f"lambda_tv={initial_loss_weights['lambda_tv']}, "
        f"lambda_topk={initial_loss_weights['lambda_topk']}, "
        f"lambda_channel={initial_loss_weights['lambda_channel']}"
    )
    if cfg['train'].get('use_loss_schedule', False):
        print(f"[Train] loss schedule enabled: {cfg['train'].get('loss_schedule', [])}")
    print(
        f"[Train] wm_t range: [{wm_t_min}, {wm_t_max}), "
        f"noise_layer={noise_type}, max_grad_norm={max_grad_norm}, "
        f"skip_nonfinite={skip_nonfinite}"
    )

    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        decoder.train()
        noise_layer.train()

        for batch in train_loader:
            batch_step += 1
            cover_img = batch['image'].to(device)    # [B, 3, H, W], [-1, 1]
            B = cover_img.size(0)
            wm_bits = generate_train_watermark(B, watermark_length, device)
            optimizer.zero_grad(set_to_none=True)
            loss_weights = get_loss_weights(cfg, global_step)
            lambda_diff = loss_weights['lambda_diff']
            lambda_img = loss_weights['lambda_img']
            lambda_wm = loss_weights['lambda_wm']
            lambda_delta = loss_weights['lambda_delta']
            lambda_tv = loss_weights['lambda_tv']
            lambda_topk = loss_weights['lambda_topk']
            lambda_channel = loss_weights['lambda_channel']

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
            if skip_nonfinite:
                nonfinite_name = first_nonfinite_tensor([
                    ('loss_diff', loss_diff),
                    ('diffusion_objective', diffusion_objective),
                ])
                if nonfinite_name is not None:
                    log_skipped_step(
                        epoch,
                        batch_step,
                        global_step,
                        active_noise_type,
                        f'{nonfinite_name}_nan',
                    )
                    continue

            if lambda_diff > 0:
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

            attacked_nonfinite = False
            with torch.amp.autocast(device_type=device.type, enabled=False):
                # KEY: NO .detach() — loss_wm backpropagates through the U-Net.
                pred_x0 = predict_start_from_noise(
                    diffusion,
                    x_t_wm.float(),
                    t_wm,
                    pred_noise_wm.float(),
                )
                pred_x0 = pred_x0.clamp(-1, 1)
                cover_img_fp32 = cover_img.float()

                # Image fidelity loss in [-1, 1]
                loss_img = F.l1_loss(pred_x0, cover_img_fp32)
                loss_delta = (pred_x0 - cover_img_fp32).abs().mean()

                # Unified degradation layers use [0, 1].
                pred_x0_01 = ((pred_x0 + 1.0) / 2.0).clamp(1e-6, 1.0 - 1e-6)
                cover_01_for_loss = ((cover_img_fp32 + 1.0) / 2.0).clamp(0.0, 1.0)
                residual_01 = pred_x0_01 - cover_01_for_loss
                loss_tv = residual_tv_loss(residual_01)
                loss_topk = residual_topk_loss(
                    residual_01,
                    cfg['train'].get('topk_delta_fraction', 0.01),
                )
                loss_channel = residual_channel_balance_loss(residual_01)

                attacked_01 = noise_layer(pred_x0_01).float()
                if noise_type == 'mixed':
                    active_noise_type = f"mixed:{noise_layer.get_last_name()}"
                attacked_nonfinite = not tensor_is_finite(attacked_01)
                attacked_01 = torch.nan_to_num(
                    attacked_01,
                    nan=0.5,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
                decoder_input = attacked_01.mul(2.0).sub(1.0)

                pred_logits = decoder(decoder_input)
                loss_wm = F.binary_cross_entropy_with_logits(
                    pred_logits, wm_bits.float()
                )
                logits_mean = pred_logits.detach().mean().item()
                logits_std = pred_logits.detach().std().item()
                sigmoid_mean = torch.sigmoid(pred_logits.detach()).mean().item()

                # ========================================================
                # 3. Total loss
                # ========================================================
                watermark_objective = (
                    lambda_img * loss_img
                    + lambda_wm * loss_wm
                    + lambda_delta * loss_delta
                    + lambda_tv * loss_tv
                    + lambda_topk * loss_topk
                    + lambda_channel * loss_channel
                )
            if skip_nonfinite and attacked_nonfinite:
                log_skipped_step(
                    epoch,
                    batch_step,
                    global_step,
                    active_noise_type,
                    'attacked_01_nonfinite',
                )
                continue
            if skip_nonfinite:
                nonfinite_name = first_nonfinite_tensor([
                    ('loss_img', loss_img),
                    ('loss_wm', loss_wm),
                    ('loss_delta', loss_delta),
                    ('loss_tv', loss_tv),
                    ('loss_topk', loss_topk),
                    ('loss_channel', loss_channel),
                    ('watermark_objective', watermark_objective),
                ])
                if nonfinite_name is not None:
                    log_skipped_step(
                        epoch,
                        batch_step,
                        global_step,
                        active_noise_type,
                        f'{nonfinite_name}_nan',
                    )
                    continue

            # ========================================================
            # 4. Backward
            # ========================================================
            grad_norm_clipped = float('nan')
            if scaler is not None:
                scaler.scale(watermark_objective).backward()
                scaler.unscale_(optimizer)
                model_gn = grad_norm(model)
                decoder_gn = grad_norm(decoder)
                wm_mlp_gn = grad_norm(model.watermark_mlp)
                wm_map_mlp_gn = (
                    grad_norm(model.watermark_map_mlp)
                    if hasattr(model, 'watermark_map_mlp')
                    else float('nan')
                )
                grads_finite, grad_reason = gradients_are_finite(
                    named_trainable_parameters
                )
                if skip_nonfinite and not grads_finite:
                    scaler.update()
                    log_skipped_step(
                        epoch,
                        batch_step,
                        global_step,
                        active_noise_type,
                        grad_reason,
                    )
                    continue
                grad_norm_clipped = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_grad_norm,
                )
                grad_norm_clipped = float(grad_norm_clipped.detach().cpu().item())
                if skip_nonfinite and not math.isfinite(grad_norm_clipped):
                    scaler.update()
                    log_skipped_step(
                        epoch,
                        batch_step,
                        global_step,
                        active_noise_type,
                        'clip_grad_norm_nonfinite',
                        grad_norm_clipped,
                    )
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                watermark_objective.backward()
                model_gn = grad_norm(model)
                decoder_gn = grad_norm(decoder)
                wm_mlp_gn = grad_norm(model.watermark_mlp)
                wm_map_mlp_gn = (
                    grad_norm(model.watermark_map_mlp)
                    if hasattr(model, 'watermark_map_mlp')
                    else float('nan')
                )
                grads_finite, grad_reason = gradients_are_finite(
                    named_trainable_parameters
                )
                if skip_nonfinite and not grads_finite:
                    log_skipped_step(
                        epoch,
                        batch_step,
                        global_step,
                        active_noise_type,
                        grad_reason,
                    )
                    continue
                grad_norm_clipped = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_grad_norm,
                )
                grad_norm_clipped = float(grad_norm_clipped.detach().cpu().item())
                if skip_nonfinite and not math.isfinite(grad_norm_clipped):
                    log_skipped_step(
                        epoch,
                        batch_step,
                        global_step,
                        active_noise_type,
                        'clip_grad_norm_nonfinite',
                        grad_norm_clipped,
                    )
                    continue
                optimizer.step()

            global_step += 1

            loss_total = (
                lambda_diff * loss_diff
                + lambda_img * loss_img.detach()
                + lambda_wm * loss_wm.detach()
                + lambda_delta * loss_delta.detach()
                + lambda_tv * loss_tv.detach()
                + lambda_topk * loss_topk.detach()
                + lambda_channel * loss_channel.detach()
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
                    'batch_step': batch_step,
                    'global_step': global_step,
                    'loss_total': loss_total.item(),
                    'loss_diff': loss_diff.item(),
                    'loss_img': loss_img.item(),
                    'loss_wm': loss_wm.item(),
                    'loss_delta': loss_delta.item(),
                    'loss_tv': loss_tv.item(),
                    'loss_topk': loss_topk.item(),
                    'loss_channel': loss_channel.item(),
                    'bit_acc': bit_acc,
                    'psnr': psnr_val,
                    'logits_std': logits_std,
                    'sigmoid_mean': sigmoid_mean,
                    'bit_flip_image_delta': float('nan'),
                    'bit_flip_logit_delta': float('nan'),
                    'lr': optimizer.param_groups[0]['lr'],
                    'noise_layer_type': active_noise_type,
                    'skipped': 0,
                    'skip_reason': '',
                    'grad_norm_clipped': grad_norm_clipped,
                }

                if debug_interval > 0 and global_step % debug_interval == 0:
                    with torch.no_grad():
                        debug_count = min(2, B)
                        debug_x_t = x_t_wm[:debug_count]
                        debug_t = t_wm[:debug_count]
                        debug_t_scaled = t_wm_scaled[:debug_count]
                        debug_cover = cover_img[:debug_count]
                        debug_wm_a = wm_bits[:debug_count]
                        debug_wm_b = 1.0 - debug_wm_a
                        pred_noise_a = model(
                            x_t=debug_x_t,
                            t=debug_t_scaled,
                            cover_img=debug_cover,
                            wm_bits=debug_wm_a,
                        )
                        pred_noise_b = model(
                            x_t=debug_x_t,
                            t=debug_t_scaled,
                            cover_img=debug_cover,
                            wm_bits=debug_wm_b,
                        )
                        pred_x0_a = predict_start_from_noise(
                            diffusion, debug_x_t, debug_t, pred_noise_a
                        )
                        pred_x0_b = predict_start_from_noise(
                            diffusion, debug_x_t, debug_t, pred_noise_b
                        )
                        pred_x0_delta = (pred_x0_a - pred_x0_b).abs().mean().item()
                        logits_a = decoder(pred_x0_a.clamp(-1, 1))
                        logits_b = decoder(pred_x0_b.clamp(-1, 1))
                        bit_flip_logit_delta = (logits_a - logits_b).abs().mean().item()
                        log_data['bit_flip_image_delta'] = pred_x0_delta
                        log_data['bit_flip_logit_delta'] = bit_flip_logit_delta

                    print(
                        f"[Debug S{global_step:06d}] "
                        f"logits_std={logits_std:.4f} "
                        f"sigmoid_mean={sigmoid_mean:.4f} "
                        f"image_delta={pred_x0_delta:.6f} "
                        f"logit_delta={bit_flip_logit_delta:.6f} "
                        f"gn(model={model_gn:.3f},dec={decoder_gn:.3f},"
                        f"wm={wm_mlp_gn:.3f},map={wm_map_mlp_gn:.3f},"
                        f"clip={grad_norm_clipped:.3f})"
                    )

                train_writer.writerow(log_data)
                train_csv.flush()

                print(
                    f"[E{epoch:03d}|B{batch_step:06d}|S{global_step:06d}] "
                    f"L={loss_total.item():.4f} "
                    f"(diff={loss_diff.item():.4f} img={loss_img.item():.4f} "
                    f"wm={loss_wm.item():.4f} delta={loss_delta.item():.4f} "
                    f"tv={loss_tv.item():.4f} topk={loss_topk.item():.4f} "
                    f"ch={loss_channel.item():.4f}) "
                    f"lambda=({lambda_diff:.2f},{lambda_img:.2f},{lambda_wm:.2f}) "
                    f"lambda_delta={lambda_delta:.2f} "
                    f"lambda_visual=({lambda_tv:.2f},{lambda_topk:.2f},{lambda_channel:.2f}) "
                    f"bit_acc={bit_acc:.3f} PSNR={psnr_val:.1f} "
                    f"logits_std={logits_std:.4f} sigmoid_mean={sigmoid_mean:.4f} "
                    f"noise_layer={active_noise_type} grad_norm={grad_norm_clipped:.3f} "
                    f"skipped_steps={skipped_steps}"
                )
                if psnr_val > 45.0 and bit_acc < 0.6:
                    print(
                        "[WARNING] High PSNR but watermark is not learning; "
                        "the model may be collapsing to near-identity images."
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
                    s_wm = generate_val_watermark(
                        s_cover.size(0),
                        watermark_length,
                        watermark_seed,
                        device,
                        offset=global_step,
                    )

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
                    s_sigmoid = torch.sigmoid(s_logits)
                    s_logits_std = s_logits.detach().std().item()
                    s_sigmoid_mean = s_sigmoid.detach().mean().item()

                    s_delta_wm = (s_wm_01 - s_cover_01).abs()
                    s_delta_deg = (s_degraded_01 - s_cover_01).abs()
                    s_psnr_wm = compute_psnr(s_wm_01, s_cover_01, max_val=1.0)
                    s_psnr_deg = compute_psnr(s_degraded_01, s_cover_01, max_val=1.0)
                    s_mae_wm = s_delta_wm.mean().item()
                    s_mae_deg = s_delta_deg.mean().item()
                    s_max_delta_wm = s_delta_wm.max().item()
                    s_max_delta_deg = s_delta_deg.max().item()
                    s_topk_delta_wm = residual_topk_loss(
                        s_wm_01 - s_cover_01,
                        cfg['train'].get('topk_delta_fraction', 0.01),
                    ).item()
                    s_tv_wm = residual_tv_loss(s_wm_01 - s_cover_01).item()
                    s_channel_delta = s_delta_wm.mean(dim=(0, 2, 3))
                    s_channel_delta_std = s_channel_delta.std(unbiased=False).item()

                    sample_writer.writerow({
                        'epoch': epoch,
                        'global_step': global_step,
                        'noise_layer_type': sample_noise_type,
                        'bit_acc': s_acc,
                        'psnr_watermarked': s_psnr_wm,
                        'psnr_degraded': s_psnr_deg,
                        'mae_watermarked': s_mae_wm,
                        'mae_degraded': s_mae_deg,
                        'max_abs_delta_watermarked': s_max_delta_wm,
                        'max_abs_delta_degraded': s_max_delta_deg,
                        'topk_abs_delta_watermarked': s_topk_delta_wm,
                        'tv_watermarked': s_tv_wm,
                        'channel_delta_r': s_channel_delta[0].item(),
                        'channel_delta_g': s_channel_delta[1].item(),
                        'channel_delta_b': s_channel_delta[2].item(),
                        'channel_delta_std': s_channel_delta_std,
                        'logits_std': s_logits_std,
                        'sigmoid_mean': s_sigmoid_mean,
                    })
                    sample_csv.flush()

                    # Save comparison grid
                    comparison = torch.cat([s_cover_01, s_wm_01, s_degraded_01], dim=0)
                    save_path = os.path.join(
                        sample_dir,
                        f'step_{global_step:06d}_{sample_noise_type}'
                        f'_acc_{s_acc:.3f}_psnr_{s_psnr_wm:.2f}.png',
                    )
                    save_image(comparison, save_path, nrow=4)
                    print(
                        f"[Sample] Saved {save_path} "
                        f"(noise={sample_noise_type}, bit_acc={s_acc:.3f}, "
                        f"psnr_wm={s_psnr_wm:.2f}, psnr_deg={s_psnr_deg:.2f}, "
                        f"mae_wm={s_mae_wm:.4f}, max_delta_wm={s_max_delta_wm:.4f}, "
                        f"topk_delta_wm={s_topk_delta_wm:.4f}, tv_wm={s_tv_wm:.4f}, "
                        f"channel_delta=({s_channel_delta[0].item():.4f},"
                        f"{s_channel_delta[1].item():.4f},"
                        f"{s_channel_delta[2].item():.4f}), "
                        f"logits_std={s_logits_std:.4f}, sigmoid_mean={s_sigmoid_mean:.4f})"
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
                B_v = v_cover.size(0)
                v_wm = generate_val_watermark(
                    B_v,
                    watermark_length,
                    watermark_seed,
                    device,
                    offset=global_step + len(val_bit_acc_clean),
                )

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
        validation_finite = all(
            math.isfinite(value)
            for value in [
                avg_acc_clean,
                avg_acc_deg,
                avg_psnr,
                avg_loss_clean,
                avg_loss_deg,
            ]
        )
        if not validation_finite:
            print(
                f"[Checkpoint Skip] epoch={epoch} reason=nonfinite_validation "
                f"skipped_steps={skipped_steps}"
            )
            continue

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
            'skipped_steps': skipped_steps,
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
            current_weights = get_loss_weights(cfg, global_step)
            print(
                "[DIAGNOSTIC] bit_acc_clean is near 0.5. Possible causes:\n"
                "  1. wm_bits not actually fed to U-Net? Check watermark_mlp.\n"
                "  2. watermark_mlp requires_grad=True? Check parameters.\n"
                "  3. loss_wm backprop to diffusion_model? Check no .detach().\n"
                "  4. decoder input range correct [-1, 1]?\n"
                "  5. lambda_wm too small? Current: {:.1f}\n"
                "  6. wm_t_max too large? Current: {}".format(
                    current_weights['lambda_wm'], wm_t_max
                )
            )

    # --- End training ---
    train_csv.close()
    val_csv.close()
    sample_csv.close()
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
                        help='Path to checkpoint to continue the same training stage')
    parser.add_argument('--init_from', type=str, default=None,
                        help='Path to checkpoint used to initialize a new training stage')
    args = parser.parse_args()

    if args.resume and args.init_from:
        parser.error(
            "[Error] --resume and --init_from cannot be used at the same time.\n"
            "Use --resume for continuing the same training stage.\n"
            "Use --init_from for initializing a new stage from a previous checkpoint."
        )

    config = load_config(args.config)
    if args.resume:
        config['_resume_path'] = args.resume
    if args.init_from:
        config['_init_from_path'] = args.init_from
    train(config)



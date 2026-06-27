"""
Decode watermark from real screen-captured photos.

Usage:
    python eval_real_screen.py \
        --checkpoint checkpoints_stage2/best.pt \
        --input_dir ./real_screen_photos/ \
        --watermark "1010101011001010" \
        --image_size 128
"""

import os
import sys
import argparse
import glob

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms.functional import center_crop, resize

sys.path.insert(0, os.path.dirname(__file__))

from models.watermark_unet import WatermarkConditionedUNet
from models.watermark_decoder import WatermarkDecoder


def main():
    parser = argparse.ArgumentParser(description='Decode watermark from real screen-captured photos')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (.pt)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing real screen-captured photos (.png/.jpg)')
    parser.add_argument('--watermark', type=str, default=None,
                        help='Expected watermark bits (e.g. "101010..."). If not provided, only outputs decoded bits.')
    parser.add_argument('--watermark_length', type=int, default=64,
                        help='Watermark bit length (default: 64)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size used during training (default: 128)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Eval] Using device: {device}")

    # --- Load checkpoint ---
    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Recover model config
    cfg = ckpt.get('config', {})
    model_cfg = cfg.get('model', {})
    diffusion_cfg = cfg.get('diffusion', {})
    image_size = args.image_size or cfg.get('data', {}).get('image_size', 128)
    watermark_length = args.watermark_length or cfg.get('data', {}).get('watermark_length', 64)

    # --- Build decoder ---
    decoder = WatermarkDecoder(watermark_length=watermark_length).to(device)
    decoder.load_state_dict(ckpt['decoder'])
    decoder.eval()

    # --- Collect image files ---
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    image_paths = []
    for ext in exts:
        image_paths.extend(sorted(glob.glob(os.path.join(args.input_dir, ext))))
        image_paths.extend(sorted(glob.glob(os.path.join(args.input_dir, ext.upper()))))

    if not image_paths:
        print(f"[Eval] ERROR: No images found in {args.input_dir}")
        sys.exit(1)

    print(f"[Eval] Found {len(image_paths)} image(s)")

    # --- Prepare expected watermark ---
    expected_bits = None
    if args.watermark:
        wm_str = args.watermark.strip()
        if len(wm_str) < watermark_length:
            wm_str = wm_str.ljust(watermark_length, '0')
        elif len(wm_str) > watermark_length:
            wm_str = wm_str[:watermark_length]
        expected_bits = torch.tensor([int(b) for b in wm_str], device=device).float()

    # --- Decode each photo ---
    print(f"{'Image':<40s} {'Decoded bits (first 32)':<35s} {'Accuracy'}")
    print("-" * 90)

    accuracies = []

    with torch.no_grad():
        for img_path in image_paths:
            # Read image (returns [C, H, W] in [0, 255] uint8)
            img = read_image(img_path).float() / 255.0

            # Preserve aspect ratio, then take the same center crop used by validation.
            if img.shape[1] != image_size or img.shape[2] != image_size:
                img = resize(img, image_size, antialias=True)
                img = center_crop(img, [image_size, image_size])

            # Add batch dim [1, C, H, W]
            img = (img * 2.0 - 1.0).unsqueeze(0).to(device)

            # Decode
            logits = decoder(img)  # [1, L]
            bits = (torch.sigmoid(logits) > 0.5).float()  # [1, L]

            # Display
            bits_str = ''.join(str(int(b)) for b in bits[0][:32].cpu())
            fname = os.path.basename(img_path)

            if expected_bits is not None:
                acc = (bits[0] == expected_bits.to(device)).float().mean().item()
                accuracies.append(acc)
                print(f"{fname:<40s} {bits_str:<35s} {acc:.4f}")
            else:
                print(f"{fname:<40s} {bits_str}")

    # --- Summary ---
    if expected_bits is not None and accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        print("-" * 90)
        print(f"Average accuracy over {len(accuracies)} image(s): {avg_acc:.4f}")

        # Save CSV
        csv_path = os.path.join(args.input_dir, 'real_screen_results.csv')
        with open(csv_path, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(['image', 'bit_accuracy'])
            for p, a in zip(image_paths, accuracies):
                writer.writerow([os.path.basename(p), a])
            writer.writerow(['AVERAGE', avg_acc])
        print(f"[Eval] Results saved to: {csv_path}")


if __name__ == '__main__':
    main()

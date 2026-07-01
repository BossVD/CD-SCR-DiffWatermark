# WaDiff - Watermark-Conditioned Diffusion Experiment

WaDiff 是一个面向抗屏摄水印的扩散模型实验框架。核心思想是：以已有载体图像为条件，通过 watermark-conditioned diffusion model 做内容保持的重绘式水印嵌入，使水印信息不再只是简单叠加的像素扰动，而是参与图像重绘生成过程。训练时同时优化扩散模型和 watermark decoder，并在 Stage2 引入可微屏摄退化层，让水印在 PIMoG/OLED/LED/Projector 等显示-拍摄退化后仍可被提取。

整体流程：

```text
cover image + watermark bits
-> watermark-conditioned diffusion model
-> watermarked image
-> screen-capture noise layer
-> watermark decoder
-> recovered watermark bits
```

仓库地址：

```text
https://github.com/BossVD/CD-SCR-DiffWatermark
```

## 代码结构

```text
.
|-- train_watermark_diffusion.py       # 主训练脚本，支持 --init_from / --resume
|-- sample_embed_watermark.py          # 单图水印嵌入、退化验证和可视化输出
|-- eval_watermark_robustness.py       # 验证集鲁棒性评估
|-- eval_real_screen.py                # 实拍/外部屏摄图像评估
|-- models/
|   |-- watermark_unet.py              # 载体图像 + watermark bits 条件 UNet
|   `-- watermark_decoder.py           # watermark decoder，默认 residual_multiscale
|-- dataset/
|   `-- watermark_image_dataset.py     # 图像读取、水印 bit 生成、裁剪和归一化
|-- NOISE_LAYER/
|   |-- build_noise_layer.py           # none/pimog/oled/led/projector/mixed 构建入口
|   |-- PIMoG_Layer.py                 # LCD/PIMoG 屏摄退化
|   |-- OLED_Layer.py                  # OLED 屏幕退化
|   |-- LED_Layer.py                   # LED 大屏点阵退化
|   `-- Projector_Layer.py             # 投影仪退化
|-- configs/
|   |-- watermark_stage1.yaml
|   |-- watermark_stage2_mixed.yaml
|   |-- watermark_stage2_amp_pimog_oled.yaml
|   |-- watermark_stage2_amp_add_led.yaml
|   |-- watermark_stage2_amp_full_mixed.yaml
|   `-- watermark_stage2_amp_full_mixed_uniform.yaml
`-- tools/
    |-- debug_oled_layer.py            # OLED forward/backward 数值稳定性测试
    |-- test_noise_layer.py
    |-- test_led_layer.py
    `-- visualize_oled_layer.py
```

## 环境安装

服务器建议使用 conda 环境：

```bash
conda create -n wadiff python=3.10 -y
conda activate wadiff
pip install -r requirements.txt
```

`requirements.txt` 当前依赖为：

```text
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.20.0,<3.0.0
pillow>=9.0.0
PyYAML>=6.0
tensorboard>=2.10.0
kornia>=0.7.0,<0.9.0
```

本机当前 AGENTS 约定所有 Python 命令使用：

```powershell
D:\Anaconda_envs\envs\wadiff\python.exe train_watermark_diffusion.py --config configs/watermark_stage1.yaml
```

下面命令以服务器常用 `PYTHONPATH=. python ...` 写法展示；在本机运行时把 `python` 替换成上面的完整路径。

## 数据集准备

配置文件默认使用 COCO 目录：

```text
/root/autodl-tmp/datasets/train2017
/root/autodl-tmp/datasets/val2017
```

对应 YAML 字段：

```yaml
data:
  train_dir: /root/autodl-tmp/datasets/train2017
  val_dir: /root/autodl-tmp/datasets/val2017
  image_size: 128
  watermark_length: 16
  watermark_seed: 42
  train_watermark_mode: random
  val_watermark_mode: deterministic_random
  max_train_images: 10000
  max_val_images: 1000
```

水印长度由 `data.watermark_length` 决定，当前 Stage1/Stage2 默认都是 16 bit。采样时传入的水印不足该长度会补 0，超出会被截断。不要按旧说明固定理解为其他长度。

## 方法要点

`WatermarkConditionedUNet` 使用两条水印条件路径：

- `wm_bits -> watermark_mlp -> time embedding`：把水印作为全局条件注入 UNet time embedding。
- `wm_bits -> watermark_map_mlp -> spatial map`：生成低分辨率水印 map，上采样后与 `x_t`、`cover_img` 拼接输入 UNet。

默认 decoder 是 `residual_multiscale`，输出 raw logits，训练使用 `BCEWithLogitsLoss`。训练和采样都采用 image-to-image 范式：从 cover image 的加噪版本出发反推 `pred_x0`，不是从纯噪声生成。

## Stage1 Clean Training

Stage1 目标是在 `noise_layer.type: none` 的 clean 条件下训练扩散模型和 decoder，让模型先学会在保持图像质量的同时稳定嵌入和提取水印。

推荐命令：

```bash
PYTHONPATH=. python train_watermark_diffusion.py \
  --config configs/watermark_stage1.yaml
```

当前 `configs/watermark_stage1.yaml` 要点：

```yaml
train:
  lr: 0.0001
  epochs: 50
  batch_size: 32
  use_amp: true
  stage: warmup
  lambda_diff: 0.0
  lambda_img: 0.1
  lambda_wm: 20.0
  save_interval: 1
  sample_interval: 5000
  log_interval: 100
  debug_interval: 500

noise_layer:
  type: none

output:
  checkpoint_dir: ./checkpoints_stage1
  sample_dir: ./outputs_stage1/samples
  log_dir: ./outputs_stage1/logs
```

Stage1 checkpoint：

```text
checkpoints_stage1/latest.pt   # 按 save_interval 保存
checkpoints_stage1/best.pt     # 按 bit_acc_clean 选择
checkpoints_stage1/final.pt    # 训练结束保存
```

同一阶段中断后继续：

```bash
PYTHONPATH=. python train_watermark_diffusion.py \
  --config configs/watermark_stage1.yaml \
  --resume checkpoints_stage1/latest.pt
```

## Stage2 Mixed Robust Training

Stage2 在稳定 Stage1 checkpoint 基础上引入屏摄退化模拟层，训练水印在不同显示/拍摄退化下仍可提取。当前支持的 `noise_layer.type`：

```text
none
pimog
oled
led
projector
mixed
```

推荐不要从出现过 NaN、黑图、`PSNR=nan`、`loss=nan` 或大量 skipped step 的 Stage2 checkpoint 继续 `--resume`。这类 checkpoint 可能已经保存了污染的 optimizer/scaler 状态。应重新从稳定的 Stage1 checkpoint 初始化：

```bash
PYTHONPATH=. python train_watermark_diffusion.py \
  --config configs/watermark_stage2_mixed.yaml \
  --init_from checkpoints_stage1/final.pt
```

`--init_from` 与 `--resume` 的区别：

- `--init_from`：用于从已有 checkpoint 初始化新阶段，只加载 diffusion model 和 decoder 权重，不继承 optimizer、scaler、epoch、global_step。
- `--resume`：用于严格续训同一阶段，会继承 optimizer、scaler、epoch、global_step 和随机状态。

当前 `configs/watermark_stage2_mixed.yaml` 是保守 curriculum 起点，只混合 PIMoG 和 OLED：

```yaml
train:
  lr: 0.00002
  epochs: 60
  use_amp: true
  max_grad_norm: 1.0
  skip_nonfinite: true
  amp_init_scale: 256
  amp_growth_interval: 1000

noise_layer:
  type: mixed
  mixed:
    candidates: [pimog, oled]
    probs: [0.70, 0.30]

output:
  checkpoint_dir: ./checkpoints_stage2_mixed
```

Stage2 checkpoint：

```text
checkpoints_stage2_mixed/latest.pt
checkpoints_stage2_mixed/best.pt     # 开启噪声层时按 bit_acc_degraded 选择
checkpoints_stage2_mixed/final.pt
```

## Stage2 AMP Curriculum 配置

仓库已提供四个 Stage2 AMP 配置，用于逐步扩大 mixed 噪声分布：

```text
configs/watermark_stage2_amp_pimog_oled.yaml
  candidates: [pimog, oled]
  probs: [0.70, 0.30]
  output.checkpoint_dir: ./checkpoints_stage2_amp_pimog_oled

configs/watermark_stage2_amp_add_led.yaml
  candidates: [pimog, oled, led]
  probs: [0.50, 0.30, 0.20]
  output.checkpoint_dir: ./checkpoints_stage2_amp_add_led

configs/watermark_stage2_amp_full_mixed.yaml
  candidates: [pimog, oled, led, projector]
  probs: [0.35, 0.30, 0.20, 0.15]
  output.checkpoint_dir: ./checkpoints_stage2_amp_full_mixed

configs/watermark_stage2_amp_full_mixed_uniform.yaml
  candidates: [pimog, oled, led, projector]
  probs: [0.25, 0.25, 0.25, 0.25]
  output.checkpoint_dir: ./checkpoints_stage2_amp_full_mixed_uniform
```

建议训练顺序：

```bash
PYTHONPATH=. python train_watermark_diffusion.py \
  --config configs/watermark_stage2_amp_pimog_oled.yaml \
  --init_from checkpoints_stage1/final.pt

PYTHONPATH=. python train_watermark_diffusion.py \
  --config configs/watermark_stage2_amp_add_led.yaml \
  --init_from checkpoints_stage2_amp_pimog_oled/best.pt

PYTHONPATH=. python train_watermark_diffusion.py \
  --config configs/watermark_stage2_amp_full_mixed.yaml \
  --init_from checkpoints_stage2_amp_add_led/best.pt
```

四类均匀 mixed 可作为稳定后的压力测试，不建议作为 Stage2 初期默认起点：

```bash
PYTHONPATH=. python train_watermark_diffusion.py \
  --config configs/watermark_stage2_amp_full_mixed_uniform.yaml \
  --init_from checkpoints_stage2_amp_full_mixed/best.pt
```

## AMP 与数值稳定保护

Stage2 支持 AMP，但不是整条链路都使用 FP16。当前训练策略是：

- UNet forward 使用 AMP autocast 加速。
- `predict_start_from_noise`、`noise_layer`、decoder 和 watermark loss 在 FP32 中执行。

原因是 OLED/LED/Projector 等退化层包含 `grid_sample`、`pow`、`sqrt`、blur、`interpolate` 等操作，半精度反向传播更容易产生不稳定梯度。因此正式 Stage2 中只对 UNet 主干使用 AMP，退化层和水印损失链路保持 FP32。

关键配置：

```yaml
train:
  use_amp: true
  amp_init_scale: 256
  amp_growth_interval: 1000
  max_grad_norm: 1.0
  skip_nonfinite: true
```

当前训练代码包含以下保护：

- 检查 non-finite `loss_diff`、`loss_img`、`loss_wm`、`watermark_objective` 等 loss。
- 检查退化层输出 `attacked_01` 是否包含 NaN/Inf。
- 检查所有可训练参数的梯度是否 finite。
- 异常 step 自动跳过，不执行 `optimizer.step()`。
- 被跳过的 step 不推进 `global_step`。
- 验证指标 non-finite 时跳过该 epoch 的 checkpoint 保存。
- AMP 模式下先 `scaler.unscale_(optimizer)`，再检查/裁剪梯度。
- 使用 `clip_grad_norm_` 和 `max_grad_norm` 控制梯度范数。
- 日志输出 active `noise_layer`、`grad_norm`、`skipped_steps`。

示例日志：

```text
[E002|B000500|S000500] L=0.0480 ... noise_layer=mixed:oled grad_norm=1.018 skipped_steps=0
```

含义：

- `B`：已处理的 batch step。
- `S`：成功执行 optimizer step 的 `global_step`。
- `skipped_steps`：由于 non-finite loss/grad/退化输出被跳过的 step 数。

## Noise Layer 说明

四类显示设备/退化层对应关系：

```text
PIMoG / LCD: 普通显示器或 LCD 屏摄退化。
OLED: 手机等 OLED 屏幕的 tone、子像素、相机模糊、banding、view color shift 等退化。
LED: 大屏 LED 点阵、灯珠结构、moire、scanline、bloom 等退化。
Projector: 投影仪反射式投影/拍摄退化，包括 gamma、falloff、hotspot、纹理、环境光等。
```

`mixed` 每个 forward 为整个 batch 抽取一个候选层，并通过 `get_last_name()` 记录实际选中的层，训练日志会显示为 `mixed:oled`、`mixed:pimog` 等。

## OLED 稳定性修复与 Debug

`OLED_Layer` 已针对 Stage2 可微训练做数值稳定处理，重点修复 sensor noise 中 `sqrt(0)` 附近可能导致梯度爆炸的问题。相关配置包括：

```yaml
noise_layer:
  oled:
    train_safe: true
    debug_finite: false
    sensor_noise_eps: 0.0001
    final_nan_to_num: true
```

独立测试命令：

```bash
PYTHONPATH=. python tools/debug_oled_layer.py
```

正常输出应类似：

```text
--- train_safe_noise_off ---
output finite=True nan=0 inf=0
input_grad finite=True nan=0 inf=0
--- train_safe_noise_on ---
output finite=True nan=0 inf=0
input_grad finite=True nan=0 inf=0
OLED debug completed without non-finite values.
```

如果脚本出现 `FloatingPointError`，或 `output/input_grad` 中有 `nan/inf`，说明 OLED 噪声层仍存在数值稳定问题，应先修复噪声层，不要直接进入 Stage2 mixed 训练。

## 采样命令

单图嵌入水印：

```bash
PYTHONPATH=. python sample_embed_watermark.py \
  --checkpoint checkpoints_stage2_mixed/best.pt \
  --config configs/watermark_stage2_mixed.yaml \
  --input path/to/cover.png \
  --watermark 1010101010101010 \
  --output outputs/sample.png \
  --t_start 300
```

同时保存退化图和对比图：

```bash
PYTHONPATH=. python sample_embed_watermark.py \
  --checkpoint checkpoints_stage2_mixed/best.pt \
  --config configs/watermark_stage2_mixed.yaml \
  --input path/to/cover.png \
  --watermark 1010101010101010 \
  --output outputs/sample.png \
  --noise_layer mixed \
  --save_degraded \
  --degradation_types pimog,oled,led,projector
```

采样脚本会输出：

- `bit_acc_clean`
- `bit_acc_degraded`
- clean/degraded recovered bits
- 水印图 `--output`
- `comparison/<name>_comparison.png`
- 使用 `--save_degraded` 时额外保存 `degraded/<name>_grid.png` 和各退化图

## 鲁棒性评估

验证集评估：

```bash
PYTHONPATH=. python eval_watermark_robustness.py \
  --checkpoint checkpoints_stage2_mixed/best.pt \
  --config configs/watermark_stage2_mixed.yaml \
  --noise_layers clean,pimog,oled,led,projector,mixed \
  --output ./outputs_stage2_mixed/eval_results.csv
```

常用参数：

```text
--data_dir      不传时使用 config.data.val_dir
--batch_size    默认 4
--t_start       默认 300
--device        默认 cuda
--seed          不传时使用 checkpoint/config 中的 seed
```

输出包括：

- 汇总 CSV：`bit_acc_<layer>`、`ber_<layer>`、`psnr_<layer>`
- 逐图 CSV：`*_per_image.csv`
- 示例图：`eval_samples/`

## 日志指标解释

训练日志字段：

```text
L: 总损失
diff: 扩散噪声预测损失
img: 图像重建 L1 损失
wm: 水印提取 BCEWithLogits 损失
delta: 水印图与原图的平均残差
tv: 残差 total variation 正则
topk: 局部大残差惩罚
ch: 通道平衡损失
lambda: 当前 diff/img/wm 权重
lambda_delta: 当前 residual mean 权重
lambda_visual: 当前 tv/topk/channel 权重
bit_acc: 当前 batch 水印 bit 准确率
PSNR: 图像质量
logits_std: decoder 输出 logits 标准差
sigmoid_mean: decoder 输出 sigmoid 后均值
noise_layer: 当前 batch 使用的噪声层
grad_norm: 梯度裁剪前返回的梯度范数
skipped_steps: 因 NaN/Inf 被跳过的 step 数
```

如果出现 `loss=nan`、`PSNR=nan`、sample 黑图或 `skipped_steps` 快速增加，应立即停止训练，检查最近日志中的 `noise_layer`，并优先运行对应噪声层 debug。

## Checkpoint 使用规则

- Stage1 `best.pt` 按 `bit_acc_clean` 保存。
- Stage2 或任意启用噪声层的训练，`best.pt` 按 `bit_acc_degraded` 保存。
- 阶段切换使用 `--init_from`。
- 同一阶段断点续训才使用 `--resume`。
- 不要从已经出现 NaN/Inf、黑图或大量 skipped step 的 Stage2 checkpoint resume。
- 修改 `data.watermark_length`、UNet 水印 map 设置或 decoder 架构后，旧 checkpoint 可能只能部分加载，正式训练前应重新确认日志中的 load/mismatch 信息。

## FAQ

### 1. 为什么 Stage2 比 Stage1 慢？

Stage2 多了可微屏摄退化层，例如 `grid_sample`、blur、`interpolate`、gamma、moire、noise 等操作，而且这些操作还要参与反向传播，所以比 clean training 慢。

### 2. 为什么不建议 Stage2 初期直接四类均匀 mixed？

可以直接四类一起训练，但不建议初期就均匀混合。LED 和 Projector 是更强退化层，容易造成梯度冲击。推荐先用 PIMoG+OLED，稳定后加入 mild LED，最后加入 Projector；四类均匀 mixed 更适合稳定后的压力测试。

### 3. Stage2 出现 NaN 怎么办？

停止训练；不要 resume 已污染的 Stage2 checkpoint；检查日志中最近的 `noise_layer`；确认 `skip_nonfinite: true` 和 `max_grad_norm` 已开启；从稳定 Stage1 checkpoint 重新 `--init_from`；必要时先单独运行对应噪声层 debug。

### 4. 为什么 AMP 不覆盖整个 Stage2？

噪声层中的 `pow`、`sqrt`、`grid_sample`、blur 等操作在半精度下更容易产生不稳定梯度。当前策略是 UNet 使用 AMP 加速，退化层和 decoder/loss 链路保持 FP32。

### 5. OLED debug 脚本有什么用？

用于独立测试 `OLED_Layer` 的 forward/backward 是否产生 NaN/Inf。进入 Stage2 mixed 前建议先运行，尤其是在修改 OLED 参数或升级 PyTorch 后。

## 需要重点核对的配置项

正式训练前建议核对：

```text
data.train_dir / data.val_dir
data.watermark_length
train.batch_size
train.lr
train.use_amp
train.max_grad_norm
train.skip_nonfinite
noise_layer.type
noise_layer.mixed.candidates
noise_layer.mixed.probs
noise_layer.oled.train_safe
noise_layer.oled.sensor_noise_eps
output.checkpoint_dir
output.sample_dir
output.log_dir
```

## 参考

- WaDiff (ECCV 2024): A Watermark-Conditioned Diffusion Model for IP Protection
- PIMoG (MM 2022): An Effective Screen-shooting Noise-Layer Simulation for Deep-Learning-Based Watermarking Network

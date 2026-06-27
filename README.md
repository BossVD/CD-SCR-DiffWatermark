# WaDiff — 基于扩散模型的水印嵌入与屏摄鲁棒性实验

## 核心思路

```
cover image + watermark bits → diffusion model → watermarked image
                                  ↓
                          PIMoG 屏摄噪声层
                                  ↓
                          watermark decoder → recovered bits
```

扩散模型在载体图像条件和水印条件共同引导下，对载体图像进行内容保持的重绘，
并在重绘过程中嵌入水印信息。训练和采样阶段均采用 image-to-image 范式，
始终从 cover image 的加噪版本出发，从不使用纯噪声 N(0,I)。

## 项目结构

```
guided_diffusion/          # 精简版 guided-diffusion（UNet + 扩散过程）
dataset/                   # 数据集加载（支持 max_images 限制）
models/                    # 条件 U-Net、水印解码器
NOISE_LAYER/               # 统一退化层（PIMoG、Projector、Mixed）
configs/                   # YAML 配置文件
train_watermark_diffusion.py    # 训练脚本
sample_embed_watermark.py       # 采样/水印嵌入
eval_watermark_robustness.py    # 鲁棒性评估
```

## 快速开始

### 1. 安装环境

```bash
conda create -n wadiff python=3.10 -y
conda activate wadiff
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. 准备数据

COCO 2017 数据集目录结构：

```
/path/to/datasets/
  train2017/
    000000000009.jpg
    ...
  val2017/
    000000000139.jpg
    ...
```

---

## Stage 1：基础水印嵌入（无屏摄退化）

目的：让模型学会在不加噪声层的情况下嵌入并提取水印。

```bash
python train_watermark_diffusion.py --config configs/watermark_diffusion.yaml
```

关键观测指标：`bit_acc_clean` 应显著高于 0.5 并逐步达到 0.85+，`loss_wm` 持续下降，`PSNR` 稳定。

### Stage 1 配置要点

```yaml
noise_layer:
  type: none             # 关闭所有退化层
train:
  lr: 0.0001
  lambda_wm: 5.0
  epochs: 50
diffusion:
  wm_t_min: 0
  wm_t_max: 200
  train_t_start: 200
output:
  checkpoint_dir: ./checkpoints
  sample_dir: ./outputs/samples
  log_dir: ./outputs/logs
```

---

## Stage 2：屏摄鲁棒性微调（PIMoG 噪声层）

Stage 2 基于 PIMoG 论文（MM 2022）的噪声层设计，包含四种屏摄退化：

| 噪声层 | 模拟效果 |
|--------|----------|
| Perspective（透视畸变） | 随机四角点扰动 + 透视变换，模拟拍摄角度偏差 |
| Illumination（光照畸变） | 点光源径向渐变 / 线光源方向渐变，模拟环境光照不均 |
| Moiré（摩尔纹） | 径向余弦 + 方向余弦取 min，模拟屏幕-相机采样干涉 |
| Gaussian（高斯噪声） | σ = √0.001，近似传感器残余噪声 |

融合公式：`I_no = 0.85 × I_D × I_PD + 0.15 × M_D + GN`

### Stage 2 训练命令

```bash
python train_watermark_diffusion.py \
  --config configs/watermark_diffusion.yaml \
  --resume checkpoints/best.pt
```

### Stage 2 配置要点

```yaml
noise_layer:
  type: pimog             # none / pimog / projector / mixed

train:
  lr: 0.00005           # 微调用更低学习率
  lambda_wm: 8.0         # 噪声层下提高水印权重
  epochs: 50

output:
  checkpoint_dir: ./checkpoints_stage2
  sample_dir: ./outputs_stage2/samples
  log_dir: ./outputs_stage2/logs
```

### 中断后恢复训练

```bash
python train_watermark_diffusion.py \
  --config configs/watermark_diffusion.yaml \
  --resume checkpoints_stage2/latest.pt
```

---

## 采样（生成带水印图）

```bash
# 基础采样（随机水印）
python sample_embed_watermark.py \
  --checkpoint checkpoints_stage2/best.pt \
  --input ./test_images/cover.png \
  --output ./outputs_stage2/watermarked.png \
  --t_start 200

# 指定水印内容
python sample_embed_watermark.py \
  --checkpoint checkpoints_stage2/best.pt \
  --input ./test_images/cover.png \
  --watermark "1010101011001010" \
  --output ./outputs_stage2/watermarked.png \
  --t_start 200
```

水印位数不足 64 位会自动补 0，超出会被截断。

---

## 屏摄鲁棒性评估

```bash
python eval_watermark_robustness.py \
  --checkpoint checkpoints_stage2/best.pt \
  --data_dir ./data/val \
  --output ./outputs_stage2/eval_results.csv
```

---

## 训练模式切换

| 模式 | max_train_images | epochs | image_size | 用途 |
|------|:---:|:---:|:---:|------|
| 快速调试 | 10000 | 10 | 64 | 验证流程跑通 |
| 全量训练 | null | 20~50 | 128 | 正式训练 |

只需改 YAML，不需要改代码。

## 关键设计

- **保持图像比例**：训练集将短边缩放到目标尺寸后随机裁剪；验证、采样和实拍评估使用中心裁剪，不把原图强制拉伸成正方形
- **确定性水印**：验证集根据相对路径和 `data.watermark_seed` 固定水印；训练集按图片和 epoch 可复现地变化，避免模型记忆“图片→水印”映射
- **实验随机种子**：`train.seed` 统一控制 Python、NumPy、PyTorch 和 DataLoader，检查点同时保存随机状态以支持可复现恢复
- **显存控制**：`use_amp: true` 启用真实 autocast；扩散分支先反向并释放计算图，再运行水印分支，避免同时保留两套 U-Net 激活
- **最佳模型指标**：关闭 PIMoG 时 `best.pt` 按 `bit_acc_clean` 保存；开启 PIMoG 时按 `bit_acc_degraded` 保存，并在每个验证 epoch 后判断
- **梯度流**：`loss_wm` 通过整个计算图反传（无 `.detach()`），同时优化 diffusion_model + decoder
- **t_diff / t_wm 分离**：噪声预测用全时间步，水印损失用小时间步保证 pred_x0 稳定
- **图像范围**：扩散模型和 decoder 使用 `[-1,1]`；统一退化层输入输出使用 `[0,1]`，训练接入点负责转换
- **image-to-image**：训练和采样始终从 cover 加噪出发，非纯噪声生成
- **统一退化配置**：仅通过 `noise_layer.type` 选择退化层，支持 `none`、`pimog`、`projector` 和 `mixed`

## 参考文献

- WaDiff (ECCV 2024): [A Watermark-Conditioned Diffusion Model for IP Protection](https://arxiv.org/abs/2403.10893)
- PIMoG (MM 2022): [An Effective Screen-shooting Noise-Layer Simulation for Deep-Learning-Based Watermarking Network](https://doi.org/10.1145/3503161.3548049)

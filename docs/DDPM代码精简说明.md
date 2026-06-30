# DDPM 代码精简说明

本文档记录 `diffusion/ddpm` 从 OpenAI guided-diffusion 风格代码精简为 SimpleSR 超分研究 baseline 的修改范围和当前保留接口。

## 1. 修改目标

本次修改目标是保留一个清晰的 DDPM 风格扩散模型代码框架，服务后续超分研究和 baseline 实现。

原则：

- 不兼容 guided-diffusion 的旧训练脚本和参数接口。
- 删除 classifier 相关代码。
- 删除 fp16 手动转换和 MixedPrecisionTrainer。
- 删除 1D/3D 卷积兼容，只保留 2D 图像模型。
- 尽量使用 PyTorch 官方模块，例如 `nn.SiLU`、`nn.Conv2d`、`F.interpolate`。
- 保留 DDPM/DDIM 训练和采样主干。
- 保留超分模型需要的低分图像条件输入。

## 2. 当前文件职责

```text
diffusion/ddpm/
  nn.py                  # 2D 网络工具函数
  unet.py                # 精简 2D UNet 和 SuperResModel
  gaussian_diffusion.py  # DDPM/DDIM 和 GaussianDiffusionSR
  respace.py             # 类 IDDPM 的 timestep respacing / 少步采样
  resample.py            # uniform timestep sampler
  losses.py              # 简单 loss 工具，目前只保留 MSE
  script_util.py         # 创建 SR UNet 和 diffusion 的工具
  train_util.py          # 轻量训练 step 辅助，不再包含 guided-diffusion TrainLoop
```

## 3. 删除的 guided-diffusion 功能

| 删除内容 | 原因 |
| --- | --- |
| classifier / EncoderUNet / classifier guidance | 当前论文目标是超分，不做类别条件生成。 |
| `class_cond` / `NUM_CLASSES` | 不做 ImageNet 类别条件。 |
| fp16 转换工具和 `MixedPrecisionTrainer` | SimpleSR 训练框架后续如需 AMP，应使用 PyTorch 官方 AMP。 |
| `conv_nd` / `avg_pool_nd` / `dims` | 只做 2D 图像超分，不需要 1D/3D 兼容。 |
| guided-diffusion `TrainLoop` | SimpleSR 已有 `BaseModel`、`train.py`、EMA、checkpoint、日志体系。 |
| `LossSecondMomentResampler` | 第一版 baseline 使用 uniform timestep 采样，简单稳定。 |
| learned variance / VLB loss | 第一版 baseline 保留固定方差和 MSE 目标。 |
| blobfile / dist_util / logger 依赖 | 避免引入 guided-diffusion 的外部训练基础设施。 |

## 4. 保留的核心能力

### 4.1 普通 DDPM

`GaussianDiffusion` 保留以下接口：

| 接口 | 作用 |
| --- | --- |
| `q_sample(x_start, t, noise=None)` | 从 `q(x_t | x_0)` 前向加噪。 |
| `q_posterior_mean_variance(x_start, x_t, t)` | 计算 `q(x_{t-1} | x_t, x_0)`。 |
| `model_predictions(model, x_t, t, **kwargs)` | 调用模型并统一得到 `pred_noise` 和 `pred_xstart`。 |
| `p_mean_variance(model, x_t, t, **kwargs)` | 计算反向一步的均值和方差。 |
| `p_sample(model, x_t, t, **kwargs)` | DDPM 反向采样一步。 |
| `p_sample_loop(model, shape, noise=None, **kwargs)` | DDPM 完整采样。 |
| `ddim_sample(model, x_t, t, eta=0.0, **kwargs)` | DDIM 采样一步。 |
| `ddim_sample_loop(model, shape, noise=None, eta=0.0, **kwargs)` | DDIM 完整采样。 |
| `training_losses(model, x_start, t=None, noise=None, **kwargs)` | 计算训练 loss。 |
| `sample(model, shape, noise=None, sampler='ddpm')` | 统一采样入口。 |

支持目标：

```text
pred_noise
pred_x0
pred_v
```

### 4.2 超分 DDPM

`GaussianDiffusionSR` 继承 `GaussianDiffusion`，统一接口为：

```python
loss, log_dict = diffusion.training_losses(model, gt, lq)
sr = diffusion.sample(model, lq, shape=None, sampler='ddpm')
```

其中：

- `gt` 是高分辨率 GT。
- `lq` 是低分辨率输入。
- `SuperResModel.forward(x_t, timesteps, lq)` 内部会将 `lq` bicubic 上采样到 `x_t` 尺寸，然后与 `x_t` 在通道维拼接。

## 5. UNet 结构

当前 `unet.py` 保留两个类：

```python
UNetModel
SuperResModel
```

`UNetModel` 是普通 DDPM denoiser。

`SuperResModel` 是超分 denoiser，输入形式为：

```python
model(x_t, timesteps, lq)
```

内部执行：

```python
lq_up = F.interpolate(lq, size=x_t.shape[-2:], mode='bicubic', align_corners=False)
model_input = torch.cat([x_t, lq_up], dim=1)
```

## 6. 少步采样

`respace.py` 保留类 IDDPM 的少步采样逻辑：

```python
space_timesteps(num_timesteps, section_counts)
SpacedDiffusion
SpacedDiffusionSR
```

示例：

```python
use_timesteps = space_timesteps(1000, "ddim50")
diffusion = SpacedDiffusionSR(use_timesteps=use_timesteps, betas=betas)
```

配置上可以理解为：

```yaml
diffusion:
  diffusion_steps: 1000
  timestep_respacing: ddim50
```

## 7. 官方 guided-diffusion 超分配置参考

OpenAI guided-diffusion 官方超分采样脚本使用 `super_res_sample.py`，模型配置核心字段来自
`sr_model_and_diffusion_defaults()` 和 `sr_create_model()`。下面配置只作为 baseline 结构参考，
本项目精简版不兼容这些原始命令行参数。

```text
large_size: 256
small_size: 64
attention_resolutions: 32,16,8
num_channels: 192
num_heads: 4
num_res_blocks: 2
resblock_updown: True
use_scale_shift_norm: True
use_fp16: True
learn_sigma: True
class_cond: False
noise_schedule: linear
diffusion_steps: 1000
```

官方 README 还给出了 128 -> 512 upsampler 的采样配置：

```text
large_size: 512
small_size: 128
attention_resolutions: 32,16
num_channels: 192
num_head_channels: 64
num_res_blocks: 2
resblock_updown: True
use_scale_shift_norm: True
use_fp16: True
learn_sigma: True
class_cond: True
noise_schedule: linear
diffusion_steps: 1000
```

官方采样示例常使用较少步数，例如：

```text
timestep_respacing: 250
```

本项目精简版的对应关系：

| guided-diffusion 字段 | SimpleSR 精简版 |
| --- | --- |
| `large_size` | `large_size`，HR 输出尺寸。 |
| `small_size` | 仅用于数据语义，模型 forward 中直接接收 `lq`。 |
| `num_channels` | `model_channels`。 |
| `num_res_blocks` | `num_res_blocks`。 |
| `num_heads` / `num_head_channels` | 当前保留 `num_heads`，暂不保留 `num_head_channels`。 |
| `resblock_updown` | 已删除，上下采样统一用简化模块。 |
| `use_scale_shift_norm` | 已删除，第一版 baseline 使用普通时间嵌入加法。 |
| `use_fp16=True` | 已删除，混合精度交给 PyTorch AMP 或上层训练框架。 |
| `learn_sigma=True` | 已删除，当前固定方差，`out_channels=3`。 |
| `class_cond=False` | 已删除 classifier 相关逻辑。 |
| `noise_schedule=linear` | 保留。 |
| `diffusion_steps=1000` | 保留。 |
| `timestep_respacing` | 保留到 `SpacedDiffusionSR`。 |

参考来源：

- OpenAI guided-diffusion `README.md` 的 Upsampling 示例。
- OpenAI guided-diffusion `guided_diffusion/script_util.py` 中的 `sr_model_and_diffusion_defaults()`。

## 8. 推荐 baseline 配置

```yaml
network_g:
  name: DDPM-SR-UNet
  path: diffusion.ddpm.unet.SuperResModel
  kwargs:
    image_size: 256
    in_channels: 3
    model_channels: 192
    out_channels: 3
    num_res_blocks: 2
    attention_resolutions: [32, 16, 8]
    channel_mult: [1, 1, 2, 2, 4, 4]
    dropout: 0.0
    num_heads: 4

diffusion:
  path: diffusion.ddpm.gaussian_diffusion.GaussianDiffusionSR
  kwargs:
    diffusion_steps: 1000
    noise_schedule: linear
    objective: pred_noise
    scale: 4
```

少步采样 baseline：

```yaml
diffusion:
  path: diffusion.ddpm.respace.SpacedDiffusionSR
  kwargs:
    diffusion_steps: 1000
    noise_schedule: linear
    timestep_respacing: ddim50
    objective: pred_noise
    scale: 4
```

## 9. 后续可扩展点

建议后续按需要逐步添加：

1. learned variance 和 VLB loss。
2. 更多条件注入方式，例如 feature injection 或 cross attention。
3. PyTorch AMP 训练。
4. Flow Matching / Rectified Flow。
5. 更完整的采样器，如 DDIM eta、DPM-Solver 等。

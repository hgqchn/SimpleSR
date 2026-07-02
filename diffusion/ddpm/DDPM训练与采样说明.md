# DDPM 训练与采样说明

本文说明 `diffusion/ddpm` 下超分辨率 DDPM 的主要训练、采样逻辑，以及 `respace.py` 的作用。

## 1. 当前配置中的扩散对象

当前超分配置将扩散相关配置从普通 `train` / `val` 配置中拆出，放在独立的 `diffusion.train` 和 `diffusion.val` 下：

```yaml
diffusion:
  train:
    path: diffusion.ddpm.gaussian_diffusion.GaussianDiffusionSR
    schedule_sampler: uniform
    kwargs:
      diffusion_steps: 1000
      noise_schedule: linear
      objective: pred_noise
      scale: ${scale}

  val:
    path: diffusion.ddpm.respace.SpacedDiffusionSR
    sampler: ddim
    kwargs:
      diffusion_steps: 1000
      noise_schedule: linear
      timestep_respacing: ddim10
      objective: pred_noise
      scale: ${scale}
```

含义是：

- 训练扩散对象：`GaussianDiffusionSR`
- 训练扩散总步数：`1000`
- 验证/采样扩散对象：`SpacedDiffusionSR`
- 验证/采样基准扩散步数：`1000`
- 验证/采样实际执行步数：`10`
- 验证/采样方式：`DDIM`

也就是说，训练时模型学习完整的 1000 步扩散过程；验证时从这 1000 个时间步中抽出 10 个代表性时间步进行少步采样。

## 2. 训练主逻辑

训练入口在 `DiffusionSRModel.optimize_parameters()`。

主要流程如下：

1. 从 dataloader 读入低分辨率图像 `lq` 和高分辨率目标图像 `gt`。
2. 使用 `diffusion.train` 构造的 `GaussianDiffusionSR` 作为训练扩散对象。
3. `ddpm_training_step()` 通过 `uniform` sampler 为每张图随机采样一个时间步 `t`。
4. `GaussianDiffusionSR.training_losses()` 对 `gt` 加噪，得到 `x_t`。
5. 超分网络接收 `x_t`、时间步 `t` 和条件图像 `lq`。
6. 当前目标是 `pred_noise`，所以网络预测噪声，loss 为预测噪声和真实噪声之间的 MSE。
7. `DiffusionSRModel` 对 loss 做反向传播和优化器更新。

训练阶段的时间步范围是完整的：

```text
t in [0, 999]
```

因此训练阶段不是 10 步训练，而是标准 1000 步 DDPM 训练。

## 3. 采样/验证主逻辑

验证入口在 `DiffusionSRModel.test()`。

主要流程如下：

1. 使用 `diffusion.val` 构造的 `SpacedDiffusionSR` 作为采样扩散对象。
2. 根据配置中的 `sampler` 选择采样方式，当前为 `ddim`。
3. `SpacedDiffusionSR` 根据 `timestep_respacing: ddim10` 从原始 1000 步中选出 10 个时间步。
4. 采样循环只执行这 10 个压缩后的时间步。
5. 每次调用模型前，`SpacedDiffusionSR` 会把压缩后的时间步映射回原始 1000 步时间索引。
6. 模型看到的时间步仍然是训练时对应的原始时间步编号。

因此验证阶段实际采样步数是：

```text
10 steps
```

但这些 10 步不是重新训练出来的新时间系统，而是原始 1000 步扩散过程的子序列。

## 4. respace.py 的作用

`respace.py` 的核心作用是：在不重新训练模型的前提下，把原始扩散过程压缩成更少的采样步数。

它主要做两件事。

第一，选择要保留的原始时间步。

例如：

```yaml
timestep_respacing: ddim10
```

表示从原始 `1000` 个时间步中选出 `10` 个时间步用于采样。

第二，维护压缩时间步到原始时间步的映射。

采样循环内部看到的是短时间步：

```text
0, 1, 2, ..., 9
```

但模型实际接收的是映射后的原始时间步，例如：

```text
0, 100, 200, ..., 900
```

这样做的原因是：模型训练时学习的是原始 1000 步时间编码。如果验证时直接把 `0..9` 传给模型，时间条件就和训练分布不一致。

## 5. 使用 respace 是否必须用 DDIM

不是必须。

`respace` 只负责减少采样时间步，并把短时间步映射回原始时间步。它本身不等于 DDIM，也不强制必须使用 DDIM。

在当前实现中，`SpacedDiffusionSR` 可以配合两种采样方式：

- `sampler: ddpm`
- `sampler: ddim`

区别是：

- `ddpm`：使用随机的反向扩散采样逻辑，同样可以在 respace 后的少步时间表上运行。
- `ddim`：使用 DDIM 采样逻辑，通常更适合少步、快速、相对稳定的验证采样。

当前配置选择：

```yaml
sampler: ddim
timestep_respacing: ddim10
```

是因为验证时希望用 10 步快速采样。它是推荐搭配，但不是语法或实现上的硬性要求。

如果改成：

```yaml
sampler: ddpm
```

仍然会使用 `SpacedDiffusionSR` 的 10 个时间步进行采样，只是反向过程会走 DDPM 的随机采样公式。

## 6. 当前训练与验证步数总结

当前配置下：

| 阶段 | 扩散类 | 配置项 | 实际步数 | 说明 |
| --- | --- | --- | --- | --- |
| 训练 | `GaussianDiffusionSR` | `diffusion_steps: 1000` | 1000 | 每个 batch 随机采样一个 `0..999` 的时间步训练 |
| 验证/采样 | `SpacedDiffusionSR` | `diffusion_steps: 1000`, `timestep_respacing: ddim10` | 10 | 从原始 1000 步中抽出 10 步采样 |

最重要的一点是：

```text
训练步数 = 1000
验证采样步数 = 10
```

`respace` 只影响验证/采样阶段的反向扩散步数，不改变训练阶段的 1000 步扩散学习目标。

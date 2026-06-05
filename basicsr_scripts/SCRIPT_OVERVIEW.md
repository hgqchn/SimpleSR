# basicsr_scripts 脚本说明

本文档概述 `basicsr_scripts` 目录下各脚本文件的作用，便于快速判断哪些脚本用于训练、测试、数据准备、指标计算、模型转换或功能验证。

## 目录概览

- `scripts/`: 训练、测试、下载、发布、数据准备、指标计算、模型转换、绘图等实用脚本。
- `test_scripts/`: 数据集、指标、学习率调度器、判别器反向传播等功能验证脚本。

## scripts

### 顶层脚本

#### `scripts/dist_train.sh`
- 用途：启动 BasicSR 的分布式训练。
- 作用：通过 `torch.distributed.launch` 调用 `basicsr/train.py`，传入配置文件和 GPU 数量。
- 适用场景：多卡训练超分、去噪、视频复原等模型。

#### `scripts/dist_test.sh`
- 用途：启动 BasicSR 的分布式测试。
- 作用：通过 `torch.distributed.launch` 调用 `basicsr/test.py`，传入配置文件和 GPU 数量。
- 适用场景：多卡评测或批量推理。

#### `scripts/download_gdrive.py`
- 用途：从 Google Drive 下载单个文件。
- 作用：接收文件 ID 和输出路径，调用 `basicsr.utils.download_util.download_file_from_google_drive` 完成下载。
- 适用场景：手动下载某个权重或数据文件。
- 备注：脚本参数名是 `--output`，但代码里实际调用的是 `args.save_path`，当前实现存在参数名不一致的问题。

#### `scripts/download_pretrained_models.py`
- 用途：批量下载官方预训练模型。
- 作用：内置多组 Google Drive 文件 ID，按模型类别下载到 `experiments/pretrained_models/<method>/`。
- 支持类别：如 `ESRGAN`、`EDVR`、`StyleGAN`、`EDSR`、`DUF`、`DFDNet`、`BasicVSR` 等。
- 适用场景：初始化实验环境、获取官方基线权重。

#### `scripts/publish_models.py`
- 用途：整理和发布预训练模型文件。
- 作用：
- 将 `.pth` 模型重新保存为兼容旧版 PyTorch 的格式。
- 计算模型文件的 SHA256 前 8 位，并将其附加到文件名中。
- 对模型内容做简单检查，确认是否包含 `params` 或 `params_ema`。
- 适用场景：准备对外发布模型文件，统一命名和兼容性。

### `scripts/data_preparation`

#### `scripts/data_preparation/create_lmdb.py`
- 用途：把图像数据集转换成 LMDB。
- 作用：
- 为 DIV2K 生成 HR/LR 子图对应的 LMDB。
- 为 REDS 生成训练数据 LMDB。
- 自动扫描图像并生成 LMDB key。
- 适用场景：训练前将磁盘图像整理为 LMDB，以提升读取效率。

#### `scripts/data_preparation/download_datasets.py`
- 用途：下载示例数据集。
- 作用：内置 Google Drive 链接，下载 `Set5`、`Set14` 等数据，并在下载后自动解压。
- 适用场景：快速准备常用评测集。

#### `scripts/data_preparation/extract_images_from_tfrecords.py`
- 用途：从 TFRecords 中提取图像，或直接写入 LMDB。
- 作用：
- 支持 CelebA、FFHQ 等 TFRecords 数据格式。
- 可输出为普通 PNG 图像目录，或直接生成 LMDB。
- 适用场景：处理 StyleGAN/人脸相关数据集的官方 TFRecords 数据。

#### `scripts/data_preparation/extract_subimages.py`
- 用途：把大图裁成小块子图。
- 作用：
- 采用多进程滑窗裁剪图像。
- 默认针对 DIV2K，分别处理 HR 和不同倍率 LR 图像。
- 可设置裁剪尺寸、步长、压缩等级、线程数等参数。
- 适用场景：超分训练前做 patch 化预处理，加快训练时 IO。

#### `scripts/data_preparation/generate_meta_info.py`
- 用途：生成数据集元信息文件。
- 作用：遍历 DIV2K 子图目录，记录文件名、尺寸、通道数，并写入 `basicsr/data/meta_info/`。
- 适用场景：配合 `meta_info_file` 方式读取数据集。

#### `scripts/data_preparation/prepare_hifacegan_dataset.py`
- 用途：生成 HiFaceGAN 风格的数据退化结果。
- 作用：
- 定义多种退化方式，如缩放、噪声、模糊、JPEG 压缩、16x16 马赛克。
- 可根据退化模板，从 GT 图像生成 LQ 图像。
- 适用场景：构造人脸复原任务的训练/测试数据对。

#### `scripts/data_preparation/regroup_reds_dataset.py`
- 用途：重新整理 REDS 数据集目录。
- 作用：把原始验证集拷贝到训练集目录，并把验证片段编号平移到 `240-269`。
- 适用场景：按 BasicSR 的 REDS 数据划分方式统一训练/验证结构。
- 备注：脚本使用 `cp -r`，更适合类 Unix 环境。

### `scripts/metrics`

#### `scripts/metrics/calculate_fid_folder.py`
- 用途：计算一个图像文件夹相对于给定统计量的 FID。
- 作用：
- 将文件夹封装为 `SingleImageDataset`。
- 提取 Inception 特征。
- 读取预先保存的真实数据统计量并计算 FID。
- 适用场景：评估生成图像或复原结果的分布质量。

#### `scripts/metrics/calculate_fid_stats_from_datasets.py`
- 用途：从数据集本身计算 FID 统计量。
- 作用：
- 构建数据集并提取 Inception 特征。
- 计算均值和协方差。
- 保存为 `inception_<name>_<size>.pth`。
- 适用场景：先为真实数据集生成 FID 参考统计文件，再供其他脚本复用。

#### `scripts/metrics/calculate_lpips.py`
- 用途：计算两组图像之间的 LPIPS 感知距离。
- 作用：
- 读取 GT 与恢复图像。
- 做归一化后调用 `lpips` 模型逐张计算。
- 输出逐图和平均 LPIPS。
- 适用场景：评估感知质量，尤其是人脸或生成图像结果。

#### `scripts/metrics/calculate_niqe.py`
- 用途：计算一个目录中图像的 NIQE 无参考质量分数。
- 作用：遍历输入目录，逐张计算 NIQE，并输出平均值。
- 适用场景：无 GT 条件下评估图像自然度。

#### `scripts/metrics/calculate_psnr_ssim.py`
- 用途：计算 GT 与恢复图像之间的 PSNR 和 SSIM。
- 作用：
- 支持 RGB 或 Y 通道评测。
- 支持裁边。
- 支持通过文件后缀匹配恢复图像。
- 可选对恢复图像做均值方差校正。
- 适用场景：传统图像复原任务的标准全参考评测。

#### `scripts/metrics/calculate_stylegan2_fid.py`
- 用途：计算 StyleGAN2 生成器的 FID。
- 作用：
- 加载 StyleGAN2 生成器权重。
- 随机采样 latent 生成图像。
- 提取 Inception 特征并与真实数据统计量比较。
- 适用场景：评估生成模型质量。

### `scripts/model_conversion`

#### `scripts/model_conversion/convert_dfdnet.py`
- 用途：把旧版或官方 DFDNet 权重映射到当前 BasicSR 的网络定义。
- 作用：按参数名规则逐项转换，包括 VGG 特征提取器、注意力模块、多尺度膨胀模块、上采样模块等。
- 适用场景：迁移 DFDNet 官方权重到当前仓库。

#### `scripts/model_conversion/convert_models.py`
- 用途：转换 EDVR 相关模型权重。
- 作用：将旧实现中的参数命名映射到当前 EDVR 网络结构命名，并保存新权重文件。
- 适用场景：兼容历史版本或第三方 EDVR 权重。

#### `scripts/model_conversion/convert_ridnet.py`
- 用途：转换 RIDNet 官方权重。
- 作用：按当前 `RIDNet` 模型的参数顺序重组旧 checkpoint，并保存为新的 `.pth`。
- 适用场景：导入 RIDNet 官方发布模型。

#### `scripts/model_conversion/convert_stylegan.py`
- 用途：转换 StyleGAN2 的生成器和判别器权重。
- 作用：
- 将 `stylegan2-pytorch` 风格的参数命名映射到 BasicSR 中的 `StyleGAN2Generator` 和 `StyleGAN2Discriminator`。
- 分别导出生成器和判别器的新权重文件。
- 适用场景：导入官方或外部 StyleGAN2 权重。

### `scripts/plot`

#### `scripts/plot/model_complexity_cmp_bsrn.py`
- 用途：绘制 BSRN 与其他超分模型的复杂度对比图。
- 作用：
- 用散点图展示参数量、PSNR、Multi-Adds 三者关系。
- 手工标注模型名并输出 `model_complexity_cmp_bsrn.png`。
- 适用场景：论文图表复现或修改展示。

#### `scripts/plot/README.md`
- 用途：说明绘图脚本的来源和相关工具。
- 作用：介绍 `plot` 目录的用法，并给出示例图和论文链接。

### `scripts/matlab_scripts`

#### `scripts/matlab_scripts/generate_bicubic_img.m`
- 用途：为图像超分任务生成标准的 GT/LR/Bicubic 数据。
- 作用：
- 对原图做 `modcrop`。
- 生成 bicubic 下采样图像。
- 可选再生成 bicubic 上采样图像。
- 适用场景：构建 Set5/Set14 等经典超分评测数据格式。

#### `scripts/matlab_scripts/generate_LR_Vimeo90K.m`
- 用途：为 Vimeo90K 生成 bicubic 下采样 LR 图像。
- 作用：遍历 septuplet 图像序列，对每帧做 `modcrop` 和 4x bicubic 下采样，并写到新目录。
- 适用场景：为视频超分任务准备 Vimeo90K 的 LR 输入。

#### `scripts/matlab_scripts/back_projection/backprojection.m`
- 用途：实现反投影后处理算法。
- 作用：通过低分辨率重投影误差迭代修正高分辨率输出。
- 适用场景：对超分结果做传统后处理，提高与 LR 观测的一致性。

#### `scripts/matlab_scripts/back_projection/main_bp.m`
- 用途：批量执行反投影后处理。
- 作用：读取 LR 图像和已有 SR 结果，调用 `backprojection.m` 迭代修正后保存。
- 适用场景：离线批处理超分结果。

#### `scripts/matlab_scripts/back_projection/main_reverse_filter.m`
- 用途：批量执行基于迭代误差补偿的反向滤波后处理。
- 作用：对已有结果做多轮上采样/下采样误差修正，并输出新结果。
- 适用场景：传统后处理实验，对比不同迭代修正策略。

## test_scripts

### `test_scripts/test_discriminator_backward.py`
- 用途：验证判别器反向传播两次与一次求和反传的梯度是否一致。
- 作用：构造一个简单判别器，对两种反传方式的梯度做逐参数比较。
- 适用场景：理解 GAN 训练实现细节，排查梯度累积逻辑。

### `test_scripts/test_ffhq_dataset.py`
- 用途：验证 `FFHQDataset` 的读取是否正确。
- 作用：
- 构建 FFHQ LMDB 数据集和 DataLoader。
- 读取若干 batch。
- 把可视化结果写入 `tmp/`。
- 适用场景：检查 FFHQ 数据预处理和数据管线。

### `test_scripts/test_lr_scheduler.py`
- 用途：验证并可视化 `CosineAnnealingRestartLR` 学习率曲线。
- 作用：模拟长迭代训练过程，记录学习率并保存曲线图 `test_lr_scheduler.png`。
- 适用场景：检查学习率策略是否符合预期。

### `test_scripts/test_niqe.py`
- 用途：验证 NIQE 指标实现。
- 作用：读取示例图像并输出 NIQE 分数。
- 适用场景：快速检查环境中的 NIQE 计算是否可用。

### `test_scripts/test_paired_image_dataset.py`
- 用途：验证 `PairedImageDataset` 的读取与配对逻辑。
- 作用：
- 支持 `folder`、`meta_info_file`、`lmdb` 三种模式。
- 读取 DIV2K 的 GT/LQ 配对样本，并将结果保存到 `tmp/`。
- 适用场景：检查配对图像数据集配置是否正确。

### `test_scripts/test_reds_dataset.py`
- 用途：验证 `REDSDataset` 的读取逻辑。
- 作用：
- 支持 `folder` 和 `lmdb` 两种模式。
- 读取 REDS 多帧样本，并将各帧 LQ 与 GT 保存到 `tmp/`。
- 适用场景：检查视频超分训练数据管线。

### `test_scripts/test_vimeo90k_dataset.py`
- 用途：验证 `Vimeo90KDataset` 的读取逻辑。
- 作用：
- 支持 `folder` 和 `lmdb` 两种模式。
- 读取 Vimeo90K 多帧样本，并导出可视化结果。
- 适用场景：检查视频复原/视频超分数据管线。

## 补充说明

- 这些脚本里有不少默认路径是硬编码的，正式使用前通常需要按本地数据目录修改。
- 部分脚本依赖 Linux/Unix 命令，如 `cp`、`mv`、`sha256sum`，在 Windows 下直接运行可能需要调整。
- `test_scripts/data/baboon.png` 是 `test_niqe.py` 使用的示例图片，不属于脚本文件本身。

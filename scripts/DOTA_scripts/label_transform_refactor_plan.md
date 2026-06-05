# DOTA 标签与图像变换整合计划

## 目标

当前标签缩放、8 种模式图像增强、任意角度旋转及其对应标签变换已经能按预期运行，但相关逻辑分散在多个文件中，存在重复读取标签文件、重复拆装 DOTA 坐标、测试脚本流程过长等问题。

本次整理目标是：

- 保留当前图像变换函数接口，不改核心行为。
- 统一 DOTA 标签变换逻辑，使用 numpy 数组和仿射矩阵表达。
- 将可复用逻辑抽成对象级函数，减少文件路径函数导致的重复读写。
- 简化 `generate_imgs.py`，让它只作为验证流程脚本。

## 当前涉及文件

- `simplesr/utils/img_utils.py`
- `DOTA_utils/utils.py`
- `DOTA_utils/dota_visual.py`
- `scripts/DOTA_scripts/generate_transformed_label.py`
- `scripts/DOTA_scripts/generate_lr_DOTA_label.py`
- `scripts/test_label_transform/generate_imgs.py`

## 职责划分

### `simplesr/utils/img_utils.py`

保留图像与通用坐标变换能力：

- `img_augment(img, mode=0)`
- `img_rotate(img, angle, center=None, scale=1.0)`
- `transform_points_by_affine(points, matrix)`
- `get_scale_affine_matrix(scale_x, scale_y)`
- `get_img_augment_affine_matrix(mode, img_shape)`
- `get_img_rotate_affine_matrix(img_shape, angle, center=None, scale=1.0)`

建议新增：

```python
def transform_polygons_by_affine(polygons, matrix):
    """变换 DOTA 多边形数组。

    输入:
        polygons: shape 为 (4, 2) 或 (N, 4, 2)
        matrix: shape 为 (2, 3)

    输出:
        与 polygons 输入 shape 一致。
    """
```

这样 `transform_dota_polygon_by_affine` 可以作为薄包装，或者直接被替换。

### `DOTA_utils/utils.py`

负责 DOTA 标签基础 IO 和对象转换：

- `parse_dota_label(label_path)`
- `write_dota_label(label_path, objects)`
- `format_number(x)`
- `collect_images(img_dir)`

建议新增：

```python
def dota_objects_to_polygons(objects):
    """list[dict] -> np.ndarray。

    输入:
        objects: 长度 N，每个 polygon shape 为 (4, 2)

    输出:
        polygons: shape 为 (N, 4, 2)
    """
```

```python
def update_dota_objects_polygons(objects, polygons):
    """将变换后的 polygon 写回 DOTA objects。

    输入:
        objects: 长度 N
        polygons: shape 为 (N, 4, 2)

    输出:
        new_objects: 长度 N，class_name / difficult 等字段保持不变
    """
```

```python
def filter_small_dota_objects(objects, min_box_size=None):
    """过滤外接矩形过小的 DOTA 目标。

    输入:
        objects: list[dict]
        min_box_size: None 或最小宽高阈值

    输出:
        filtered_objects: list[dict]
    """
```

### `scripts/DOTA_scripts/generate_transformed_label.py`

作为标签变换的文件级与对象级封装入口。

建议新增统一对象级函数：

```python
def transform_dota_objects(objects, matrix, min_box_size=None):
    """统一变换 DOTA objects。

    输入:
        objects: list[dict]
        matrix: shape 为 (2, 3)

    输出:
        transformed_objects: list[dict]
    """
```

建议新增统一文件级函数：

```python
def transform_dota_label_file(src_label_path, dst_label_path, matrix, min_box_size=None):
    """读取 DOTA 标签文件，应用仿射矩阵，写出新标签文件。"""
```

然后已有功能改成薄包装：

```python
def augment_dota_label_file(src_label_path, dst_label_path, img_shape, mode):
    matrix = get_img_augment_affine_matrix(mode, img_shape)
    transform_dota_label_file(src_label_path, dst_label_path, matrix)
```

```python
def rotate_dota_label_file(src_label_path, dst_label_path, img_shape, angle, center=None, scale=1.0):
    matrix = get_img_rotate_affine_matrix(img_shape, angle, center=center, scale=scale)
    transform_dota_label_file(src_label_path, dst_label_path, matrix)
```

```python
def scale_dota_label_file(src_label_path, dst_label_path, scale_x, scale_y, min_box_size=None):
    matrix = get_scale_affine_matrix(scale_x, scale_y)
    transform_dota_label_file(src_label_path, dst_label_path, matrix, min_box_size=min_box_size)
```

### `scripts/DOTA_scripts/generate_lr_DOTA_label.py`

保留批量缩放入口：

```python
def scale_dota_label_folder(...)
```

但内部不再逐行解析坐标，改为调用统一的 `scale_dota_label_file` 或 `transform_dota_label_file`。

该文件可以作为“批量生成 LR 标签”的命令脚本，不再承载底层坐标变换实现。

### `scripts/test_label_transform/generate_imgs.py`

当前脚本将裁剪、缩放、增强、旋转、写图、写标签全部堆在 `main` 中。

建议拆成以下流程函数：

```python
def prepare_output_dirs(output_dir):
    """创建 imgs / labels 输出目录。"""
```

```python
def generate_center_crop_sample(img, objects, crop_size, mode="center"):
    """生成固定大小裁剪图像和对应 DOTA 标签。

    输入:
        img: shape 为 (H, W, C)
        objects: list[dict]
        crop_size: (crop_h, crop_w)

    输出:
        cropped_img: shape 为 (crop_h, crop_w, C)
        cropped_objects: list[dict]
    """
```

```python
def generate_scaled_sample(img, objects, scale):
    """生成缩放图像和对应标签对象。

    输入:
        img: shape 为 (H, W, C)
        objects: list[dict]

    输出:
        scaled_img: shape 由缩放函数决定
        scaled_objects: list[dict]
    """
```

```python
def generate_augment_samples(img, objects, modes=range(8)):
    """生成 8 种 img_augment 图像和标签。"""
```

```python
def generate_rotate_samples(img, objects, angles):
    """生成任意角度旋转图像和标签。"""
```

核心调整是：测试脚本优先使用 `objects`，不要为了标签变换反复写出再读取中间标签文件。

### `DOTA_utils/dota_visual.py`

保持轻量职责：

- 读取图像
- 读取标签
- 绘制 polygon
- 绘制类别文字

不放标签变换逻辑。

可选优化：

```python
def visualize_dota_objects(image, objects, save_path=None, ...):
    """直接可视化已在内存中的 objects，避免必须从 label_path 读取。"""
```

这样 `generate_imgs.py` 调试时可以直接可视化内存对象。

## 推荐实施顺序

1. 在 `DOTA_utils/utils.py` 增加 objects 与 polygons 的转换函数，以及小目标过滤函数。
2. 在 `img_utils.py` 增加 `transform_polygons_by_affine`，简化 DOTA 标签变换包装。
3. 在 `generate_transformed_label.py` 增加统一的对象级和文件级标签变换函数。
4. 让 `generate_lr_DOTA_label.py` 复用统一文件级缩放函数。
5. 重构 `generate_imgs.py`，将裁剪、缩放、增强、旋转拆成函数，并优先传递图像数组和标签对象。
6. 可选增加 `visualize_dota_objects`，方便调试内存中的增强结果。

## 预期结果

整理后，底层坐标变换只有一条路径：

```python
points / polygons + affine_matrix -> transformed points / polygons
```

DOTA 标签变换只有一条对象级路径：

```python
objects -> polygons -> affine transform -> objects
```

文件级函数只负责：

```python
read label -> transform objects -> write label
```

测试脚本只负责组织流程：

```python
read image/label -> crop -> scale/augment/rotate -> write outputs
```

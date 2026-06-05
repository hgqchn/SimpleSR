import importlib
from functools import partial
import os
from typing import Any, Mapping, Sequence

def get_object_path(obj):
    """获取对象所属类型的导入路径字符串。

    输入
    ----
    obj:
        任意 Python 对象。

    输出
    ----
    str
        对象类型路径，格式通常为 ``模块名.类名``。
    """
    return obj.__module__ + "." + obj.__class__.__name__


def get_object_from_path(obj_path):
    """根据 ``模块路径.对象名`` 的字符串导入对象。

    输入
    ----
    obj_path: str
        对象路径字符串，例如 ``package.module.ClassName``。

    输出
    ----
    Any
        导入到的对象（可能是类、函数、变量等）。
    """
    module_path, object_name = obj_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"{e} 无法导入模块 {module_path}")
    try:
        obj = getattr(module, object_name)
    except AttributeError as e:
        raise AttributeError(f"{e} 模块 {module_path} 中不存在 {object_name} 的对象")
    return obj


def instantiate_object(class_path, *args, **kwargs):
    """根据类路径导入并实例化对象。

    输入
    ----
    class_path: str
        类路径字符串，例如 ``package.module.ClassName``。
    *args:
        传给类构造函数的位置参数。
    **kwargs:
        传给类构造函数的关键字参数。

    输出
    ----
    Any
        实例化后的对象实例。
    """
    cls = get_object_from_path(class_path)
    if not isinstance(cls, type):
        raise TypeError(f"{class_path} 不是一个类，而是 {type(cls)}")
    try:
        instance = cls(*args, **kwargs)
        return instance
    except Exception as e:
        raise TypeError(f"实例化类 {class_path} 失败: {e}")


def partial_function(func, *args, **kwargs):
    """把函数（或函数路径）封装为 ``functools.partial`` 对象。

    输入
    ----
    func: Callable | str
        可调用对象，或其导入路径字符串。``package.module.FunctionName``
    *args:
        预绑定的位置参数。
    **kwargs:
        预绑定的关键字参数。

    输出
    ----
    functools.partial
        绑定了部分参数的可调用对象。
    """
    if isinstance(func, str):
        func = get_object_from_path(func)
    if not callable(func):
        raise TypeError(f"{func}不是可调用对象")
    try:
        return partial(func, *args, **kwargs)
    except Exception as e:
        raise TypeError(f"partial 失败: {e}")


def instantiate_model_from_cfg(model_config: Mapping[str, Any]):
    """根据配置文件中的 ``model_config`` 字段实例化模型或对象。

    输入
    ----
    model_config: Mapping[str, Any]
        模型配置字典，必须包含 ``model_path`` 或 ``path`` 字段指定类路径。
        可选的 ``model_kwargs`` 字段提供初始化参数。
    ----
    Any
        根据配置实例化得到的对象。
    """
    if "model_path" in model_config:
        model_path = model_config["model_path"]
    elif "path" in model_config:
        model_path = model_config["path"]
    else:
        raise KeyError("配置文件中缺少 model_path 或 path")

    kwargs = model_config.get("model_kwargs") or {}
    return instantiate_object(model_path, **kwargs)

def get_by_dotted_key(config: Mapping[str, Any], key: str) -> Any:
    """根据点分隔 key 从嵌套字典中取值。

    输入
    ----
    config: Mapping[str, Any]
        嵌套配置字典。
    key: str
        点分隔的层级键，例如 ``"model.generator"``。

    输出
    ----
    Any
        对应 key 的值，可以是字典、列表、标量或其他对象。

    示例
    ----
    >>> cfg = {"model": {"generator": {"path": "models.UNet"}}}
    >>> get_by_dotted_key(cfg, "model.generator")
    {'path': 'models.UNet'}
    """
    current: Any = config

    for part in key.split("."):
        if not isinstance(current, Mapping):
            raise KeyError(f"无法继续索引 '{part}'，当前对象不是字典: {current}")

        if part not in current:
            raise KeyError(f"配置中不存在 key: {key}")

        current = current[part]

    return current


def instantiate_from_config(
    config: Mapping[str, Any],
    *,
    path_keys: Sequence[str] = (
        "target",
        "path",
        "object_path",
        "class_path",
        "model_path",
    ),
    kwargs_keys: Sequence[str] = (
        "params",
        "kwargs",
        "model_kwargs",
        "init_args",
    ),
    extra_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """根据配置字典实例化 Python 对象。

    支持多种字段名，例如:
    - target / path / object_path / class_path / model_path
    - params / kwargs / model_kwargs / init_args

    输入
    ----
    config: Mapping[str, Any]
        对象配置字典。
    path_keys: Sequence[str]
        用于寻找对象路径的候选字段名。
    kwargs_keys: Sequence[str]
        用于寻找初始化参数的候选字段名。
    extra_kwargs: Mapping[str, Any] | None
        额外传入的初始化参数；会覆盖配置中的同名参数。

    输出
    ----
    Any
        实例化后的对象。
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"config 必须是字典类型，但得到: {type(config)}")

    object_path = None
    for path_key in path_keys:
        if path_key in config:
            object_path = config[path_key]
            break

    if object_path is None:
        raise KeyError(
            f"配置中缺少对象路径字段，支持的字段包括: {tuple(path_keys)}"
        )

    kwargs: dict[str, Any] = {}
    for kwargs_key in kwargs_keys:
        if kwargs_key in config:
            raw_kwargs = config[kwargs_key]
            if raw_kwargs is None:
                raw_kwargs = {}

            if not isinstance(raw_kwargs, Mapping):
                raise TypeError(
                    f"'{kwargs_key}' 必须是字典类型，但得到: {type(raw_kwargs)}"
                )

            kwargs.update(raw_kwargs)
            break

    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)

    return instantiate_object(object_path, **kwargs)




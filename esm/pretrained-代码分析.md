## pretrained-代码分析
以上代码 `pretrained.py` 实现了一个本地预训练模型的注册与加载机制，主要用于管理和加载不同版本的 ESM3 和 ESMC 模型。以下是对代码各部分功能的详细分析：

### 1. 导入模块和类型定义

```python
from typing import Callable

import torch
import torch.nn as nn

from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.tokenization import get_model_tokenizers
from esm.utils.constants.esm3 import data_root
from esm.utils.constants.models import (
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
    ESMC_300M,
    ESMC_600M,
)
```

- **类型导入**：
  - `Callable` 用于类型注解，表示可以调用的对象。
- **PyTorch 导入**：
  - `torch` 和 `torch.nn` 提供了深度学习相关的功能。
- **ESM 模型导入**：
  - 导入了 ESM3 和 ESMC 模型及其相关的解码器和编码器。
- **其他工具函数和常量**：
  - `get_model_tokenizers` 用于获取模型的分词器。
  - `data_root` 用于获取数据根目录路径。
  - 导入了一系列模型名称常量，用于后续的模型注册和加载。

### 2. 类型别名定义

```python
ModelBuilder = Callable[[torch.device | str], nn.Module]
```

- 定义了 `ModelBuilder` 类型，表示一个接受 `torch.device` 或 `str` 类型参数，并返回 `nn.Module` 的可调用对象（通常是一个函数）。

### 3. 模型构建函数

以下函数分别定义了不同版本的 ESM3 和 ESMC 模型的构建过程：

#### ESM3 结构编码器

```python
def ESM3_structure_encoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_structure_encoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model
```

- **功能**：
  - 创建 `StructureTokenEncoder` 模型实例，设置相关参数。
  - 将模型设为评估模式 (`eval`)。
  - 从指定路径加载预训练权重 (`.pth` 文件)。
  - 将权重加载到模型中并返回模型。

#### ESM3 结构解码器

```python
def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_structure_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model
```

- **功能**：
  - 创建 `StructureTokenDecoder` 模型实例，设置相关参数。
  - 将模型设为评估模式 (`eval`)。
  - 加载预训练权重并返回模型。

#### ESM3 功能解码器

```python
def ESM3_function_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = FunctionTokenDecoder().eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_function_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model
```

- **功能**：
  - 创建 `FunctionTokenDecoder` 模型实例。
  - 将模型设为评估模式 (`eval`)。
  - 加载预训练权重并返回模型。

#### ESMC 300M 版本

```python
def ESMC_300M_202412(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = ESMC(
            d_model=960,
            n_heads=15,
            n_layers=30,
            tokenizer=get_model_tokenizers(ESM3_OPEN_SMALL).sequence,
        ).eval()
    state_dict = torch.load(
        data_root("esmc-300") / "data/weights/esmc_300m_2024_12_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)

    return model
```

- **功能**：
  - 创建 `ESMC` 模型实例，设置模型参数并获取相应的分词器。
  - 将模型设为评估模式 (`eval`)。
  - 加载预训练权重并返回模型。

#### ESMC 600M 版本

```python
def ESMC_600M_202412(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = ESMC(
            d_model=1152,
            n_heads=18,
            n_layers=36,
            tokenizer=get_model_tokenizers(ESM3_OPEN_SMALL).sequence,
        ).eval()
    state_dict = torch.load(
        data_root("esmc-600") / "data/weights/esmc_600m_2024_12_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)

    return model
```

- **功能**：
  - 创建 `ESMC` 模型实例，设置更高的模型参数。
  - 将模型设为评估模式 (`eval`)。
  - 加载预训练权重并返回模型。

#### ESM3 开放小模型

```python
def ESM3_sm_open_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=get_model_tokenizers(ESM3_OPEN_SMALL),
        ).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_sm_open_v1.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model
```

- **功能**：
  - 创建 `ESM3` 模型实例，设置更高的模型参数，并传入之前定义的结构编码器、结构解码器和功能解码器函数。
  - 获取相应的分词器。
  - 将模型设为评估模式 (`eval`)。
  - 加载预训练权重并返回模型。

### 4. 本地模型注册表

```python
LOCAL_MODEL_REGISTRY: dict[str, ModelBuilder] = {
    ESM3_OPEN_SMALL: ESM3_sm_open_v0,
    ESM3_STRUCTURE_ENCODER_V0: ESM3_structure_encoder_v0,
    ESM3_STRUCTURE_DECODER_V0: ESM3_structure_decoder_v0,
    ESM3_FUNCTION_DECODER_V0: ESM3_function_decoder_v0,
    ESMC_600M: ESMC_600M_202412,
    ESMC_300M: ESMC_300M_202412,
}
```

- **功能**：
  - 定义一个字典 `LOCAL_MODEL_REGISTRY`，键为模型名称字符串，值为对应的模型构建函数。
  - 该注册表用于管理和快速访问不同的预训练模型。

### 5. 加载本地模型的函数

```python
def load_local_model(
    model_name: str, device: torch.device = torch.device("cpu")
) -> nn.Module:
    if model_name not in LOCAL_MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in local model registry.")
    return LOCAL_MODEL_REGISTRY[model_name](device)
```

- **功能**：
  - 根据模型名称 `model_name` 和设备 `device` 加载相应的模型。
  - 首先检查模型名称是否在 `LOCAL_MODEL_REGISTRY` 中，如果不存在则抛出错误。
  - 如果存在，则调用对应的模型构建函数，传入设备参数，返回加载好的模型实例。

### 6. 注册新的本地模型

```python
def register_local_model(model_name: str, model_builder: ModelBuilder) -> None:
    LOCAL_MODEL_REGISTRY[model_name] = model_builder
```

- **功能**：
  - 允许用户动态地向 `LOCAL_MODEL_REGISTRY` 中添加新的模型。
  - 接受模型名称 `model_name` 和对应的构建函数 `model_builder`，将其添加到注册表中。

### 总结

整个 `pretrained.py` 文件的主要功能是：

1. **定义不同版本的 ESM3 和 ESMC 模型的构建函数**，每个函数负责创建模型实例、加载预训练权重并返回模型。
2. **建立一个本地模型注册表**，将模型名称与对应的构建函数关联起来，方便统一管理和调用。
3. **提供加载模型的接口**，通过 `load_local_model` 函数，可以根据模型名称和设备快速加载所需的预训练模型。
4. **支持动态注册新模型**，通过 `register_local_model` 函数，可以向注册表中添加自定义的模型构建函数，扩展模型管理的灵活性。

这种设计使得管理和使用多个预训练模型变得简洁且高效，特别适用于需要频繁切换和加载不同模型的场景。

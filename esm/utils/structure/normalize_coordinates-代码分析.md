## normalize_coordinates-代码分析
这段代码 `normalize_coordinates.py` 主要用于对蛋白质的三维坐标进行标准化处理。标准化的目的是通过构建一个参考坐标系（基于蛋白质骨架的N、CA、C原子），将蛋白质的坐标进行平移和旋转，使其在统一的参考系下表示，便于后续的分析和比较。以下是对代码的详细中文分析：

## 导入模块

```python
from typing import TypeVar

import numpy as np
import torch
from torch import Tensor

from esm.utils import residue_constants as RC
from esm.utils.structure.affine3d import Affine3D
```

- **`typing.TypeVar`**：用于定义类型变量，增强代码的类型提示。
- **`numpy as np`**：引入NumPy库，主要用于数组操作。
- **`torch` 和 `Tensor`**：引入PyTorch库及其Tensor类型，用于高效的张量计算。
- **`esm.utils.residue_constants as RC`**：引入蛋白质残基的常量定义，通常包括原子的序号、名称等信息。
- **`esm.utils.structure.affine3d import Affine3D`**：引入3D仿射变换工具，用于处理坐标系的旋转和平移。

## 类型定义

```python
ArrayOrTensor = TypeVar("ArrayOrTensor", np.ndarray, Tensor)
```

- **`ArrayOrTensor`**：定义一个类型变量，可以是NumPy数组 (`np.ndarray`) 或 PyTorch的 `Tensor`。用于函数参数和返回值的类型注解，增加代码的灵活性和可读性。

## 函数定义

### 1. `atom3_to_backbone_frames`

```python
def atom3_to_backbone_frames(bb_positions: torch.Tensor) -> Affine3D:
    N, CA, C = bb_positions.unbind(dim=-2)
    return Affine3D.from_graham_schmidt(C, CA, N)
```

- **功能**：根据蛋白质骨架的N、CA、C原子的位置，构建一个3D仿射坐标系（Affine3D）。
- **参数**：
  - `bb_positions`：形状为 `[L, 3, 3]` 的张量，代表蛋白质的N、CA、C三个原子的坐标，其中 `L` 是蛋白质的长度（氨基酸残基数）。
- **实现步骤**：
  1. 使用 `unbind(dim=-2)` 将 `bb_positions` 在倒数第二个维度上拆分为三个独立的张量，分别对应N、CA、C原子的坐标。
  2. 调用 `Affine3D.from_graham_schmidt(C, CA, N)`，利用格拉汉-施密特正交化算法构建坐标系。具体来说，以C原子为基准，CA原子为参考，N原子用于定义坐标系的方向。
- **返回值**：一个 `Affine3D` 对象，代表构建好的坐标系。

### 2. `index_by_atom_name`

```python
def index_by_atom_name(
    atom37: ArrayOrTensor, atom_names: str | list[str], dim: int = -2
) -> ArrayOrTensor:
    squeeze = False
    if isinstance(atom_names, str):
        atom_names = [atom_names]
        squeeze = True
    indices = [RC.atom_order[atom_name] for atom_name in atom_names]
    dim = dim % atom37.ndim
    index = tuple(slice(None) if dim != i else indices for i in range(atom37.ndim))
    result = atom37[index]  # type: ignore
    if squeeze:
        result = result.squeeze(dim)
    return result
```

- **功能**：根据给定的原子名称，从包含37种原子的张量或数组中提取对应原子的坐标。
- **参数**：
  - `atom37`：形状为 `[L, 37, 3]` 的数组或张量，代表蛋白质中每个氨基酸残基的37种原子的坐标。
  - `atom_names`：单个原子名称字符串或字符串列表，指定需要提取的原子名称。
  - `dim`：指定要索引的维度，默认为倒数第二维（即原子种类维度）。
- **实现步骤**：
  1. 判断 `atom_names` 是否为单个字符串，如果是，则将其转化为列表，并设置 `squeeze` 标志为 `True`，以便后续去除单一维度。
  2. 使用 `RC.atom_order` 获取每个原子名称对应的索引。
  3. 处理 `dim`，确保其在有效范围内（通过取模运算）。
  4. 构建索引元组，对于非目标维度使用 `slice(None)`，目标维度使用对应的原子索引列表。
  5. 使用构建好的索引元组从 `atom37` 中提取所需的原子坐标。
  6. 如果 `squeeze` 为 `True`，则在目标维度上去除单一维度。
- **返回值**：提取后的原子坐标，类型与输入的 `atom37` 相同（即 `np.ndarray` 或 `Tensor`）。

### 3. `get_protein_normalization_frame`

```python
def get_protein_normalization_frame(coords: Tensor) -> Affine3D:
    """Given a set of coordinates for a protein, compute a single frame that can be used to normalize the coordinates.
    Specifically, we compute the average position of the N, CA, and C atoms use those 3 points to construct a frame
    using the Gram-Schmidt algorithm. The average CA position is used as the origin of the frame.

    Args:
        coords (torch.FloatTensor): [L, 37, 3] tensor of coordinates

    Returns:
        Affine3D: tensor of Affine3D frame
    """
    bb_coords = index_by_atom_name(coords, ["N", "CA", "C"], dim=-2)
    coord_mask = torch.all(torch.all(torch.isfinite(bb_coords), dim=-1), dim=-1)

    average_position_per_n_ca_c = bb_coords.masked_fill(
        ~coord_mask[..., None, None], 0
    ).sum(-3) / (coord_mask.sum(-1)[..., None, None] + 1e-8)
    frame = atom3_to_backbone_frames(average_position_per_n_ca_c.float())

    return frame
```

- **功能**：根据蛋白质的坐标计算一个标准化的坐标系（Affine3D），用于将蛋白质坐标标准化到该坐标系下。
- **参数**：
  - `coords`：形状为 `[L, 37, 3]` 的张量，代表蛋白质的所有原子的坐标。
- **实现步骤**：
  1. 使用 `index_by_atom_name` 提取N、CA、C三个骨架原子的坐标，得到 `bb_coords`，形状为 `[L, 3, 3]`。
  2. 生成一个掩码 `coord_mask`，用于标识哪些位置的N、CA、C原子的坐标是有效的（即非无穷大且有限）。
  3. 计算每个蛋白质残基的N、CA、C原子的平均坐标 `average_position_per_n_ca_c`：
     - 首先，将无效坐标的位置用0填充。
     - 然后，对有效位置的N、CA、C坐标求和，并除以有效点的数量（加上一个极小值以避免除零错误）。
  4. 使用 `atom3_to_backbone_frames` 根据平均的N、CA、C坐标构建标准化坐标系 `frame`。
- **返回值**：一个 `Affine3D` 对象，代表构建好的标准化坐标系。

### 4. `apply_frame_to_coords`

```python
def apply_frame_to_coords(coords: Tensor, frame: Affine3D) -> Tensor:
    """Given a set of coordinates and a single frame, apply the frame to the coordinates.

    Args:
        coords (torch.FloatTensor): [L, 37, 3] tensor of coordinates
        frame (Affine3D): Affine3D frame

    Returns:
        torch.FloatTensor: [L, 37, 3] tensor of transformed coordinates
    """
    coords_trans_rot = frame[..., None, None].invert().apply(coords)

    # only transform coordinates with frame that have a valid rotation
    valid_frame = frame.trans.norm(dim=-1) > 0

    is_inf = torch.isinf(coords)
    coords = coords_trans_rot.where(valid_frame[..., None, None, None], coords)
    coords.masked_fill_(is_inf, torch.inf)

    return coords
```

- **功能**：将标准化的坐标系应用到蛋白质的所有坐标上，实现坐标的平移和旋转。
- **参数**：
  - `coords`：形状为 `[L, 37, 3]` 的张量，代表蛋白质的所有原子的坐标。
  - `frame`：一个 `Affine3D` 对象，代表要应用的标准化坐标系。
- **实现步骤**：
  1. 对 `frame` 进行逆变换（`invert()`），然后应用到 `coords` 上，得到旋转和平移后的坐标 `coords_trans_rot`。
  2. 判断 `frame` 是否有效（即其平移向量的范数是否大于0），得到 `valid_frame` 掩码。
  3. 标记出原始坐标中为无穷大的位置 `is_inf`。
  4. 对于有效的帧，将旋转后的坐标赋值给 `coords`；对于无效的帧，保持原始坐标不变。
  5. 将原始坐标中为无穷大的位置重新赋值为无穷大，确保这些特殊值不被修改。
- **返回值**：形状为 `[L, 37, 3]` 的张量，代表经过标准化坐标系变换后的蛋白质坐标。

### 5. `normalize_coordinates`

```python
def normalize_coordinates(coords: Tensor) -> Tensor:
    return apply_frame_to_coords(coords, get_protein_normalization_frame(coords))
```

- **功能**：对蛋白质的坐标进行标准化处理，整合前面定义的函数，实现完整的坐标标准化流程。
- **参数**：
  - `coords`：形状为 `[L, 37, 3]` 的张量，代表蛋白质的所有原子的坐标。
- **实现步骤**：
  1. 调用 `get_protein_normalization_frame(coords)` 计算标准化的坐标系 `frame`。
  2. 调用 `apply_frame_to_coords(coords, frame)` 将该坐标系应用到所有原子坐标上，得到标准化后的坐标。
- **返回值**：形状为 `[L, 37, 3]` 的张量，代表标准化后的蛋白质坐标。

## 总体功能总结

整个 `normalize_coordinates.py` 模块的主要功能是对蛋白质的三维坐标进行标准化处理。具体步骤如下：

1. **提取骨架原子坐标**：从所有原子坐标中提取出N、CA、C三个骨架原子的坐标。
2. **计算平均坐标**：计算所有残基的N、CA、C原子的平均坐标，用于定义标准化坐标系的原点和方向。
3. **构建标准化坐标系**：使用格拉汉-施密特正交化算法，根据平均N、CA、C坐标构建一个3D仿射坐标系。
4. **应用坐标系变换**：将构建好的标准化坐标系应用到所有原子坐标上，实现蛋白质整体的平移和旋转，使其在统一的参考系下表示。

这种标准化处理在蛋白质结构分析、比较和机器学习等领域非常有用，因为它消除了由于蛋白质在空间中的不同位置和朝向带来的差异，使得不同蛋白质或同一蛋白质的不同构象可以在统一的框架下进行比较和分析。

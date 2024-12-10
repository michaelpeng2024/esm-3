## protein_structure-代码分析
这段代码 `protein_structure.py` 主要用于处理和分析蛋白质结构数据，利用了 PyTorch 和 NumPy 等库进行高效的张量计算。以下是对代码各部分功能的详细中文分析：

### 1. 导入模块和类型定义

```python
from __future__ import annotations

from typing import Tuple, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast  # type: ignore

from esm.utils import residue_constants
from esm.utils.misc import unbinpack
from esm.utils.structure.affine3d import Affine3D

ArrayOrTensor = TypeVar("ArrayOrTensor", np.ndarray, Tensor)
```

- **`__future__` 导入**：确保代码在不同 Python 版本中具有一致的行为，尤其是注解相关的功能。
- **类型提示**：通过 `TypeVar` 定义 `ArrayOrTensor`，表示可以是 NumPy 的 `ndarray` 或 PyTorch 的 `Tensor`。
- **导入库**：
  - **NumPy 和 PyTorch**：用于数值计算和张量操作。
  - **ESM 工具模块**：包括残基常量、杂项函数 `unbinpack` 以及 `Affine3D` 类，用于处理蛋白质结构相关的数据和变换。

### 2. 根据原子名称索引

```python
def index_by_atom_name(
    atom37: ArrayOrTensor, atom_names: str | list[str], dim: int = -2
) -> ArrayOrTensor:
    squeeze = False
    if isinstance(atom_names, str):
        atom_names = [atom_names]
        squeeze = True
    indices = [residue_constants.atom_order[atom_name] for atom_name in atom_names]
    dim = dim % atom37.ndim
    index = tuple(slice(None) if dim != i else indices for i in range(atom37.ndim))
    result = atom37[index]  # type: ignore
    if squeeze:
        result = result.squeeze(dim)
    return result
```

- **功能**：根据给定的原子名称（如 "N", "CA", "C"）从 `atom37` 数据中提取对应的原子坐标。
- **参数**：
  - `atom37`：包含所有原子坐标的数组或张量，通常形状为 `(B, N, 37, 3)`，其中 `B` 是批量大小，`N` 是残基数量，37 是所有可能的原子数，3 是三维坐标。
  - `atom_names`：单个原子名称字符串或原子名称列表。
  - `dim`：要进行索引的维度，默认是倒数第二维（通常对应于原子类型维）。
- **处理步骤**：
  1. 如果 `atom_names` 是字符串，则转换为列表，并在最后移除多余的维度。
  2. 根据 `residue_constants.atom_order` 获取每个原子名称对应的索引。
  3. 构建索引元组，选择目标维度上的指定原子。
  4. 提取对应的原子坐标，如果原来是单个原子名称，则去除多余的维度。

### 3. 从 atom37 推断 C-beta 原子位置

```python
def infer_cbeta_from_atom37(
    atom37: ArrayOrTensor, L: float = 1.522, A: float = 1.927, D: float = -2.143
):
    """
    Inspired by a util in trDesign:
    https://github.com/gjoni/trDesign/blob/f2d5930b472e77bfacc2f437b3966e7a708a8d37/02-GD/utils.py#L92

    input:  atom37, (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """
    N = index_by_atom_name(atom37, "N", dim=-2)
    CA = index_by_atom_name(atom37, "CA", dim=-2)
    C = index_by_atom_name(atom37, "C", dim=-2)

    if isinstance(atom37, np.ndarray):

        def normalize(x: ArrayOrTensor):
            return x / np.linalg.norm(x, axis=-1, keepdims=True)

        cross = np.cross
    else:
        normalize = F.normalize  # type: ignore
        cross = torch.cross

    with np.errstate(invalid="ignore"):  # inf - inf = nan is ok here
        vec_nca = N - CA
        vec_nc = N - C
    nca = normalize(vec_nca)
    n = normalize(cross(vec_nc, nca))  # type: ignore
    m = [nca, cross(n, nca), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return CA + sum([m * d for m, d in zip(m, d)])
```

- **功能**：基于已知的 `N`, `CA`, `C` 三个原子的坐标，推断出 C-beta (`CB`) 原子的坐标。
- **参数**：
  - `atom37`：包含所有原子坐标的数据结构。
  - `L`, `A`, `D`：C-beta 原子的构建参数，分别表示键长（Length）、键角（Angle）和二面角（Dihedral angle）。
- **处理步骤**：
  1. 使用 `index_by_atom_name` 提取 `N`, `CA`, `C` 原子的坐标。
  2. 根据 `atom37` 的类型（NumPy 数组或 PyTorch 张量）选择相应的归一化和叉乘函数。
  3. 计算向量 `vec_nca`（N - CA）和 `vec_nc`（N - C）。
  4. 归一化这些向量，计算正交基向量 `n`。
  5. 使用这些基向量和给定的参数 `L`, `A`, `D` 计算 C-beta 的坐标。

### 4. 计算对齐张量

```python
@torch.no_grad()
@autocast("cuda", enabled=False)
def compute_alignment_tensors(
    mobile: torch.Tensor,
    target: torch.Tensor,
    atom_exists_mask: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
):
    """
    Align two batches of structures with support for masking invalid atoms using PyTorch.

    Args:
    - mobile (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, N, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, N, 3)
    - atom_exists_mask (torch.Tensor, optional): Mask for Whether an atom exists of shape (B, N)
    - sequence_id (torch.Tensor, optional): Sequence id tensor for binpacking.

    Returns:
    - centered_mobile (torch.Tensor): Batch of coordinates of structure centered mobile (B, N, 3)
    - centroid_mobile (torch.Tensor): Batch of coordinates of mobile centroid (B, 3)
    - centered_target (torch.Tensor): Batch of coordinates of structure centered target (B, N, 3)
    - centroid_target (torch.Tensor): Batch of coordinates of target centroid (B, 3)
    - rotation_matrix (torch.Tensor): Batch of rotation matrices (B, 3, 3)
    - num_valid_atoms (torch.Tensor): Batch of number of valid atoms for alignment (B,)
    """

    # Ensure both batches have the same number of structures, atoms, and dimensions
    if sequence_id is not None:
        mobile = unbinpack(mobile, sequence_id, pad_value=torch.nan)
        target = unbinpack(target, sequence_id, pad_value=torch.nan)
        if atom_exists_mask is not None:
            atom_exists_mask = unbinpack(atom_exists_mask, sequence_id, pad_value=0)
        else:
            atom_exists_mask = torch.isfinite(target).all(-1)

    assert mobile.shape == target.shape, "Batch structure shapes do not match!"

    # Number of structures in the batch
    batch_size = mobile.shape[0]

    # if [B, Nres, Natom, 3], resize
    if mobile.dim() == 4:
        mobile = mobile.view(batch_size, -1, 3)
    if target.dim() == 4:
        target = target.view(batch_size, -1, 3)
    if atom_exists_mask is not None and atom_exists_mask.dim() == 3:
        atom_exists_mask = atom_exists_mask.view(batch_size, -1)

    # Number of atoms
    num_atoms = mobile.shape[1]

    # Apply masks if provided
    if atom_exists_mask is not None:
        mobile = mobile.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
        target = target.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
    else:
        atom_exists_mask = torch.ones(
            batch_size, num_atoms, dtype=torch.bool, device=mobile.device
        )

    num_valid_atoms = atom_exists_mask.sum(dim=-1, keepdim=True)
    # Compute centroids for each batch
    centroid_mobile = mobile.sum(dim=-2, keepdim=True) / num_valid_atoms.unsqueeze(-1)
    centroid_target = target.sum(dim=-2, keepdim=True) / num_valid_atoms.unsqueeze(-1)

    # Handle potential division by zero if all atoms are invalid in a structure
    centroid_mobile[num_valid_atoms == 0] = 0
    centroid_target[num_valid_atoms == 0] = 0

    # Center structures by subtracting centroids
    centered_mobile = mobile - centroid_mobile
    centered_target = target - centroid_target

    centered_mobile = centered_mobile.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
    centered_target = centered_target.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)

    # Compute covariance matrix for each batch
    covariance_matrix = torch.matmul(centered_mobile.transpose(1, 2), centered_target)

    # Singular Value Decomposition for each batch
    u, _, v = torch.svd(covariance_matrix)

    # Calculate rotation matrices for each batch
    rotation_matrix = torch.matmul(u, v.transpose(1, 2))

    return (
        centered_mobile,
        centroid_mobile,
        centered_target,
        centroid_target,
        rotation_matrix,
        num_valid_atoms,
    )
```

- **功能**：对两批蛋白质结构进行对齐，支持掩码无效原子的情况，并返回对齐相关的张量。
- **装饰器**：
  - `@torch.no_grad()`：在计算过程中不追踪梯度，节省内存和计算资源。
  - `@autocast("cuda", enabled=False)`：禁用自动混合精度，确保计算精度。
- **参数**：
  - `mobile`：待对齐的蛋白质结构坐标，形状为 `(B, N, 3)`，其中 `B` 是批量大小，`N` 是原子数。
  - `target`：参考蛋白质结构坐标，形状同上。
  - `atom_exists_mask`：可选的掩码，表示哪些原子存在，形状为 `(B, N)`。
  - `sequence_id`：可选的序列 ID，用于批处理对齐（binpacking）。
- **处理步骤**：
  1. 如果提供了 `sequence_id`，则使用 `unbinpack` 函数将批量数据展开，填充值分别为 `torch.nan` 和 `0`。
  2. 确保 `mobile` 和 `target` 的形状相同。
  3. 如果输入张量有四维（例如包含残基和原子类型），则将其重塑为三维。
  4. 应用 `atom_exists_mask`，将不存在的原子坐标设为 `0`。
  5. 计算每个结构的有效原子数，并计算 `mobile` 和 `target` 的质心。
  6. 通过减去质心将结构居中。
  7. 计算协方差矩阵，并进行奇异值分解（SVD）以获取旋转矩阵。
  8. 返回居中后的坐标、质心、旋转矩阵以及有效原子数。

### 5. 计算未对齐的 RMSD

```python
@torch.no_grad()
@autocast("cuda", enabled=False)
def compute_rmsd_no_alignment(
    aligned: torch.Tensor,
    target: torch.Tensor,
    num_valid_atoms: torch.Tensor,
    reduction: str = "batch",
) -> torch.Tensor:
    """
    Compute RMSD between two batches of structures without alignment.

    Args:
    - aligned (torch.Tensor): Batch of aligned coordinates in shape (B, N, 3)
    - target (torch.Tensor): Batch of target coordinates in shape (B, N, 3)
    - num_valid_atoms (torch.Tensor): Batch of number of valid atoms for alignment (B,)
    - reduction (str): One of "batch", "per_sample", "per_residue".

    Returns:

    If reduction == "batch":
        (torch.Tensor): 0-dim, Average Root Mean Square Deviation between the structures for each batch
    If reduction == "per_sample":
        (torch.Tensor): (B,)-dim, Root Mean Square Deviation between the structures for each batch
    If reduction == "per_residue":
        (torch.Tensor): (B, N)-dim, Root Mean Square Deviation between the structures for each residue in the batch
    """
    if reduction not in ("per_residue", "per_sample", "batch"):
        raise ValueError("Unrecognized reduction: '{reduction}'")
    # Compute RMSD for each batch
    diff = aligned - target
    if reduction == "per_residue":
        mean_squared_error = diff.square().view(diff.size(0), -1, 9).mean(dim=-1)
    else:
        mean_squared_error = diff.square().sum(dim=(1, 2)) / (
            num_valid_atoms.squeeze(-1) * 3
        )

    rmsd = torch.sqrt(mean_squared_error)
    if reduction in ("per_sample", "per_residue"):
        return rmsd
    elif reduction == "batch":
        avg_rmsd = rmsd.masked_fill(num_valid_atoms.squeeze(-1) == 0, 0).sum() / (
            (num_valid_atoms > 0).sum() + 1e-8
        )
        return avg_rmsd
    else:
        raise ValueError(reduction)
```

- **功能**：计算两批蛋白质结构之间的均方根偏差（RMSD），不进行对齐。
- **参数**：
  - `aligned`：对齐后的结构坐标，形状为 `(B, N, 3)`。
  - `target`：参考结构坐标，形状同上。
  - `num_valid_atoms`：每个结构中有效原子的数量，形状为 `(B,)`。
  - `reduction`：指定结果的聚合方式，可以是 `"batch"`, `"per_sample"`, 或 `"per_residue"`。
- **处理步骤**：
  1. 计算坐标差异 `diff = aligned - target`。
  2. 根据 `reduction` 的不同，计算均方误差：
     - `"per_residue"`：按残基计算，每个残基有 3 个坐标，平方后平均。
     - 其他情况：总和除以有效原子数乘以 3。
  3. 计算 RMSD，即均方误差的平方根。
  4. 根据 `reduction` 的类型，返回相应的聚合结果：
     - `"per_sample"` 和 `"per_residue"`：直接返回计算的 RMSD。
     - `"batch"`：对所有样本的 RMSD 取平均，忽略有效原子数为 0 的情况。

### 6. 计算仿射变换和 RMSD

```python
@torch.no_grad()
@autocast("cuda", enabled=False)
def compute_affine_and_rmsd(
    mobile: torch.Tensor,
    target: torch.Tensor,
    atom_exists_mask: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
) -> Tuple[Affine3D, torch.Tensor]:
    """
    Compute RMSD between two batches of structures with support for masking invalid atoms using PyTorch.

    Args:
    - mobile (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, N, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, N, 3)
    - atom_exists_mask (torch.Tensor, optional): Mask for Whether an atom exists of shape (B, N)
    - sequence_id (torch.Tensor, optional): Sequence id tensor for binpacking.

    Returns:
    - affine (Affine3D): Transformation between mobile and target structure
    - avg_rmsd (torch.Tensor): Average Root Mean Square Deviation between the structures for each batch
    """

    (
        centered_mobile,
        centroid_mobile,
        centered_target,
        centroid_target,
        rotation_matrix,
        num_valid_atoms,
    ) = compute_alignment_tensors(
        mobile=mobile,
        target=target,
        atom_exists_mask=atom_exists_mask,
        sequence_id=sequence_id,
    )

    # Apply rotation to mobile centroid
    translation = torch.matmul(-centroid_mobile, rotation_matrix) + centroid_target
    affine = Affine3D.from_tensor_pair(
        translation, rotation_matrix.unsqueeze(dim=-3).transpose(-2, -1)
    )

    # Apply transformation to centered structure to compute rmsd
    rotated_mobile = torch.matmul(centered_mobile, rotation_matrix)
    avg_rmsd = compute_rmsd_no_alignment(
        rotated_mobile, centered_target, num_valid_atoms, reduction="batch"
    )

    return affine, avg_rmsd
```

- **功能**：计算两批蛋白质结构之间的仿射变换（包括旋转和平移）以及平均 RMSD。
- **参数**：
  - `mobile`：待对齐的蛋白质结构坐标，形状为 `(B, N, 3)`。
  - `target`：参考蛋白质结构坐标，形状同上。
  - `atom_exists_mask`：可选的掩码，表示哪些原子存在，形状为 `(B, N)`。
  - `sequence_id`：可选的序列 ID，用于批处理对齐。
- **处理步骤**：
  1. 调用 `compute_alignment_tensors` 计算居中后的坐标、质心、旋转矩阵及有效原子数。
  2. 计算平移向量，使得 `mobile` 的质心对齐到 `target` 的质心。
  3. 使用 `Affine3D.from_tensor_pair` 创建仿射变换对象，包含旋转和平移。
  4. 将旋转矩阵应用到居中的 `mobile` 坐标，得到旋转后的坐标。
  5. 调用 `compute_rmsd_no_alignment` 计算旋转后的 `mobile` 与 `target` 之间的平均 RMSD。
  6. 返回仿射变换对象和平均 RMSD。

### 7. 计算未对齐的 GDT_TS

```python
def compute_gdt_ts_no_alignment(
    aligned: torch.Tensor,
    target: torch.Tensor,
    atom_exists_mask: torch.Tensor,
    reduction: str = "batch",
) -> torch.Tensor:
    """
    Compute GDT_TS between two batches of structures without alignment.

    Args:
    - aligned (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, N, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, N, 3)
    - atom_exists_mask (torch.Tensor): Mask for Whether an atom exists of shape (B, N). noo
    - reduction (str): One of "batch", "per_sample".

    Returns:
    If reduction == "batch":
        (torch.Tensor): 0-dim, GDT_TS between the structures for each batch
    If reduction == "per_sample":
        (torch.Tensor): (B,)-dim, GDT_TS between the structures for each sample in the batch
    """
    if reduction not in ("per_sample", "batch"):
        raise ValueError("Unrecognized reduction: '{reduction}'")

    if atom_exists_mask is None:
        atom_exists_mask = torch.isfinite(target).all(dim=-1)

    deviation = torch.linalg.vector_norm(aligned - target, dim=-1)
    num_valid_atoms = atom_exists_mask.sum(dim=-1)

    # Compute GDT_TS
    score = (
        ((deviation < 1) * atom_exists_mask).sum(dim=-1) / num_valid_atoms
        + ((deviation < 2) * atom_exists_mask).sum(dim=-1) / num_valid_atoms
        + ((deviation < 4) * atom_exists_mask).sum(dim=-1) / num_valid_atoms
        + ((deviation < 8) * atom_exists_mask).sum(dim=-1) / num_valid_atoms
    ) * 0.25

    if reduction == "batch":
        return score.mean()
    elif reduction == "per_sample":
        return score
    else:
        raise ValueError("Unrecognized reduction: '{reduction}'")
```

- **功能**：计算两批蛋白质结构之间的全局距离测试评分（GDT_TS），不进行对齐。
- **参数**：
  - `aligned`：待比较的蛋白质结构坐标，形状为 `(B, N, 3)`。
  - `target`：参考蛋白质结构坐标，形状同上。
  - `atom_exists_mask`：表示哪些原子存在的掩码，形状为 `(B, N)`。
  - `reduction`：指定结果的聚合方式，可以是 `"batch"` 或 `"per_sample"`。
- **处理步骤**：
  1. 检查 `reduction` 参数是否有效。
  2. 如果未提供 `atom_exists_mask`，则根据 `target` 的有限性自动生成掩码。
  3. 计算每个原子的位置偏差 `deviation`，即 `aligned` 和 `target` 之间的欧氏距离。
  4. 计算每个结构中有效原子的数量 `num_valid_atoms`。
  5. 计算 GDT_TS 分数：
     - GDT_TS 通常通过四个距离阈值（1 Å, 2 Å, 4 Å, 8 Å）下的匹配比例来计算。
     - 对每个阈值，计算偏差小于阈值的原子比例，并取平均。
  6. 根据 `reduction` 参数返回相应的结果：
     - `"batch"`：对所有样本的 GDT_TS 取平均。
     - `"per_sample"`：返回每个样本的 GDT_TS 分数。

### 总结

该模块主要提供了一系列函数，用于处理和分析蛋白质结构数据，具体包括：

1. **原子索引**：根据原子名称提取对应的坐标。
2. **C-beta 原子推断**：基于已知的 `N`, `CA`, `C` 原子坐标，推断 `CB` 原子的位置。
3. **结构对齐**：计算两批蛋白质结构之间的仿射变换（旋转和平移），并对齐结构。
4. **RMSD 计算**：在对齐后的基础上，计算结构之间的均方根偏差，以评估对齐质量。
5. **GDT_TS 计算**：在不对齐的情况下，计算结构之间的全局距离测试评分，用于评估结构相似性。

这些功能在蛋白质结构预测、对比和评估中具有广泛应用，特别是在结构生物学和计算生物学领域。

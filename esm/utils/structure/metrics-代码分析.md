## metrics-代码分析
以上代码实现了用于评估蛋白质结构预测准确性的两种主要指标：LDDT（Local Distance Difference Test）和GDT_TS（Global Distance Test Total Score）。这些指标广泛应用于蛋白质结构预测领域，用以量化预测结构与真实结构之间的相似性。以下是对代码各部分功能的详细分析：

## 1. 引入必要的库和模块

```python
import torch
from einops import rearrange

from esm.utils import residue_constants
from esm.utils.misc import unbinpack
from esm.utils.structure.protein_structure import (
    compute_alignment_tensors,
    compute_gdt_ts_no_alignment,
)
```

- **torch**: 用于张量运算，支持GPU加速。
- **einops.rearrange**: 用于张量的重排，简化维度变换操作。
- **esm.utils.residue_constants**: 包含与氨基酸残基相关的常量，如原子序号等。
- **esm.utils.misc.unbinpack**: 用于处理序列ID的辅助函数。
- **esm.utils.structure.protein_structure**: 提供计算对齐张量和GDT_TS的辅助函数。

## 2. 计算LDDT指标的函数 `compute_lddt`

```python
def compute_lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
    sequence_id: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

### 功能概述

`compute_lddt` 函数用于计算蛋白质的LDDT评分。LDDT是一种局部距离差异测试指标，用于评估预测结构与真实结构在局部范围内的距离一致性。

### 输入参数

- **all_atom_pred_pos** (`torch.Tensor`): 预测的所有原子的位置，形状为 `[Nstates x] B x (L * Natoms) x 3`。`Nstates` 表示不同的预测状态（如模型不同层的输出），`B` 是批量大小，`L` 是残基长度，`Natoms` 是每个残基的原子数。
- **all_atom_positions** (`torch.Tensor`): 真实的所有原子的位置，形状为 `[B x (L * Natoms) x 3]`。
- **all_atom_mask** (`torch.Tensor`): 原子存在的掩码，形状为 `[B x (L * Natoms)]`，用于指示哪些原子在计算中被考虑。
- **cutoff** (`float`): 评分计算中考虑的最大距离阈值，默认为15.0 Å。
- **eps** (`float`): 防止除零的微小常数，默认为1e-10。
- **per_residue** (`bool`): 是否返回每个残基的LDDT评分，默认为`True`。
- **sequence_id** (`torch.Tensor | None`): 序列ID张量，用于序列打包，仅支持LDDT_CA计算。

### 计算步骤

1. **计算真实结构和预测结构的距离矩阵**：
    - `dmat_true`: 计算真实结构中所有原子对之间的欧氏距离矩阵。
    - `dmat_pred`: 计算预测结构中所有原子对之间的欧氏距离矩阵。

2. **确定需要评分的原子对**：
    - 根据距离阈值`cutoff`、原子存在的掩码`all_atom_mask`以及去除自对角线（自身与自身的距离）来确定哪些原子对需要进行评分。

3. **处理序列ID（如果提供）**：
    - 如果提供了`sequence_id`，则仅计算序列ID相同的原子对的距离差异。

4. **计算距离差异的绝对值**：
    - `dist_l1`: 计算预测距离与真实距离之间的绝对差值。

5. **评分规则**：
    - 根据距离差异`dist_l1`的大小，将其分为四个区间（<0.5 Å, <1.0 Å, <2.0 Å, <4.0 Å），每个区间分别赋予不同的评分权重（0.25）。

6. **归一化和最终评分**：
    - 根据需要评分的原子对数量进行归一化，得到最终的LDDT评分。根据`per_residue`参数，返回每个残基的评分或整个蛋白质的平均评分。

### 返回值

- **LDDT评分**：
    - 如果`per_residue`为`True`，返回形状为`[(Nstates x) B x (L * Natoms)]`的张量，每个残基对应一个LDDT评分。
    - 如果`per_residue`为`False`，返回形状为`[(Nstates x) B]`的张量，每个蛋白质对应一个整体LDDT评分。

## 3. 计算Cα原子的LDDT评分函数 `compute_lddt_ca`

```python
def compute_lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
    sequence_id: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

### 功能概述

`compute_lddt_ca` 函数专门用于计算仅基于Cα（α碳原子）的LDDT评分。这在某些情况下（如仅关注主链的结构准确性）非常有用。

### 实现步骤

1. **提取Cα原子的位置和掩码**：
    - 使用`residue_constants.atom_order["CA"]`获取Cα原子的索引。
    - 如果`all_atom_pred_pos`的维度不是3维，则提取Cα原子的预测位置。
    - 同样提取真实结构中Cα原子的位置和对应的掩码。

2. **调用`compute_lddt`**：
    - 使用提取的Cα原子的位置和掩码，调用前述的`compute_lddt`函数计算LDDT评分。

### 返回值

- 返回与`compute_lddt`相同格式的LDDT评分，但仅基于Cα原子。

## 4. 计算GDT_TS指标的函数 `compute_gdt_ts`

```python
def compute_gdt_ts(
    mobile: torch.Tensor,
    target: torch.Tensor,
    atom_exists_mask: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
    reduction: str = "per_sample",
):
    ...
```

### 功能概述

`compute_gdt_ts` 函数用于计算GDT_TS（Global Distance Test Total Score）指标。GDT_TS是一种全局距离测试总评分，用于评估预测结构与真实结构在整体上的相似性。

### 输入参数

- **mobile** (`torch.Tensor`): 需要进行叠加的预测结构坐标，形状为 `(B, N, 3)`，其中`B`是批量大小，`N`是原子数量。
- **target** (`torch.Tensor`): 真实结构的坐标，形状为 `(B, N, 3)`。
- **atom_exists_mask** (`torch.Tensor | None`): 原子存在的掩码，形状为 `(B, N)`。如果为`None`，则自动根据`target`的有限值进行判断。
- **sequence_id** (`torch.Tensor | None`): 序列ID张量，用于序列打包。
- **reduction** (`str`): 指定结果的聚合方式，可以是`"batch"`（整体评分）、`"per_sample"`（每个样本评分）或`"per_residue"`（每个残基评分），默认为`"per_sample"`。

### 计算步骤

1. **处理原子存在的掩码**：
    - 如果`atom_exists_mask`为`None`，则根据`target`中所有坐标是否为有限值来生成掩码。

2. **计算对齐张量**：
    - 调用`compute_alignment_tensors`，根据提供的`mobile`和`target`坐标，以及掩码和序列ID，计算对齐所需的张量，包括中心化的移动和目标结构、旋转矩阵等。

3. **应用旋转矩阵**：
    - 使用旋转矩阵将中心化后的移动结构旋转到与目标结构对齐。

4. **处理序列ID（如果提供）**：
    - 如果提供了`sequence_id`，则对`atom_exists_mask`进行解包（unbinpack），确保掩码与打包后的结构对齐。

5. **计算GDT_TS**：
    - 调用`compute_gdt_ts_no_alignment`，传入旋转后的移动结构、中心化的目标结构、掩码以及指定的`reduction`方式，计算GDT_TS评分。

### 返回值

- **GDT_TS评分**：
    - 根据`reduction`参数的不同，返回不同形状的张量：
        - `"batch"`: 返回0维张量，表示整个批次的GDT_TS评分。
        - `"per_sample"`: 返回形状为`(B,)`的张量，表示每个样本的GDT_TS评分。
        - `"per_residue"`: 返回每个残基的评分（具体实现需查看`compute_gdt_ts_no_alignment`的返回值）。

## 总结

上述代码提供了两个主要的结构评估指标计算函数：

1. **LDDT（Local Distance Difference Test）**：
    - 评估预测结构在局部范围内与真实结构的距离一致性。
    - 提供了全原子LDDT和仅基于Cα原子的LDDT两种计算方式。

2. **GDT_TS（Global Distance Test Total Score）**：
    - 评估预测结构与真实结构在全局范围内的相似性，通过对齐和距离阈值的方式计算得分。

这些函数利用PyTorch的高效张量运算，能够处理批量数据，并支持多种灵活的评分方式（如按样本、按残基或整体评分），适用于大规模蛋白质结构预测模型的性能评估。

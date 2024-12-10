## predicted_aligned_error-代码分析
这段代码 `predicted_aligned_error.py` 实现了与蛋白质结构预测相关的功能，主要涉及预测对齐误差（Predicted Aligned Error, PAE）和TM分数（Template Modeling score, TM-score）的计算以及相应的损失函数。这些指标在评估和优化蛋白质结构预测模型（如AlphaFold）中起着关键作用。以下是对代码中各部分功能的详细分析：

## 1. 导入模块

```python
import torch
import torch.nn.functional as F
from esm.utils.structure.affine3d import Affine3D
```

- **torch** 和 **torch.nn.functional**：用于张量操作和神经网络功能。
- **Affine3D**：来自 `esm`（Evolutionary Scale Modeling）库，用于处理三维仿射变换，主要涉及蛋白质的几何结构。

## 2. 辅助函数

### 2.1 `masked_mean`

```python
def masked_mean(
    mask: torch.Tensor,
    value: torch.Tensor,
    dim: int | None | tuple[int, ...] = None,
    eps=1e-10,
) -> torch.Tensor:
    """Compute the mean of `value` where only positions where `mask == true` are counted."""
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
```

- **功能**：计算 `value` 张量在 `mask` 为 `True` 的位置上的均值。
- **参数**：
  - `mask`：布尔张量，指示哪些位置需要被计算。
  - `value`：要计算均值的张量。
  - `dim`：指定计算的维度。
  - `eps`：一个很小的常数，防止除零错误。
- **实现细节**：
  - 首先将 `mask` 扩展到与 `value` 相同的形状。
  - 使用 `mask` 对 `value` 进行掩蔽，然后在指定维度上求和，并除以 `mask` 为 `True` 的元素数量，得到均值。

### 2.2 `_pae_bins`

```python
def _pae_bins(
    max_bin: float = 31, num_bins: int = 64, device: torch.device = torch.device("cpu")
):
    bins = torch.linspace(0, max_bin, steps=(num_bins - 1), device=device)
    step = max_bin / (num_bins - 2)
    bin_centers = bins + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers
```

- **功能**：生成PAE的分箱中心值，用于将预测的误差离散化。
- **参数**：
  - `max_bin`：最大分箱值，默认为31。
  - `num_bins`：分箱数量，默认为64。
  - `device`：张量所在设备，默认为CPU。
- **实现细节**：
  - 使用 `torch.linspace` 生成从0到 `max_bin` 的等间距分箱边界。
  - 计算每个分箱的中心，并将最后一个分箱中心延伸一个步长，以确保包含最大值。

### 2.3 `_compute_pae_masks`

```python
def _compute_pae_masks(mask: torch.Tensor):
    square_mask = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).bool()
    return square_mask
```

- **功能**：生成PAE计算所需的对称掩码矩阵。
- **参数**：
  - `mask`：布尔张量，指示哪些氨基酸残基需要被考虑。
- **实现细节**：
  - 通过对 `mask` 进行扩展并进行外积操作，生成一个二维的对称掩码矩阵，表示哪些残基对之间需要计算PAE。

## 3. 核心功能函数

### 3.1 `compute_predicted_aligned_error`

```python
def compute_predicted_aligned_error(
    logits: torch.Tensor,
    aa_mask: torch.Tensor,
    sequence_id: torch.Tensor | None = None,
    max_bin: float = 31,
) -> torch.Tensor:
    bins = _pae_bins(max_bin, logits.shape[-1], logits.device)
    square_mask = _compute_pae_masks(aa_mask)
    min_v = torch.finfo(logits.dtype).min
    probs = logits.masked_fill(~square_mask.unsqueeze(-1), min_v).softmax(dim=-1)

    return (probs * bins).sum(dim=-1)
```

- **功能**：根据模型的logits输出和氨基酸掩码，计算预测的对齐误差（PAE）。
- **参数**：
  - `logits`：模型输出的logits，通常为未归一化的预测分布。
  - `aa_mask`：氨基酸掩码，指示哪些残基是有效的。
  - `sequence_id`：序列ID，可选，用于多序列情况下的区分（本函数中未使用）。
  - `max_bin`：最大分箱值，默认为31。
- **实现细节**：
  - 使用 `_pae_bins` 生成分箱中心值。
  - 生成PAE的对称掩码矩阵 `square_mask`。
  - 将 `logits` 中不在掩码中的位置填充为最小浮点数，以确保这些位置在softmax后概率接近零。
  - 对填充后的 `logits` 进行softmax归一化，得到概率分布 `probs`。
  - 计算预测的PAE值，方法是将概率分布与分箱中心值相乘并在分箱维度上求和，得到每对残基之间的PAE。

### 3.2 `compute_tm`

```python
@torch.no_grad
def compute_tm(logits: torch.Tensor, aa_mask: torch.Tensor, max_bin: float = 31.0):
    square_mask = _compute_pae_masks(aa_mask)
    seqlens = aa_mask.sum(-1, keepdim=True)
    bins = _pae_bins(max_bin, logits.shape[-1], logits.device)
    d0 = 1.24 * (seqlens.clamp_min(19) - 15) ** (1 / 3) - 1.8
    f_d = 1.0 / (1 + (bins / d0.unsqueeze(-1)) ** 2)

    min_v = torch.finfo(logits.dtype).min
    probs = logits.masked_fill(~square_mask.unsqueeze(-1), min_v).softmax(dim=-1)
    # This is the sum over bins
    ptm = (probs * f_d.unsqueeze(-2)).sum(dim=-1)
    # This is the mean over residues j
    ptm = masked_mean(square_mask, ptm, dim=-1)
    # Then we do a max over residues i
    return ptm.max(dim=-1).values
```

- **功能**：计算TM分数，用于评估预测结构与目标结构的相似性。
- **参数**：
  - `logits`：模型输出的logits，用于计算概率分布。
  - `aa_mask`：氨基酸掩码。
  - `max_bin`：最大分箱值，默认为31。
- **实现细节**：
  - 生成PAE的对称掩码矩阵 `square_mask`。
  - 计算序列长度 `seqlens`，并通过公式计算参数 `d0`，用于调整TM分数的灵敏度。
  - 计算一个衰减函数 `f_d`，用于根据距离调整分数权重。
  - 将 `logits` 中不在掩码中的位置填充为最小浮点数，并进行softmax归一化，得到概率分布 `probs`。
  - 计算加权概率 `ptm`，首先在分箱维度上求和，再在残基j维度上计算掩码均值，最后在残基i维度上取最大值，得到每个序列的TM分数。

### 3.3 `tm_loss`

```python
def tm_loss(
    logits: torch.Tensor,
    pred_affine: torch.Tensor,
    targ_affine: torch.Tensor,
    targ_mask: torch.Tensor,
    tm_mask: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
    max_bin: float = 31,
):
    pred = Affine3D.from_tensor(pred_affine)
    targ = Affine3D.from_tensor(targ_affine)

    def transform(affine: Affine3D):
        pts = affine.trans[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    with torch.no_grad():
        sq_diff = (transform(pred) - transform(targ)).square().sum(dim=-1)

        num_bins = logits.shape[-1]
        sq_bins = torch.linspace(
            0, max_bin, num_bins - 1, device=logits.device
        ).square()
        # Gets the bin id by using a sum.
        true_bins = (sq_diff[..., None] > sq_bins).sum(dim=-1).long()

    errors = F.cross_entropy(logits.movedim(3, 1), true_bins, reduction="none")
    square_mask = _compute_pae_masks(targ_mask)
    loss = masked_mean(square_mask, errors, dim=(-1, -2))

    if tm_mask is not None:
        loss = masked_mean(tm_mask, loss, dim=None)
    else:
        loss = loss.mean()

    return loss
```

- **功能**：计算基于预测PAE和真实PAE之间的交叉熵损失，用于优化模型的结构预测能力。
- **参数**：
  - `logits`：模型输出的logits，用于预测PAE分布。
  - `pred_affine`：预测的仿射变换参数（通常表示预测的蛋白质结构）。
  - `targ_affine`：目标（真实）的仿射变换参数。
  - `targ_mask`：目标的氨基酸掩码。
  - `tm_mask`：可选的TM分数掩码，用于进一步过滤损失计算的区域。
  - `sequence_id`：序列ID，可选（本函数中未使用）。
  - `max_bin`：最大分箱值，默认为31。
- **实现细节**：
  - 将预测和目标的仿射变换参数转换为 `Affine3D` 对象。
  - 定义一个内部函数 `transform`，用于对仿射变换进行逆变换并应用到点集上。
  - 在无梯度计算的上下文中：
    - 计算预测和目标结构的点集差异的平方和 `sq_diff`。
    - 生成平方后的分箱边界 `sq_bins`。
    - 根据 `sq_diff` 确定每对残基的真实分箱 `true_bins`。
  - 计算交叉熵损失 `errors`，其中 `logits` 的维度需要调整以匹配 `cross_entropy` 的输入要求。
  - 生成PAE的对称掩码矩阵 `square_mask`。
  - 使用 `masked_mean` 计算在掩码区域内的平均损失。
  - 如果提供了 `tm_mask`，进一步对损失进行掩蔽均值计算；否则，对所有损失取平均。
  - 返回最终的损失值。

## 4. 代码整体流程与应用场景

### 4.1 代码流程

1. **输入数据**：
   - `logits`：模型输出的未归一化预测值，通常是PAE的logits。
   - `aa_mask`：氨基酸掩码，指示哪些残基需要被考虑。
   - `pred_affine` 和 `targ_affine`：预测和目标的仿射变换参数，表示蛋白质结构。
   
2. **PAE计算**：
   - 通过 `compute_predicted_aligned_error` 函数，将 `logits` 转换为PAE值，反映残基对之间的预测误差。

3. **TM分数计算**：
   - 使用 `compute_tm` 函数，根据PAE值和序列长度，计算TM分数，评估预测结构与目标结构的相似性。

4. **损失计算**：
   - 通过 `tm_loss` 函数，基于预测和目标的结构差异计算交叉熵损失，用于优化模型参数，提高预测精度。

### 4.2 应用场景

- **蛋白质结构预测**：在诸如AlphaFold等模型中，用于评估和优化预测的蛋白质三维结构的准确性。
- **模型训练**：`tm_loss` 函数作为损失函数，可以在训练过程中指导模型学习更准确的结构预测。
- **结果评估**：通过计算PAE和TM分数，可以量化预测结果与真实结构的匹配程度，作为模型性能的指标。

## 5. 关键点总结

- **掩码机制**：通过 `aa_mask` 和 `square_mask`，代码有效地处理了有效残基对的选择，避免计算无效或噪声数据。
- **分箱策略**：使用 `_pae_bins` 函数将连续的误差值离散化，有助于将回归问题转化为分类问题，便于使用交叉熵损失。
- **仿射变换处理**：利用 `Affine3D` 类对蛋白质结构进行仿射变换，确保结构差异的准确计算。
- **无梯度计算**：在计算 `tm_loss` 的过程中，使用 `torch.no_grad()` 以节省内存和计算资源，因为这部分计算不需要反向传播。

通过上述分析，可以看出这段代码在蛋白质结构预测领域具有重要的应用价值，能够有效地计算预测误差和优化模型性能。

## sampling-代码分析
`sampling.py` 是一个用于处理蛋白质数据采样操作的模块，集成了张量操作、分词器管理和采样策略等多种功能。该模块主要在 ESM（Evolutionary Scale Modeling）框架内使用，旨在生成和注释蛋白质序列及其结构。以下是对该代码的详细分析：

### **1. 导入与依赖**

- **标准库：**
  - `warnings`：用于发出警告信息。
  - `typing.Literal`：用于类型提示，限制参数为特定的字面量值。

- **第三方库：**
  - `attr`：用于声明式的类定义，提升代码的可读性和可维护性。
  - `torch` 和 `torch.nn.functional as F`：PyTorch 库，用于张量操作和神经网络功能。

- **自定义模块（ESM 框架）：**
  - **API 组件：**
    - `ESMProteinTensor`：表示蛋白质张量的数据结构。
    - `SamplingConfig` 和 `SamplingTrackConfig`：用于采样操作的配置类。
  - **分词器组件：**
    - `TokenizerCollectionProtocol` 和 `get_invalid_tokenizer_ids`：用于分词器管理的协议和工具。
    - `InterProQuantizedTokenizer`：专门用于功能注释的分词器类。
  - **常量：**
    - `MAX_RESIDUE_ANNOTATIONS` 和 `SASA_DISCRETIZATION_BOUNDARIES`：定义特定注释的限制和边界的常量。

### **2. 辅助函数**

#### **a. `_non_batched_dims`**

```python
def _non_batched_dims(k: str, v: torch.Tensor):
    ...
```

- **目的**：确定不同蛋白质数据轨道（如序列、结构）的非批处理维度数量。
- **功能**：通过模式匹配返回基于轨道类型的预期维度数。
- **使用场景**：确保在处理过程中张量的形状与预期维度一致。

#### **b. `_tensorize_like`**

```python
def _tensorize_like(value: int | float | torch.Tensor, logits: torch.Tensor):
    ...
```

- **目的**：将标量值转换为与提供的 `logits` 形状和设备兼容的张量。
- **功能**：处理标量值的广播，使其匹配张量的维度，确保张量操作的无缝进行。

#### **c. `get_sampling_mask`**

```python
def get_sampling_mask(
    tokens: torch.Tensor, sampling_track_config: SamplingTrackConfig, mask_idx: int
):
    ...
```

- **目的**：生成一个布尔掩码，指示哪些位置在令牌张量中有资格进行采样。
- **功能**：
  - 屏蔽序列开始（BOS）和序列结束（EOS）令牌。
  - 排除除掩码令牌之外的特殊令牌，基于 `sampling_track_config` 的配置。
  - 如果指定，仅限制在被掩码的令牌位置进行采样。

### **3. 类 `_BatchedESMProteinTensor`**

```python
class _BatchedESMProteinTensor(ESMProteinTensor):
    ...
```

- **目的**：处理批量化的 `ESMProteinTensor`，允许对蛋白质张量进行批处理操作。
- **方法：**
  - **`from_protein_tensor`**：
    - **功能**：将单个 `ESMProteinTensor` 转换为批处理形式，即在第一个维度上增加一个批次维度。
  - **`__len__`**：
    - **功能**：返回批次中每个蛋白质的长度。
  - **`batch_size` 属性**：
    - **功能**：返回当前批次的大小。
  - **`slice`**：
    - **功能**：从批次中提取第 `i` 个蛋白质的子张量，可能限制到特定的序列长度。
  - **`set_slice`**：
    - **功能**：更新批次中第 `i` 个蛋白质的特定子张量。

### **4. 配置相关函数**

#### **a. `get_default_sampling_config`**

```python
def get_default_sampling_config(
    tokenizers: TokenizerCollectionProtocol,
) -> SamplingConfig:
    ...
```

- **目的**：生成默认的采样配置，基于提供的分词器集合。
- **功能**：
  - 遍历 `SamplingConfig` 的所有轨道。
  - 为每个轨道设置 `SamplingTrackConfig`，包括无效的分词器 ID、温度、top-p 等参数。
  - 对某些轨道（如 `secondary_structure`、`sasa`、`function`），不限制仅采样被掩码的令牌。

#### **b. `validate_sampling_config`**

```python
def validate_sampling_config(
    sampling_config: SamplingConfig, on_invalid: Literal["raise", "warn"] = "warn"
):
    ...
```

- **目的**：验证采样配置的有效性，确保参数设置合理。
- **功能**：
  - 检查所有采样轨道的 `topk_logprobs` 是否不超过 `MAX_TOP_K`。
  - 根据 `on_invalid` 参数决定是抛出异常还是发出警告。

### **5. 采样函数**

#### **a. `sample_logits`**

```python
def sample_logits(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    valid_ids: list[int] = [],
    top_p: float | torch.Tensor = 1.0,
    mask_logits_of_invalid_ids: bool = True,
):
    ...
```

- **目的**：从给定的 logits 中进行采样，生成下一个令牌的 ID。
- **功能**：
  - 如果 `valid_ids` 为空，抛出错误，因为无法从中采样。
  - 如果 `top_p` 小于 1.0，调用 `top_p_logits` 进行 top-p 过滤。
  - 根据 `temperature` 调整 logits 的软性概率分布。
  - 如果 `mask_logits_of_invalid_ids` 为真，屏蔽无效的 ID（将其 logits 设置为负无穷）。
  - 如果温度为 0，选择 logits 最高的 ID。
  - 否则，根据调整后的概率分布进行多项式采样。

#### **b. `sample_function_logits`**

```python
def sample_function_logits(
    logits: torch.Tensor,
    tokenizer: InterProQuantizedTokenizer,
    top_p: float | torch.Tensor = 1.0,
    temperature: float | torch.Tensor = 1.0,
    p_none_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

- **目的**：针对功能注释的 logits 进行采样，生成功能 ID。
- **功能**：
  - 验证 logits 的形状与分词器的深度一致。
  - 应用 top-p 过滤（如果 `top_p` < 1.0）。
  - 计算 log 概率，并处理 `<none>` 类别的概率。
  - 根据 `p_none_threshold` 决定是否选择 `<none>`。
  - 返回采样的 ID 和对应的 log 概率。

#### **c. `sample_residue_annotation_logits`**

```python
def sample_residue_annotation_logits(
    logits: torch.Tensor, annotation_threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

- **目的**：从残基注释的 logits 中采样顶级注释。
- **功能**：
  - 对 logits 进行排序，选择前 `MAX_RESIDUE_ANNOTATIONS` 个。
  - 计算这些注释的概率，并根据 `annotation_threshold` 过滤低概率的注释。
  - 返回过滤后的注释索引和对应的 log 概率。

#### **d. `sample_sasa_logits`**

```python
def sample_sasa_logits(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    sampling_track_config: SamplingTrackConfig,
    mask_idx: int,
    valid_ids: list[int],
    mask_logits_of_invalid_ids: bool = True,
) -> torch.Tensor:
    ...
```

- **目的**：从 SASA（溶剂可及表面积）的 logits 中采样 SASA 值。
- **功能**：
  - 如果 `mask_logits_of_invalid_ids` 为真，屏蔽无效的 ID。
  - 计算 SASA 概率分布，并选择最大概率的索引。
  - 根据 `SASA_DISCRETIZATION_BOUNDARIES` 计算具体的 SASA 值。
  - 使用 `sampling_mask` 过滤不需要采样的位置，将不符合条件的位置设置为无穷大。
  - 返回最终的 SASA 值张量。

#### **e. `top_p_logits`**

```python
def top_p_logits(logits: torch.Tensor, top_p: float | torch.Tensor) -> torch.Tensor:
    ...
```

- **目的**：应用 top-p 过滤，保留累计概率不超过 `top_p` 的 logits。
- **功能**：
  - 对 logits 进行排序，计算累计概率。
  - 生成一个掩码，仅保留累计概率不超过 `top_p` 的令牌。
  - 确保至少保留一个令牌（即概率最高的令牌）。
  - 将不在 top-p 范围内的 logits 设置为负无穷大。
  - 返回经过 top-p 过滤后的 logits。

### **6. 总结**

`sampling.py` 模块通过定义一系列的类和函数，提供了灵活且高效的采样机制，适用于蛋白质序列和结构的生成与注释。其核心功能包括：

- **批处理支持**：通过 `_BatchedESMProteinTensor` 类，实现对批量蛋白质数据的高效管理和操作。
- **配置管理**：提供默认的采样配置生成和验证机制，确保采样过程的参数设置合理。
- **多样化的采样策略**：包括基于温度和 top-p 的采样、功能注释采样、残基注释采样和 SASA 采样等，满足不同的应用需求。
- **张量操作优化**：通过辅助函数如 `_tensorize_like` 和 `get_sampling_mask`，优化了张量的广播和掩码生成过程，提高了计算效率。

总体而言，该模块在处理复杂的蛋白质数据采样任务时，提供了强大而灵活的工具，适用于研究和应用中的各种需求。

## sasa_tokenizer-代码分析
**总体功能概述**：  
`SASADiscretizingTokenizer` 类的作用是将连续的溶剂可及表面积（Solvent Accessible Surface Area, 简称SASA）浮点数值根据给定的边界进行离散化分桶，将其映射成对应的离散“词元”（tokens），从而使得这些连续数据可以以类似文本的形式输入给语言模型或其他基于 token 的模型进行处理。该类同时支持特殊标记（如 `<pad>`, `<motif>`, `<unk>`），能够对数据进行编码、解码，并提供特定的属性帮助模型识别和处理特殊 token。

下面对代码各个部分进行详细分析：

### 类的继承与初始化

```python
class SASADiscretizingTokenizer(EsmTokenizerBase):
    """Tokenizer for Solvent Accessible Surface Area (SASA)."""

    def __init__(self, boundaries: list[float] = C.SASA_DISCRETIZATION_BOUNDARIES):
        self._boundaries = sorted(boundaries)
```

- `SASADiscretizingTokenizer` 继承自 `EsmTokenizerBase`，意味着它符合 ESM （一种蛋白质语言模型）相关的 Tokenizer 接口规范。
- `boundaries` 是一组浮点数阈值，用来将连续的 SASA 值分成若干区间。代码中对 `boundaries` 进行了排序（`sorted(boundaries)`），确保区间有序划分。

### 特殊Token

```python
@cached_property
def special_tokens(self) -> list[str]:
    return ["<pad>", "<motif>", "<unk>"]
```

- `special_tokens` 定义了三个特殊标记：`<pad>`, `<motif>`, `<unk>`。
- `<pad>` 通常用作填充符（padding token），`<unk>` 作为未知词元，`<motif>` 可以作为特定序列特征起标记。
- 在此实现中 `<pad>` 同时被用来表示多种特殊角色（包括 BOS、EOS、chain break 等），这在实际应用中也许比较特殊，但在此代码中是设计如此。

### 构建词表（Vocabulary）

```python
@cached_property
def vocab(self) -> list[str]:
    boundary_strs = ["0"] + [str(b) for b in self._boundaries] + ["inf"]
    range_tokens = [
        f"<{low}-{high}>"
        for low, high in zip(boundary_strs[:-1], boundary_strs[1:])
    ]
    return self.special_tokens + range_tokens
```

- 根据传入的 `boundaries`，通过 `boundary_strs` 数组构建一系列区间段的字符串形式，例如：`["0", "5.5", "12.0", "inf"]` 代表从 0 到 5.5、5.5 到 12.0、12.0 到无穷大(`inf`)等区间。
- 使用 `<low-high>` 的格式将每个区间转换成一个独立的 token。例如 `<0-5.5>` 、`<5.5-12>`、`<12-inf>`。
- 最终 `vocab` 的结构是：`[<pad>, <motif>, <unk>, <0-x>, <x-y>, ..., <z-inf>]`。
- 这样实现后，vocab中前几个是特殊符号，后面的是SASA值离散后的区间token。

### Midpoints（中点值）计算与Tensor表示

```python
@cached_property
def midpoints_tensor(self) -> torch.Tensor:
    boundaries = [0] + self._boundaries + [self._boundaries[-1] * 2]
    midpoint_tokens = [
        (float(high) + float(low)) / 2
        for low, high in zip(boundaries[:-1], boundaries[1:])
    ]
    midpoint_tokens = [float("nan"), float("nan"), float("nan")] + midpoint_tokens
    return torch.Tensor(midpoint_tokens)
```

- 为了在后续解码成浮点值时有个参考，本代码通过在每个区间的上下界之间取中点来表示该 token 对应的数值。
- 例如，区间 `<0-5.5>` 的中点为 (0 + 5.5) / 2 = 2.75。
- 第三个参数（`midpoint_tokens`）列表的前三个元素设为 `nan` 对应前三个特殊 token（`<pad>, <motif>, <unk>`），因为这些特殊 token 并没有数值意义。
- `midpoints_tensor` 返回一个 PyTorch Tensor，索引与 `vocab` 中token的顺序对应。

### vocab_to_index 映射

```python
@cached_property
def vocab_to_index(self) -> dict[str, int]:
    return {word: i for i, word in enumerate(self.vocab)}
```

- `vocab_to_index` 构建了从 token 字串到 token 索引（整数）的映射，以便快速由词元文本取得其对应的id。

### 特殊 token 掩码与检查

```python
def get_special_tokens_mask(self, tokens: torch.Tensor) -> torch.Tensor:
    return tokens < len(self.special_tokens)
```

- 给定一个 token id 的张量，可以通过检查 `id < special_tokens数量` 来判断该位置上是否是特殊 token。
- 返回一个布尔张量，True表示该位置为特殊 token。

### 编码方法（encode）

```python
def encode(
    self, values: list[float | str], add_special_tokens: bool = True
) -> torch.Tensor:
    ids = []
    if add_special_tokens:
        ids.append(self.vocab_to_index["<pad>"])  # BOS

    for value in values:
        if isinstance(value, (float, int)):
            bucket = torch.bucketize(value, torch.tensor(self._boundaries))
            token_id = len(self.special_tokens) + bucket
        elif isinstance(value, str):
            token_id = self.vocab_to_index[value]
        else:
            raise TypeError(value)
        ids.append(token_id)

    if add_special_tokens:
        ids.append(self.vocab_to_index["<pad>"])  # EOS

    return torch.tensor(ids, dtype=torch.int64)
```

- 传入一组 values（可以是浮点SASA值，也可以是已经存在于vocab中的特殊字符串token）。
- 如果是数值，会使用 `torch.bucketize` 根据 `boundaries` 找到其所属的区间下标，然后将该区间下标加上特殊token数目的偏移量，以得到对应token的id。
- 如果是字符串，则直接从 `vocab_to_index` 查找。
- `add_special_tokens` 为 True 时，在序列头尾分别添加 `<pad>` token 的 id，作为BOS和EOS。
- 最终返回一个整型张量。

### 解码方法

```python
def decode_float(self, encoded: torch.Tensor) -> list[float]:
    decoded = self.midpoints_tensor[encoded.cpu()]
    nan_mask = torch.isnan(decoded)
    np_arr = decoded.numpy()
    np_arr[nan_mask.numpy()] = None
    return np_arr.tolist()
```

- `decode_float` 将 token id 张量映射回浮点数表示，即通过 `midpoints_tensor` 查找对应中点值。
- 对于特殊 token（NaN），则返回 `None`。
- 最终以Python列表返回。

```python
def decode(self, encoded: torch.Tensor) -> str:
    return ",".join(self.vocab[i] for i in encoded)
```

- `decode` 将 token id 列表解码为对应的 token 字串，并用逗号拼接成一个字符串。

```python
def decode_list(self, encoded: torch.Tensor) -> list[str]:
    return [self.vocab[i] for i in encoded]
```

- `decode_list` 返回一个字符串列表的形式，更加灵活。

### 特定token的属性（mask, bos, eos, pad, chain_break）

```python
@property
def mask_token(self) -> str:
    return "<pad>"

@property
def bos_token(self) -> str:
    return "<pad>"

@property
def eos_token(self) -> str:
    return "<pad>"

@property
def pad_token(self) -> str:
    return "<pad>"

@property
def chain_break_token(self) -> str:
    return "<pad>"
```

- 在此实现中，`<pad>` 一并扮演了 mask, bos, eos, pad, chain_break 的角色。这在常规实现中较不寻常，但这里的设计显然是将 `<pad>` 作多用途特殊token。

```python
@property
def mask_token_id(self) -> int:
    return self.vocab_to_index[self.mask_token]

# 同理 bos_token_id, eos_token_id, pad_token_id, chain_break_token_id
```

- 提供这些属性方便外部使用时直接获得对应特殊token的ID。

### 所有 token ID 列表与特殊 token ID 列表

```python
@property
def all_token_ids(self):
    return list(range(len(self.vocab)))

@property
def special_token_ids(self):
    return [self.vocab_to_index[token] for token in self.special_tokens]
```

- `all_token_ids` 返回整个词表的 ID 列表（0 到 vocab大小-1）
- `special_token_ids` 返回所有特殊token的ID构成的列表。

---

**总结：**  
`SASADiscretizingTokenizer` 将连续的溶剂可及表面积值离散化为一系列区间 token，并可将这些 token 编码为ID供模型使用，也可将模型输出的ID解码回区间中点值或对应的字符串表示。同时，类中提供了特殊 token 的标记和处理方式，使得在构建输入序列时可以自动添加/识别特殊 token（例如 `<pad>` 用于填充或标记序列开始和结束）。  
该代码的主要用途是将连续的SASA数值特征转换为可输入至类似语言模型的架构中，从而进行下游的预测或分析。

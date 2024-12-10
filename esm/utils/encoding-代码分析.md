## encoding-代码分析
这段代码（`encoding.py`）主要实现了蛋白质相关数据的编码与标记化（tokenization）功能，它通过调用特定的Tokenizer和Encoder将生物学数据（如蛋白质序列、二级结构、溶剂可及表面积SASA、三维结构坐标以及功能性注释）转换为模型可处理的张量表示（tensor tokens），以用于下游的深度学习模型（如ESM系列模型）。

下面将对代码的功能与逻辑进行详细拆解：

### 全局概述

代码中定义了一系列函数和默认值，用于将不同类型的生物学数据（蛋白质序列、结构、二级结构、SASA值、功能注释等）转换为Token ID的形式。相关组件包括：

- **Tokenizer**: 不同类型的tokenizer负责对各类输入进行离散化和索引化。例如，对蛋白质序列使用`EsmSequenceTokenizer`，对三维结构使用`StructureTokenizer`和`StructureTokenEncoder`组合，对二级结构使用`SecondaryStructureTokenizer`，对SASA使用`SASADiscretizingTokenizer`，对功能注释使用`InterProQuantizedTokenizer`和`ResidueAnnotationsTokenizer`。
- **默认值生成函数**: 当实际数据缺失时，会使用Mask Token或默认值（如`C.MASK_STR_SHORT`或`None`）来填充序列，以确保长度一致并保持数据格式的完整性。
- **编码流程**: 针对不同输入数据类型的流程包括：
  1. 将输入数据进行适当的字符替换（将mask符号替换为tokenizer能识别的mask token）；
  2. 调用对应的tokenizer对数据编码（encode）以获得索引化的tensor；
  3. 对于结构数据，需要使用`StructureTokenEncoder`从坐标计算结构token。
  4. 根据需要添加BOS（序列起始）和EOS（序列结束）token；
  5. 若必要，还会进行张量的pad操作，使数据对齐。

### 具体函数解析

#### 默认填充值生成函数

1. `get_default_sequence(sequence_length: int) -> str`  
   返回给定长度的默认序列字符串，其中全部字符为`C.MASK_STR_SHORT`（即mask标记，用于在无真实序列输入时的占位）。

2. `get_default_secondary_structure(sequence_length: int) -> str`  
   与序列类似，返回给定长度的默认二级结构字符串，同样使用`C.MASK_STR_SHORT`填充。

3. `get_default_sasa(sequence_length: int) -> Sequence[float | str | None]`  
   返回给定长度的SASA默认值列表，用`None`占位，表示没有实际的SASA值。

#### 序列(tokenize_sequence)

```python
def tokenize_sequence(
    sequence: str,
    sequence_tokenizer: EsmSequenceTokenizer,
    add_special_tokens: bool = True,
) -> torch.Tensor:
```

- 将输入的氨基酸序列中所有的`C.MASK_STR_SHORT`替换为`sequence_tokenizer.mask_token`，保证tokenizer可以识别。  
- 使用`sequence_tokenizer.encode()`方法对序列进行编码。  
- `add_special_tokens=True`时会在编码的结果中自动加入BOS与EOS之类的特殊token。  
- 最终返回一个`torch.Tensor`类型的token索引张量。

#### 结构(tokenize_structure)

```python
def tokenize_structure(
    coordinates: torch.Tensor,
    structure_encoder: StructureTokenEncoder,
    structure_tokenizer: StructureTokenizer,
    reference_sequence: str = "",
    add_special_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

- 输入为蛋白质结构坐标`coordinates`（形状一般为[L, 37, 3]，L为残基数）。
- `ProteinChain.from_atom37()`根据坐标生成`ProteinChain`对象，此对象可根据坐标自动识别残基数量，并可关联参考序列（若提供）。
- `chain.to_structure_encoder_inputs()`返回可直接输入`StructureTokenEncoder`的数据，包括坐标、plDDT值及residue_index。
- 调用`structure_encoder.encode()`将坐标等信息转化为结构token（通常是将3D坐标数据通过卷积或者特定编码器转换为离散的结构表示）。
- 返回值包括坐标张量（L,37,3）、plddt张量（L,）、和结构token张量（L,）。
- 若`add_special_tokens=True`，则在token首尾添加BOS和EOS，同时对坐标和token进行相应的pad操作，BOS和EOS位置的坐标和plDDT值按指定的方式填充。

#### 二级结构(tokenize_secondary_structure)

```python
def tokenize_secondary_structure(
    secondary_structure: str | Sequence[str],
    secondary_structure_tokenizer: SecondaryStructureTokenizer,
    add_special_tokens: bool = True,
) -> torch.Tensor:
```

- 将输入的二级结构序列中mask标记(`C.MASK_STR_SHORT`)替换为`secondary_structure_tokenizer.mask_token`。
- 对二级结构逐字符进行token化编码。
- 若`add_special_tokens=True`会自动加入BOS与EOS token。
- 返回一个包含二级结构token id的张量。

#### SASA(tokenize_sasa)

```python
def tokenize_sasa(
    sasa: Sequence[float | str | None],
    sasa_tokenizer: SASADiscretizingTokenizer,
    add_special_tokens: bool = True,
):
```

- 对SASA序列中`None`值替换为`mask_token`。
- 调用`SASADiscretizingTokenizer.encode()`对序列离散化。  
- 加入特殊符号（BOS、EOS）后返回SASA token张量。

#### 功能注释(tokenize_function_annotations)

```python
def tokenize_function_annotations(
    function_annotations: Sequence[FunctionAnnotation],
    reference_sequence: str,
    function_tokenizer: EsmFunctionTokenizer,
    residue_annotation_tokenizer: ResidueAnnotationsTokenizer,
    add_special_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
```

- 利用`encode_function_annotations()`对功能注释进行编码。  
- 功能注释包含两个层面：全局的功能token与针对每个残基的注释token。  
- 返回分别是 function_tokens 和 residue_annotation_tokens 的张量。

#### 获取默认token张量函数

这些函数在没有实际数据时生成对应类型的默认token张量，包括BOS、EOS、以及用mask或pad填充的主体内容。这些函数与上面类似，但它们不对实际数据操作，而是直接根据tokenizer的配置信息生成全mask（或者全pad）的默认token序列。

1. `get_default_sequence_tokens(sequence_length: int, sequence_tokenizer: EsmSequenceTokenizer) -> torch.Tensor`  
   根据序列长度创建一个全mask的序列token张量，并在开头结尾加入BOS和EOS。

2. `get_default_structure_tokens(sequence_length: int, structure_tokenizer: StructureTokenizer) -> torch.Tensor`  
   类似序列token，生成结构token的默认张量，全使用mask token填充，并在首尾加入BOS、EOS。

3. `get_default_secondary_structure_tokens(sequence_length: int, secondary_structure_tokenizer: SecondaryStructureTokenizer) -> torch.Tensor`
   为二级结构生成默认的mask序列token。

4. `get_default_sasa_tokens(sequence_length: int, sasa_tokenizer: SASADiscretizingTokenizer) -> torch.Tensor`
   为SASA生成默认token。

5. `get_default_function_tokens(sequence_length: int, function_tokenizer: EsmFunctionTokenizer) -> torch.Tensor`
   为功能注释生成默认token，注意这里是二维的（(L+2) x depth），且用`pad_token_id`填充。

6. `get_default_residue_annotation_tokens(sequence_length: int, residue_annotation_tokenizer: ResidueAnnotationsTokenizer) -> torch.Tensor`
   为残基注释生成默认token，同样是二维的（(L+2) x C.MAX_RESIDUE_ANNOTATIONS），用`pad_token_id`填充并加入BOS和EOS。

### 总结

这段代码的核心是将多模态（序列、结构、注释等）蛋白质数据转化为统一的张量化表示，以输入至下游模型（如ESM系列）。各类Tokenizer、Encoder从字符或坐标等原始数据中生成对应的离散索引token。代码还提供了默认值生成功能，使得在数据缺失或者需要对模型进行预填充时，可以轻松获取mask或pad填充的token化张量。

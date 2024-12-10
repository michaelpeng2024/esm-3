## esm3-代码分析
这段代码 `esm3.py` 主要用于配置和管理与蛋白质序列和结构相关的数据、模型以及各种常量。以下是对代码各部分功能的详细分析：

### 1. **导入模块**

```python
import os
from functools import cache
from pathlib import Path

from huggingface_hub import snapshot_download
```

- **os**: 用于与操作系统进行交互，如访问环境变量。
- **functools.cache**: 用于缓存函数的返回值，以提高性能。
- **pathlib.Path**: 提供面向对象的文件系统路径操作。
- **huggingface_hub.snapshot_download**: 从 Hugging Face Hub 下载模型快照。

### 2. **常量定义**

代码中定义了多个常量，主要用于标识特殊的序列标记、VQVAE（向量量化变分自编码器）相关参数、结构标记、掩码标记以及词汇表等。

#### 2.1 **序列相关的常量**

```python
SEQUENCE_BOS_TOKEN = 0
SEQUENCE_PAD_TOKEN = 1
SEQUENCE_EOS_TOKEN = 2
SEQUENCE_CHAINBREAK_TOKEN = 31
SEQUENCE_MASK_TOKEN = 32
```

- **SEQUENCE_BOS_TOKEN**: 序列开始标记，值为0。
- **SEQUENCE_PAD_TOKEN**: 填充标记，值为1。
- **SEQUENCE_EOS_TOKEN**: 序列结束标记，值为2。
- **SEQUENCE_CHAINBREAK_TOKEN**: 链断裂标记，值为31。
- **SEQUENCE_MASK_TOKEN**: 掩码标记，值为32。

#### 2.2 **VQVAE相关常量**

```python
VQVAE_CODEBOOK_SIZE = 4096
VQVAE_SPECIAL_TOKENS = {
    "MASK": VQVAE_CODEBOOK_SIZE,
    "EOS": VQVAE_CODEBOOK_SIZE + 1,
    "BOS": VQVAE_CODEBOOK_SIZE + 2,
    "PAD": VQVAE_CODEBOOK_SIZE + 3,
    "CHAINBREAK": VQVAE_CODEBOOK_SIZE + 4,
}
VQVAE_DIRECTION_LOSS_BINS = 16
VQVAE_PAE_BINS = 64
VQVAE_MAX_PAE_BIN = 31.0
VQVAE_PLDDT_BINS = 50
```

- **VQVAE_CODEBOOK_SIZE**: 码本大小，4096。
- **VQVAE_SPECIAL_TOKENS**: 定义了掩码、序列开始、结束、填充和链断裂的特殊标记，这些标记的值从4096开始递增。
- **VQVAE_DIRECTION_LOSS_BINS**、**VQVAE_PAE_BINS**、**VQVAE_MAX_PAE_BIN**、**VQVAE_PLDDT_BINS**: 与VQVAE模型损失计算和预测相关的参数，具体用于方向损失、PAE（Predicted Aligned Error）和PLDDT（Predicted Local Distance Difference Test）的离散化。

#### 2.3 **结构相关的常量**

```python
STRUCTURE_MASK_TOKEN = VQVAE_SPECIAL_TOKENS["MASK"]
STRUCTURE_BOS_TOKEN = VQVAE_SPECIAL_TOKENS["BOS"]
STRUCTURE_EOS_TOKEN = VQVAE_SPECIAL_TOKENS["EOS"]
STRUCTURE_PAD_TOKEN = VQVAE_SPECIAL_TOKENS["PAD"]
STRUCTURE_CHAINBREAK_TOKEN = VQVAE_SPECIAL_TOKENS["CHAINBREAK"]
STRUCTURE_UNDEFINED_TOKEN = 955
```

- 将VQVAE的特殊标记映射到结构相关的标记。
- **STRUCTURE_UNDEFINED_TOKEN**: 未定义的结构标记，值为955。

#### 2.4 **其他标记**

```python
SASA_PAD_TOKEN = 0
SS8_PAD_TOKEN = 0
INTERPRO_PAD_TOKEN = 0
RESIDUE_PAD_TOKEN = 0
```

- **SASA_PAD_TOKEN**、**SS8_PAD_TOKEN**、**INTERPRO_PAD_TOKEN**、**RESIDUE_PAD_TOKEN**: 分别用于SASA（溶剂可及表面积）、SS8（8类二级结构）、InterPro和残基的填充标记，均为0。

#### 2.5 **字符串标记**

```python
CHAIN_BREAK_STR = "|"

SEQUENCE_BOS_STR = "<cls>"
SEQUENCE_EOS_STR = "<eos>"

MASK_STR_SHORT = "_"
SEQUENCE_MASK_STR = "<mask>"
SASA_MASK_STR = "<unk>"
SS8_MASK_STR = "<unk>"
```

- **CHAIN_BREAK_STR**: 链断裂的字符串表示。
- **SEQUENCE_BOS_STR**、**SEQUENCE_EOS_STR**: 序列开始和结束的字符串表示。
- **MASK_STR_SHORT**、**SEQUENCE_MASK_STR**、**SASA_MASK_STR**、**SS8_MASK_STR**: 掩码的不同字符串表示。

#### 2.6 **词汇表**

```python
# fmt: off
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]
# fmt: on

SSE_8CLASS_VOCAB = "GHITEBSC"
SSE_3CLASS_VOCAB = "HEC"
SSE_8CLASS_TO_3CLASS_MAP = {
    "G": "H",
    "H": "H",
    "I": "H",
    "T": "C",
    "E": "E",
    "B": "E",
    "S": "C",
    "C": "C",
}
```

- **SEQUENCE_VOCAB**: 定义了序列的词汇表，包括特殊标记和氨基酸单字母代码。
- **SSE_8CLASS_VOCAB**: 8类二级结构的词汇表。
- **SSE_3CLASS_VOCAB**: 3类二级结构的词汇表。
- **SSE_8CLASS_TO_3CLASS_MAP**: 将8类二级结构映射到3类的转换字典。

#### 2.7 **SASA离散化边界**

```python
SASA_DISCRETIZATION_BOUNDARIES = [
    0.8,
    4.0,
    9.6,
    16.4,
    24.5,
    32.9,
    42.0,
    51.5,
    61.2,
    70.9,
    81.6,
    93.3,
    107.2,
    125.4,
    151.4,
]
```

- 定义了SASA（溶剂可及表面积）离散化的边界，用于将连续的SASA值转换为离散的类别。

#### 2.8 **其他常量**

```python
MAX_RESIDUE_ANNOTATIONS = 16

TFIDF_VECTOR_SIZE = 58641
```

- **MAX_RESIDUE_ANNOTATIONS**: 每个残基的最大注释数，值为16。
- **TFIDF_VECTOR_SIZE**: TF-IDF向量的大小，值为58641。

### 3. **函数定义**

#### 3.1 **data_root函数**

```python
@staticmethod
@cache
def data_root(model: str):
    if "INFRA_PROVIDER" in os.environ:
        return Path("")
    # Try to download from hugginface if it doesn't exist
    if model.startswith("esm3"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esm3-sm-open-v1"))
    elif model.startswith("esmc-300"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12"))
    elif model.startswith("esmc-600"):
        path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-600m-2024-12"))
    else:
        raise ValueError(f"{model=} is an invalid model name.")
    return path
```

- **功能**: 根据提供的模型名称，确定并返回数据根路径。如果环境变量 `INFRA_PROVIDER` 存在，则返回当前路径；否则，从 Hugging Face Hub 下载相应的模型快照。
- **参数**: `model` - 模型名称字符串。
- **缓存**: 使用 `@cache` 装饰器缓存函数结果，避免重复下载。
- **支持的模型**:
  - 以 `esm3` 开头的模型下载自 `EvolutionaryScale/esm3-sm-open-v1`。
  - 以 `esmc-300` 开头的模型下载自 `EvolutionaryScale/esmc-300m-2024-12`。
  - 以 `esmc-600` 开头的模型下载自 `EvolutionaryScale/esmc-600m-2024-12`。
- **错误处理**: 如果模型名称不符合上述模式，抛出 `ValueError`。

### 4. **数据路径配置**

```python
IN_REPO_DATA_FOLDER = Path(__file__).parents[2] / "data"

INTERPRO_ENTRY = IN_REPO_DATA_FOLDER / "entry_list_safety_29026.list"
INTERPRO_HIERARCHY = IN_REPO_DATA_FOLDER / "ParentChildTreeFile.txt"
INTERPRO2GO = IN_REPO_DATA_FOLDER / "ParentChildTreeFile.txt"
INTERPRO_2ID = "data/tag_dict_4_safety_filtered.json"

LSH_TABLE_PATHS = {"8bit": "data/hyperplanes_8bit_58641.npz"}

KEYWORDS_VOCABULARY = (
    IN_REPO_DATA_FOLDER / "keyword_vocabulary_safety_filtered_58641.txt"
)
KEYWORDS_IDF = IN_REPO_DATA_FOLDER / "keyword_idf_safety_filtered_58641.npy"

RESID_CSV = "data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv"
INTERPRO2KEYWORDS = IN_REPO_DATA_FOLDER / "interpro_29026_to_keywords_58641.csv"
```

- **IN_REPO_DATA_FOLDER**: 定义了数据文件夹的根路径，位于当前文件的上两级目录中的 `data` 文件夹。
- **INTERPRO_ENTRY**、**INTERPRO_HIERARCHY**、**INTERPRO2GO**: 指向 InterPro 相关的数据文件，用于蛋白质功能和结构的注释。
- **INTERPRO_2ID**: 指向一个 JSON 文件，可能用于 InterPro ID 的映射。
- **LSH_TABLE_PATHS**: 定义了局部敏感哈希（LSH）表的路径，当前支持 `8bit` 类型。
- **KEYWORDS_VOCABULARY**、**KEYWORDS_IDF**: 指向关键词词汇表和对应的逆文档频率（IDF）文件，用于关键词相关的处理。
- **RESID_CSV**: 指向一个 CSV 文件，包含 UniRef90 和 MGnify90 数据集中超过1000个蛋白质的残基注释。
- **INTERPRO2KEYWORDS**: 指向一个 CSV 文件，用于将 InterPro 条目映射到关键词。

### 5. **总结**

整体而言，`esm3.py` 文件主要承担以下功能：

1. **定义常量**：包括序列标记、VQVAE相关参数、结构标记、掩码标记、词汇表等，为后续的模型训练和推理提供基础配置。
2. **管理数据路径**：通过 `data_root` 函数和路径配置，确保所需的数据和模型能够正确加载和访问。
3. **支持模型下载**：利用 Hugging Face Hub 提供的 `snapshot_download` 方法，根据模型名称自动下载和缓存相应的模型数据。

这些配置和路径管理对于构建和运行基于 ESM-3 和 ESMC 模型的蛋白质序列和结构分析系统至关重要。通过集中管理这些常量和路径，代码确保了系统的可维护性和扩展性，使得不同部分的模块能够一致且高效地访问所需资源。

## protein_complex-代码分析
这段代码 `protein_complex.py` 实现了一个用于表示和操作蛋白质复合物的类及相关功能。该模块主要依赖于多个生物信息学和科学计算库，如 `biotite`, `numpy`, `torch` 等，结合数据类 (`dataclass`) 和其他现代 Python 特性来高效管理蛋白质结构数据。以下是对代码各部分功能的详细分析：

## 1. 导入模块和初始化

```python
from __future__ import annotations

import io
import itertools
import re
import warnings
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from subprocess import check_output
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Sequence

import biotite.structure as bs
import brotli
import msgpack
import msgpack_numpy
import numpy as np
import torch
from biotite.database import rcsb
from biotite.structure.io.pdb import PDBFile

from esm.utils import residue_constants
from esm.utils.constants import esm3 as esm3_c
from esm.utils.misc import slice_python_object_as_numpy
from esm.utils.structure.affine3d import Affine3D
from esm.utils.structure.aligner import Aligner
from esm.utils.structure.metrics import (
    compute_gdt_ts,
    compute_lddt_ca,
)
from esm.utils.structure.protein_chain import (
    PathOrBuffer,
    ProteinChain,
)
from esm.utils.structure.protein_structure import (
    index_by_atom_name,
)

msgpack_numpy.patch()

SINGLE_LETTER_CHAIN_IDS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)
```

### 功能说明：

- **未来导入**：使用 `from __future__ import annotations` 以支持未来的类型注解特性，延迟类型注解解析，提高代码性能和兼容性。
- **标准库导入**：包括 `io`, `itertools`, `re`, `warnings`, `dataclasses`, `functools`, `pathlib`, `subprocess`, `tempfile`, `typing` 等，用于文件操作、正则表达式处理、数据类定义、缓存属性、路径操作、子进程管理、临时目录创建和类型注解。
- **第三方库导入**：
  - `biotite`：用于生物信息学中的结构操作。
  - `brotli`：用于数据压缩。
  - `msgpack` 和 `msgpack_numpy`：用于高效的数据序列化，特别是与 NumPy 结合。
  - `numpy` 和 `torch`：用于数值计算和深度学习。
  - `esm.utils` 下的各种模块：用于残基常量、仿射变换、对齐、度量计算和蛋白链结构处理。
- **初始化**：
  - `msgpack_numpy.patch()`：扩展 `msgpack` 以支持 NumPy 数据类型。
  - `SINGLE_LETTER_CHAIN_IDS`：定义了一组单字母链标识符，用于标识蛋白质复合物中的不同链。

## 2. 函数 `protein_chain_to_protein_complex`

```python
def protein_chain_to_protein_complex(chain: ProteinChain) -> ProteinComplex:
    if "|" not in chain.sequence:
        return ProteinComplex.from_chains([chain])
    chain_breaks = np.array(list(chain.sequence)) == "|"
    chain_break_inds = np.where(chain_breaks)[0]
    chain_break_inds = np.concatenate([[0], chain_break_inds, [len(chain)]])
    chain_break_inds = np.array(list(zip(chain_break_inds[:-1], chain_break_inds[1:])))
    complex_chains = []
    for start, end in chain_break_inds:
        if start != 0:
            start += 1
        complex_chains.append(chain[start:end])
    complex_chains = [
        ProteinChain.from_atom37(
            chain.atom37_positions,
            sequence=chain.sequence,
            chain_id=SINGLE_LETTER_CHAIN_IDS[i],
            entity_id=i,
        )
        for i, chain in enumerate(complex_chains)
    ]
    return ProteinComplex.from_chains(complex_chains)
```

### 功能说明：

- **作用**：将一个 `ProteinChain` 对象转换为 `ProteinComplex` 对象。如果序列中包含 `"|"`，则表示链之间有断裂，需要将其拆分为多个链。
- **步骤**：
  1. 检查链的序列中是否包含 `"|"`，若没有，直接将单个链转换为复合物。
  2. 找出序列中 `"|"` 的位置，作为链的断裂点。
  3. 根据断裂点将序列拆分为多个子链。
  4. 为每个子链分配一个唯一的链ID和实体ID。
  5. 使用 `ProteinComplex.from_chains` 方法将子链列表组合成一个 `ProteinComplex` 对象。

## 3. 数据类定义

### 3.1 `ProteinComplexMetadata`

```python
@dataclass
class ProteinComplexMetadata:
    entity_lookup: dict[int, int]
    chain_lookup: dict[int, str]
    chain_boundaries: list[tuple[int, int]]
```

### 功能说明：

- **作用**：存储蛋白质复合物的元数据，包括实体查找表、链查找表和链边界信息。
- **字段**：
  - `entity_lookup`：实体ID到唯一序列ID的映射。
  - `chain_lookup`：链ID到链标识符的映射。
  - `chain_boundaries`：每条链在整体序列中的起始和结束位置。

### 3.2 `DockQSingleScore` 和 `DockQResult`

```python
@dataclass
class DockQSingleScore:
    native_chains: tuple[str, str]
    DockQ: float
    interface_rms: float
    ligand_rms: float
    fnat: float
    fnonnat: float
    clashes: float
    F1: float
    DockQ_F1: float

@dataclass
class DockQResult:
    total_dockq: float
    native_interfaces: int
    chain_mapping: dict[str, str]
    interfaces: dict[tuple[str, str], DockQSingleScore]
    aligned: ProteinComplex
    aligned_rmsd: float
```

### 功能说明：

- **DockQSingleScore**：
  - 存储单个接口的 DockQ 评分，包括原生链对、DockQ 值、界面RMS、配体RMS、接触比例等指标。
- **DockQResult**：
  - 存储整体 DockQ 评分结果，包括总DockQ值、原生接口数量、链映射关系、各接口的评分详情、对齐后的复合物以及对齐的RMSD值。

### 3.3 `AtomIndexer`

```python
class AtomIndexer:
    def __init__(self, structure: ProteinComplex, property: str, dim: int):
        self.structure = structure
        self.property = property
        self.dim = dim

    def __getitem__(self, atom_names: str | list[str]) -> np.ndarray:
        return index_by_atom_name(
            getattr(self.structure, self.property), atom_names, self.dim
        )
```

### 功能说明：

- **作用**：提供对蛋白质复合物中原子属性的索引访问，如位置、掩码等。
- **方法**：
  - `__getitem__`：根据原子名称（单个或列表）返回对应的属性值数组。

## 4. 数据类 `ProteinComplex`

```python
@dataclass
class ProteinComplex:
    """Dataclass with atom37 representation of an entire protein complex."""

    id: str
    sequence: str
    entity_id: np.ndarray  # entities map to unique sequences
    chain_id: np.ndarray  # multiple chains might share an entity id
    sym_id: np.ndarray  # complexes might be copies of the same chain
    residue_index: np.ndarray
    insertion_code: np.ndarray
    atom37_positions: np.ndarray
    atom37_mask: np.ndarray
    confidence: np.ndarray
    metadata: ProteinComplexMetadata
```

### 功能说明：

- **作用**：表示一个蛋白质复合物，包含其序列、原子位置、掩码、置信度等信息。
- **字段**：
  - `id`：复合物的标识符。
  - `sequence`：氨基酸序列。
  - `entity_id`、`chain_id`、`sym_id`：用于标识实体、链和对称性。
  - `residue_index`：残基索引。
  - `insertion_code`：插入码。
  - `atom37_positions`：每个残基的37个原子的位置。
  - `atom37_mask`：原子的掩码，指示哪些原子存在。
  - `confidence`：置信度评分。
  - `metadata`：元数据，包含链和实体的查找表及边界信息。

### 4.1 `__post_init__` 方法

```python
def __post_init__(self):
    l = len(self.sequence)
    assert self.atom37_positions.shape[0] == l, (self.atom37_positions.shape, l)
    assert self.atom37_mask.shape[0] == l, (self.atom37_mask.shape, l)
    assert self.residue_index.shape[0] == l, (self.residue_index.shape, l)
    assert self.insertion_code.shape[0] == l, (self.insertion_code.shape, l)
    assert self.confidence.shape[0] == l, (self.confidence.shape, l)
    assert self.entity_id.shape[0] == l, (self.entity_id.shape, l)
    assert self.chain_id.shape[0] == l, (self.chain_id.shape, l)
    assert self.sym_id.shape[0] == l, (self.sym_id.shape, l)
```

### 功能说明：

- **作用**：在数据类实例化后进行一致性检查，确保所有相关数组的长度与序列长度匹配。
- **检查内容**：`atom37_positions`、`atom37_mask`、`residue_index`、`insertion_code`、`confidence`、`entity_id`、`chain_id`、`sym_id` 的第一个维度长度应与序列长度一致。

### 4.2 `__getitem__` 方法

```python
def __getitem__(self, idx: int | list[int] | slice | np.ndarray):
    """This function slices protein complexes without consideration of chain breaks
    NOTE: When slicing with a boolean mask, it's possible that the output array won't
    be the expected length. This is because we do our best to preserve chainbreak tokens.
    """

    if isinstance(idx, int):
        idx = [idx]
    if isinstance(idx, list):
        raise ValueError(
            "ProteinComplex doesn't supports indexing with lists of indices"
        )

    if isinstance(idx, np.ndarray):
        is_chainbreak = np.asarray([s == "|" for s in self.sequence])
        idx = idx.astype(bool) | is_chainbreak

    complex = self._unsafe_slice(idx)
    if len(complex) == 0:
        return complex

    # detect runs of chainbreaks by searching for instances of '||' in complex.sequence
    chainbreak_runs = np.asarray(
        [
            complex.sequence[i : i + 2] == "||"
            for i in range(len(complex.sequence) - 1)
        ]
        + [complex.sequence[-1] == "|"]
    )
    # We should remove as many chainbreaks as possible from the start of the sequence
    for i in range(len(chainbreak_runs)):
        if complex.sequence[i] == "|":
            chainbreak_runs[i] = True
        else:
            break
    complex = complex._unsafe_slice(~chainbreak_runs)
    return complex
```

### 功能说明：

- **作用**：支持对 `ProteinComplex` 对象的切片操作，但不考虑链断裂。
- **支持的索引类型**：整数、切片、NumPy 数组。列表索引不支持，会抛出错误。
- **处理逻辑**：
  1. 如果索引是整数，转换为列表。
  2. 如果是列表，抛出错误提示不支持。
  3. 如果是 NumPy 数组，结合序列中的 `"|"` 作为链断裂标记，对索引进行逻辑或操作，保留链断裂位置。
  4. 使用 `_unsafe_slice` 方法进行实际切片。
  5. 处理连续的链断裂标记，去除多余的链断裂。
  6. 返回切片后的 `ProteinComplex` 对象。

### 4.3 `_unsafe_slice` 方法

```python
def _unsafe_slice(self, idx: int | list[int] | slice | np.ndarray):
    sequence = slice_python_object_as_numpy(self.sequence, idx)
    return replace(
        self,
        sequence=sequence,
        entity_id=self.entity_id[..., idx],
        chain_id=self.chain_id[..., idx],
        sym_id=self.sym_id[..., idx],
        residue_index=self.residue_index[..., idx],
        insertion_code=self.insertion_code[..., idx],
        atom37_positions=self.atom37_positions[..., idx, :, :],
        atom37_mask=self.atom37_mask[..., idx, :],
        confidence=self.confidence[..., idx],
    )
```

### 功能说明：

- **作用**：根据给定的索引对 `ProteinComplex` 对象进行切片，返回新的 `ProteinComplex` 实例。
- **实现方式**：使用 `dataclasses.replace` 方法创建一个新的实例，切片各个相关属性。

### 4.4 `__len__` 方法

```python
def __len__(self):
    return len(self.sequence)
```

### 功能说明：

- **作用**：返回蛋白质复合物中残基的数量，即序列长度。

### 4.5 `atoms` 属性

```python
@cached_property
def atoms(self) -> AtomIndexer:
    return AtomIndexer(self, property="atom37_positions", dim=-2)
```

### 功能说明：

- **作用**：懒加载属性，返回一个 `AtomIndexer` 实例，用于索引和访问原子位置数据。
- **特点**：使用 `cached_property` 装饰器，确保属性只计算一次，后续访问直接返回缓存结果。

### 4.6 `chain_iter` 方法

```python
def chain_iter(self) -> Iterable[ProteinChain]:
    boundaries = [i for i, s in enumerate(self.sequence) if s == "|"]
    boundaries = [-1, *boundaries, len(self)]
    for i in range(len(boundaries) - 1):
        c = self.__getitem__(slice(boundaries[i] + 1, boundaries[i + 1]))
        yield c.as_chain()
```

### 功能说明：

- **作用**：生成蛋白质复合物中每条链的 `ProteinChain` 对象。
- **实现方式**：
  1. 找出序列中 `"|"` 的位置，作为链的断裂点。
  2. 根据断裂点切片生成各个子链。
  3. 将每个子链转换为 `ProteinChain` 对象并生成。

### 4.7 `as_chain` 方法

```python
def as_chain(self, force_conversion: bool = False) -> ProteinChain:
    """Convert the ProteinComplex to a ProteinChain.

    Args:
        force_conversion (bool): Forces the conversion into a protein chain even if the complex has multiple chains.
            The purpose of this is to use ProteinChain specific functions (like cbeta_contacts).

    """
    if not force_conversion:
        assert len(np.unique(self.chain_id)) == 1, f"{self.id}"
        assert len(np.unique(self.entity_id)) == 1, f"{self.id}"
        if self.chain_id[0] not in self.metadata.chain_lookup:
            warnings.warn("Chain ID not found in metadata, using 'A' as default")
        if self.entity_id[0] not in self.metadata.entity_lookup:
            warnings.warn("Entity ID not found in metadata, using None as default")
        chain_id = self.metadata.chain_lookup.get(self.chain_id[0], "A")
        entity_id = self.metadata.entity_lookup.get(self.entity_id[0], None)
    else:
        chain_id = "A"
        entity_id = None

    return ProteinChain(
        id=self.id,
        sequence=self.sequence,
        chain_id=chain_id,
        entity_id=entity_id,
        atom37_positions=self.atom37_positions,
        atom37_mask=self.atom37_mask,
        residue_index=self.residue_index,
        insertion_code=self.insertion_code,
        confidence=self.confidence,
    )
```

### 功能说明：

- **作用**：将 `ProteinComplex` 转换为单个 `ProteinChain` 对象。
- **参数**：
  - `force_conversion`：强制转换，即使复合物包含多条链，也将其转换为单链。默认情况下，如果复合物包含多条链，会触发断言错误。
- **实现逻辑**：
  - 如果不强制转换，检查复合物是否仅包含单一链和单一实体。如果元数据中缺少链ID或实体ID的映射，则发出警告并使用默认值。
  - 如果强制转换，使用默认的链ID `"A"` 和 `entity_id` 为 `None`。
  - 创建并返回一个新的 `ProteinChain` 对象，包含复合物的所有相关属性。

### 4.8 类方法 `from_pdb` 和 `from_rcsb`

```python
@classmethod
def from_pdb(cls, path: PathOrBuffer, id: str | None = None) -> "ProteinComplex":
    atom_array = PDBFile.read(path).get_structure(
        model=1, extra_fields=["b_factor"]
    )

    chains = []
    for chain in bs.chain_iter(atom_array):
        chain = chain[~chain.hetero]
        if len(chain) == 0:
            continue
        chains.append(ProteinChain.from_atomarray(chain, id))
    return ProteinComplex.from_chains(chains)

@classmethod
def from_rcsb(cls, pdb_id: str):
    """Fetch a protein complex from the RCSB PDB database."""
    f: io.StringIO = rcsb.fetch(pdb_id, "pdb")  # type: ignore
    return cls.from_pdb(f, id=pdb_id)
```

### 功能说明：

- **from_pdb**：
  - **作用**：从PDB文件创建一个 `ProteinComplex` 对象。
  - **参数**：
    - `path`：PDB文件的路径或缓冲区。
    - `id`：复合物的标识符，默认为 `None`。
  - **实现步骤**：
    1. 使用 `biotite` 读取PDB文件，获取原子数组。
    2. 遍历原子数组中的每条链，过滤掉杂合原子（`hetero`）。
    3. 将每条有效链转换为 `ProteinChain` 对象，并收集到链列表中。
    4. 使用 `ProteinComplex.from_chains` 方法将链列表组合成一个 `ProteinComplex` 对象。

- **from_rcsb**：
  - **作用**：从RCSB PDB数据库中获取PDB ID对应的蛋白质复合物。
  - **参数**：
    - `pdb_id`：PDB数据库中的蛋白质ID。
  - **实现步骤**：
    1. 使用 `biotite` 的 `rcsb.fetch` 方法从RCSB数据库获取PDB文件内容。
    2. 调用 `from_pdb` 方法解析PDB内容并创建 `ProteinComplex` 对象。

### 4.9 `to_pdb` 和 `to_pdb_string` 方法

```python
def to_pdb(self, path: PathOrBuffer, include_insertions: bool = True):
    atom_array = None
    for chain in self.chain_iter():
        carr = (
            chain.atom_array
            if include_insertions
            else chain.atom_array_no_insertions
        )
        atom_array = carr if atom_array is None else atom_array + carr
    f = PDBFile()
    f.set_structure(atom_array)
    f.write(path)

def to_pdb_string(self, include_insertions: bool = True) -> str:
    buf = io.StringIO()
    self.to_pdb(buf, include_insertions=include_insertions)
    buf.seek(0)
    return buf.read()
```

### 功能说明：

- **to_pdb**：
  - **作用**：将 `ProteinComplex` 对象写入PDB文件。
  - **参数**：
    - `path`：目标PDB文件的路径或缓冲区。
    - `include_insertions`：是否包含插入码的原子，默认为 `True`。
  - **实现步骤**：
    1. 遍历复合物中的每条链，选择包含或不包含插入码的原子数组。
    2. 将所有链的原子数组合并为一个整体原子数组。
    3. 使用 `biotite` 的 `PDBFile` 类将合并后的原子数组写入指定路径。

- **to_pdb_string**：
  - **作用**：将 `ProteinComplex` 对象转换为PDB格式的字符串。
  - **参数**：
    - `include_insertions`：是否包含插入码的原子，默认为 `True`。
  - **实现步骤**：
    1. 创建一个 `StringIO` 缓冲区。
    2. 调用 `to_pdb` 方法将PDB内容写入缓冲区。
    3. 返回缓冲区的字符串内容。

### 4.10 `normalize_chain_ids_for_pdb` 方法

```python
def normalize_chain_ids_for_pdb(self):
    # Since PDB files have 1-letter chain IDs and don't support the idea of a symmetric index,
    # we can normalize it instead which might be necessary for DockQ and to_pdb.
    ids = SINGLE_LETTER_CHAIN_IDS
    chains = []
    for i, chain in enumerate(self.chain_iter()):
        chain.chain_id = ids[i]
        if i > len(ids):
            raise RuntimeError("Too many chains to write to PDB file")
        chains.append(chain)

    return ProteinComplex.from_chains(chains)
```

### 功能说明：

- **作用**：规范化链ID以符合PDB文件的单字母链ID要求，避免链ID重复或超出范围。
- **实现逻辑**：
  1. 遍历复合物中的每条链。
  2. 为每条链分配一个唯一的单字母链ID。
  3. 如果链数量超过预定义的链ID数量，抛出错误。
  4. 使用 `from_chains` 方法重新组合规范化后的链列表，返回新的 `ProteinComplex` 对象。

### 4.11 序列化和反序列化方法

```python
def state_dict(self, backbone_only=False):
    """This state dict is optimized for storage, so it turns things to fp16 whenever
    possible. Note that we also only support int32 residue indices, I'm hoping we don't
    need more than 2**32 residues..."""
    dct = {k: v for k, v in vars(self).items()}
    for k, v in dct.items():
        if isinstance(v, np.ndarray):
            match v.dtype:
                case np.int64:
                    dct[k] = v.astype(np.int32)
                case np.float64 | np.float32:
                    dct[k] = v.astype(np.float16)
                case _:
                    pass
        elif isinstance(v, ProteinComplexMetadata):
            dct[k] = asdict(v)
    dct["atom37_positions"] = dct["atom37_positions"][dct["atom37_mask"]]
    return dct

def to_blob(self, backbone_only=False) -> bytes:
    return brotli.compress(msgpack.dumps(self.state_dict(backbone_only)), quality=5)

@classmethod
def from_state_dict(cls, dct):
    atom37 = np.full((*dct["atom37_mask"].shape, 3), np.nan)
    atom37[dct["atom37_mask"]] = dct["atom37_positions"]
    dct["atom37_positions"] = atom37
    dct = {
        k: (v.astype(np.float32) if k in ["atom37_positions", "confidence"] else v)
        for k, v in dct.items()
    }
    dct["metadata"] = ProteinComplexMetadata(**dct["metadata"])
    return cls(**dct)

@classmethod
def from_blob(cls, input: Path | str | io.BytesIO | bytes):
    """NOTE(@zlin): blob + sparse coding + brotli + fp16 reduces memory
    of chains from 52G/1M chains to 20G/1M chains, I think this is a good first
    shot at compressing and dumping chains to disk. I'm sure there's better ways."""
    match input:
        case Path() | str():
            bytes = Path(input).read_bytes()
        case io.BytesIO():
            bytes = input.getvalue()
        case _:
            bytes = input
    return cls.from_state_dict(
        msgpack.loads(brotli.decompress(bytes), strict_map_key=False)
    )
```

### 功能说明：

- **state_dict**：
  - **作用**：将 `ProteinComplex` 对象转换为一个字典，优化存储，使用半精度浮点数（`fp16`）和32位整数（`int32`）减少内存占用。
  - **细节**：
    - 遍历对象属性，将 `int64` 转换为 `int32`，`float64` 和 `float32` 转换为 `float16`。
    - 将 `ProteinComplexMetadata` 转换为字典。
    - 根据 `atom37_mask` 筛选 `atom37_positions`，仅保留有效原子的位置。

- **to_blob**：
  - **作用**：将 `ProteinComplex` 对象序列化为压缩的二进制数据。
  - **实现方式**：先调用 `state_dict` 获取优化后的字典，然后使用 `msgpack` 进行序列化，最后使用 `brotli` 压缩。

- **from_state_dict**：
  - **作用**：从状态字典中恢复 `ProteinComplex` 对象。
  - **实现步骤**：
    1. 根据 `atom37_mask` 重建 `atom37_positions`，将有效位置设置为原始值，无效位置填充为 `NaN`。
    2. 将特定字段转换为 `float32` 类型。
    3. 重建 `ProteinComplexMetadata` 对象。
    4. 使用解包运算符 `**` 创建 `ProteinComplex` 实例。

- **from_blob**：
  - **作用**：从压缩的二进制数据恢复 `ProteinComplex` 对象。
  - **实现步骤**：
    1. 根据输入类型（路径、字符串、`BytesIO` 或字节）读取压缩数据。
    2. 使用 `brotli` 解压缩，然后使用 `msgpack` 反序列化。
    3. 调用 `from_state_dict` 方法恢复对象。

### 4.12 `from_chains` 类方法

```python
@classmethod
def from_chains(cls, chains: Sequence[ProteinChain]):
    if not chains:
        raise ValueError(
            "Cannot create a ProteinComplex from an empty list of chains"
        )

    # TODO: Make a proper protein complex class
    def join_arrays(arrays: Sequence[np.ndarray], sep: np.ndarray):
        full_array = []
        for array in arrays:
            full_array.append(array)
            full_array.append(sep)
        full_array = full_array[:-1]
        return np.concatenate(full_array, 0)

    sep_tokens = {
        "residue_index": np.array([-1]),
        "insertion_code": np.array([""]),
        "atom37_positions": np.full([1, 37, 3], np.nan),
        "atom37_mask": np.zeros([1, 37], dtype=bool),
        "confidence": np.array([0]),
    }

    array_args: dict[str, np.ndarray] = {
        name: join_arrays([getattr(chain, name) for chain in chains], sep)
        for name, sep in sep_tokens.items()
    }

    multimer_arrays = []
    chain2num_max = -1
    chain2num = {}
    ent2num_max = -1
    ent2num = {}
    total_index = 0
    chain_boundaries = []
    for i, c in enumerate(chains):
        num_res = c.residue_index.shape[0]
        if c.chain_id not in chain2num:
            chain2num[c.chain_id] = (chain2num_max := chain2num_max + 1)
        chain_id_array = np.full([num_res], chain2num[c.chain_id], dtype=np.int64)

        if c.entity_id is None:
            entity_num = (ent2num_max := ent2num_max + 1)
        else:
            if c.entity_id not in ent2num:
                ent2num[c.entity_id] = (ent2num_max := ent2num_max + 1)
            entity_num = ent2num[c.entity_id]
        entity_id_array = np.full([num_res], entity_num, dtype=np.int64)

        sym_id_array = np.full([num_res], i, dtype=np.int64)

        multimer_arrays.append(
            {
                "chain_id": chain_id_array,
                "entity_id": entity_id_array,
                "sym_id": sym_id_array,
            }
        )

        chain_boundaries.append((total_index, total_index + num_res))
        total_index += num_res + 1

    sep = np.array([-1])
    update = {
        name: join_arrays([dct[name] for dct in multimer_arrays], sep=sep)
        for name in ["chain_id", "entity_id", "sym_id"]
    }
    array_args.update(update)

    metadata = ProteinComplexMetadata(
        chain_boundaries=chain_boundaries,
        chain_lookup={v: k for k, v in chain2num.items()},
        entity_lookup={v: k for k, v in ent2num.items()},
    )

    return cls(
        id=chains[0].id,
        sequence=esm3_c.CHAIN_BREAK_STR.join(chain.sequence for chain in chains),
        metadata=metadata,
        **array_args,
    )
```

### 功能说明：

- **作用**：根据一系列 `ProteinChain` 对象创建一个 `ProteinComplex` 对象。
- **实现步骤**：
  1. 检查链列表是否为空，若为空则抛出错误。
  2. 定义 `join_arrays` 函数，用于将多个数组连接起来，并在每个数组之间插入分隔符。
  3. 定义 `sep_tokens`，用于在不同链之间插入的分隔符数组。
  4. 遍历每条链，将其各个属性（如 `residue_index`, `insertion_code`, `atom37_positions` 等）连接起来，并在链之间插入分隔符。
  5. 构建 `multimer_arrays`，用于存储每条链的 `chain_id`, `entity_id`, `sym_id`。
  6. 为每条链分配唯一的 `chain_id` 和 `entity_id`，并记录链的边界位置。
  7. 更新 `array_args`，包括链ID、实体ID和对称性ID。
  8. 构建 `ProteinComplexMetadata`，包含链边界、链查找表和实体查找表。
  9. 使用 `esm3_c.CHAIN_BREAK_STR` 连接各条链的序列，形成复合物的整体序列。
  10. 创建并返回一个新的 `ProteinComplex` 对象，包含所有组合后的属性和元数据。

### 4.13 `infer_oxygen` 方法

```python
def infer_oxygen(self) -> ProteinComplex:
    """Oxygen position is fixed given N, CA, C atoms. Infer it if not provided."""
    O_vector = torch.tensor([0.6240, -1.0613, 0.0103], dtype=torch.float32)
    N, CA, C = torch.from_numpy(self.atoms[["N", "CA", "C"]]).float().unbind(dim=1)
    N = torch.roll(N, -3)
    N[..., -1, :] = torch.nan

    # Get the frame defined by the CA-C-N atom
    frames = Affine3D.from_graham_schmidt(CA, C, N)
    O = frames.apply(O_vector)
    atom37_positions = self.atom37_positions.copy()
    atom37_mask = self.atom37_mask.copy()

    atom37_positions[:, residue_constants.atom_order["O"]] = O.numpy()
    atom37_mask[:, residue_constants.atom_order["O"]] = ~np.isnan(
        atom37_positions[:, residue_constants.atom_order["O"]]
    ).any(-1)
    new_chain = replace(
        self, atom37_positions=atom37_positions, atom37_mask=atom37_mask
    )
    return new_chain
```

### 功能说明：

- **作用**：根据已知的N、CA、C原子位置推断氧原子（O）的坐标位置，如果氧原子的位置未提供。
- **实现步骤**：
  1. 定义一个固定的氧向量 `O_vector`。
  2. 提取复合物中所有N、CA、C原子的坐标，并转换为 `torch` 张量。
  3. 使用格雷厄姆-施密特正交化方法（`Affine3D.from_graham_schmidt`）基于CA、C、N原子构建局部坐标系。
  4. 应用仿射变换，将氧向量转换到全局坐标系，得到氧原子的坐标。
  5. 将推断出的氧原子坐标更新到 `atom37_positions` 中，并更新 `atom37_mask` 以反映氧原子的存在。
  6. 使用 `dataclasses.replace` 创建一个新的 `ProteinComplex` 对象，包含更新后的氧原子位置和掩码。
  7. 返回新的 `ProteinComplex` 对象。

### 4.14 `concat` 类方法

```python
@classmethod
def concat(cls, objs: list[ProteinComplex]) -> ProteinComplex:
    pdb_ids = [obj.id for obj in objs]
    if len(set(pdb_ids)) > 1:
        raise RuntimeError(
            "Concatention of protein complexes across different PDB ids is unsupported"
        )
    return ProteinComplex.from_chains(
        list(itertools.chain.from_iterable(obj.chain_iter() for obj in objs))
    )
```

### 功能说明：

- **作用**：将多个 `ProteinComplex` 对象连接成一个新的 `ProteinComplex` 对象。
- **实现步骤**：
  1. 检查所有待连接的复合物是否具有相同的PDB ID，若不同则抛出错误。
  2. 使用 `itertools.chain.from_iterable` 将所有复合物中的链迭代器展开，生成一个链的列表。
  3. 调用 `from_chains` 方法，将所有链组合成一个新的 `ProteinComplex` 对象。

### 4.15 `_sanity_check_complexes_are_comparable` 方法

```python
def _sanity_check_complexes_are_comparable(self, other: ProteinComplex):
    assert len(self) == len(other), "Protein complexes must have the same length"
    assert len(list(self.chain_iter())) == len(
        list(other.chain_iter())
    ), "Protein complexes must have the same number of chains"
```

### 功能说明：

- **作用**：检查两个 `ProteinComplex` 对象是否具有相同的长度和链数量，以确保它们可以进行比较或对齐。
- **实现步骤**：
  1. 检查两个复合物的序列长度是否相同。
  2. 检查两个复合物的链数量是否相同。

### 4.16 度量计算方法 `lddt_ca` 和 `gdt_ts`

```python
def lddt_ca(
    self,
    target: ProteinComplex,
    mobile_inds: list[int] | np.ndarray | None = None,
    target_inds: list[int] | np.ndarray | None = None,
    compute_chain_assignment: bool = True,
    **kwargs,
) -> float | np.ndarray:
    """Compute the LDDT between this protein complex and another.

    Arguments:
        target (ProteinComplex): The other protein complex to compare to.
        mobile_inds (list[int], np.ndarray, optional): The indices of the mobile atoms to align. These are NOT residue indices
        target_inds (list[int], np.ndarray, optional): The indices of the target atoms to align. These are NOT residue indices

    Returns:
        float | np.ndarray: The LDDT score between the two protein chains, either
            a single float or per-residue LDDT scores if `per_residue` is True.
    """
    if compute_chain_assignment:
        aligned = self.dockq(target).aligned
    else:
        aligned = self
    lddt = compute_lddt_ca(
        torch.tensor(aligned.atom37_positions[mobile_inds]).unsqueeze(0),
        torch.tensor(target.atom37_positions[target_inds]).unsqueeze(0),
        torch.tensor(aligned.atom37_mask[mobile_inds]).unsqueeze(0),
        **kwargs,
    )
    return float(lddt) if lddt.numel() == 1 else lddt.numpy().flatten()

def gdt_ts(
    self,
    target: ProteinComplex,
    mobile_inds: list[int] | np.ndarray | None = None,
    target_inds: list[int] | np.ndarray | None = None,
    compute_chain_assignment: bool = True,
    **kwargs,
) -> float | np.ndarray:
    """Compute the GDT_TS between this protein complex and another.

    Arguments:
        target (ProteinComplex): The other protein complex to compare to.
        mobile_inds (list[int], np.ndarray, optional): The indices of the mobile atoms to align. These are NOT residue indices
        target_inds (list[int], np.ndarray, optional): The indices of the target atoms to align. These are NOT residue indices

    Returns:
        float: The GDT_TS score between the two protein chains.
    """
    if compute_chain_assignment:
        aligned = self.dockq(target).aligned
    else:
        aligned = self
    gdt_ts = compute_gdt_ts(
        mobile=torch.tensor(
            index_by_atom_name(aligned.atom37_positions[mobile_inds], "CA"),
            dtype=torch.float32,
        ).unsqueeze(0),
        target=torch.tensor(
            index_by_atom_name(target.atom37_positions[target_inds], "CA"),
            dtype=torch.float32,
        ).unsqueeze(0),
        atom_exists_mask=torch.tensor(
            index_by_atom_name(aligned.atom37_mask[mobile_inds], "CA", dim=-1)
            & index_by_atom_name(target.atom37_mask[target_inds], "CA", dim=-1)
        ).unsqueeze(0),
        **kwargs,
    )
    return float(gdt_ts) if gdt_ts.numel() == 1 else gdt_ts.numpy().flatten()
```

### 功能说明：

- **lddt_ca**：
  - **作用**：计算当前蛋白质复合物与目标复合物之间的 LDDT（Local Distance Difference Test）评分，基于 Cα 原子。
  - **参数**：
    - `target`：目标 `ProteinComplex` 对象。
    - `mobile_inds` 和 `target_inds`：用于对齐的移动和目标原子索引。
    - `compute_chain_assignment`：是否进行链分配，默认为 `True`。
  - **实现步骤**：
    1. 如果需要链分配，调用 `dockq` 方法对齐两个复合物，获取对齐后的复合物。
    2. 使用 `compute_lddt_ca` 函数计算 LDDT 分数。
    3. 返回单一的 LDDT 分数或每残基的分数数组。

- **gdt_ts**：
  - **作用**：计算当前蛋白质复合物与目标复合物之间的 GDT_TS（Global Distance Test - Total Score）评分，基于 Cα 原子。
  - **参数**：
    - 同 `lddt_ca` 方法。
  - **实现步骤**：
    1. 如果需要链分配，调用 `dockq` 方法对齐两个复合物，获取对齐后的复合物。
    2. 提取对齐后和目标复合物的 Cα 原子坐标。
    3. 计算 Cα 原子的存在掩码。
    4. 使用 `compute_gdt_ts` 函数计算 GDT_TS 分数。
    5. 返回单一的 GDT_TS 分数或每残基的分数数组。

### 4.17 `dockq` 方法

```python
def dockq(self, native: ProteinComplex):
    # This function uses dockqv2 to compute the DockQ score. Because it does a mapping
    # over all possible chains, it's quite slow. Be careful not to use this in an inference loop
    # or something that requires fast scoring. It defaults to 8 CPUs.

    try:
        pass
    except BaseException:
        raise RuntimeError(
            "DockQ is not installed. Please update your environment."
        )
    self._sanity_check_complexes_are_comparable(native)

    def sanity_check_chain_ids(pc: ProteinComplex):
        ids = []
        for i, chain in enumerate(pc.chain_iter()):
            if i > len(SINGLE_LETTER_CHAIN_IDS):
                raise ValueError("Too many chains to write to PDB file")
            if len(chain.chain_id) > 1:
                raise ValueError(
                    "We only supports single letter chain IDs for DockQ"
                )
            ids.append(chain.chain_id)
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate chain IDs in protein complex: {ids}")
        return ids

    sanity_check_chain_ids(self)
    sanity_check_chain_ids(native)

    with TemporaryDirectory() as tdir:
        dir = Path(tdir)
        self.to_pdb(dir / "self.pdb")
        native.to_pdb(dir / "native.pdb")

        output = check_output(["DockQ", dir / "self.pdb", dir / "native.pdb"])
    lines = output.decode().split("\n")

    # Remove the header comments
    start_index = next(
        i for i, line in enumerate(lines) if line.startswith("Model")
    )
    lines = lines[start_index:]

    result = {}
    interfaces = []
    current_interface: dict = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Model  :"):
            pass  # Tmp pdb file location, it's useless...
        elif line.startswith("Native :"):
            pass  # Tmp pdb file location, it's useless...
        elif line.startswith("Total DockQ"):
            total_dockq_match = re.search(
                r"Total DockQ over (\d+) native interfaces: ([\d.]+) with (.*) model:native mapping",
                line,
            )
            if total_dockq_match:
                result["value"] = float(total_dockq_match.group(2))
                result["native interfaces"] = int(total_dockq_match.group(1))
                native_chains, self_chains = total_dockq_match.group(3).split(":")
                result["mapping"] = dict(zip(native_chains, self_chains))
            else:
                raise RuntimeError(
                    "Failed to parse DockQ output, maybe your DockQ version is wrong?"
                )
        elif line.startswith("Native chains:"):
            if current_interface:
                interfaces.append(current_interface)
            current_interface = {
                "Native chains": line.split(":")[1].strip().split(", ")
            }
        elif line.startswith("Model chains:"):
            current_interface["Model chains"] = (
                line.split(":")[1].strip().split(", ")
            )
        elif ":" in line:
            key, value = line.split(":", 1)
            current_interface[key.strip()] = float(value.strip())

    if current_interface:
        interfaces.append(current_interface)

    def parse_dict(d: dict[str, Any]) -> DockQSingleScore:
        return DockQSingleScore(
            native_chains=tuple(d["Native chains"]),  # type: ignore
            DockQ=float(d["DockQ"]),
            interface_rms=float(d["irms"]),
            ligand_rms=float(d["Lrms"]),  # Note the capitalization difference
            fnat=float(d["fnat"]),
            fnonnat=float(d["fnonnat"]),
            clashes=float(d["clashes"]),
            F1=float(d["F1"]),
            DockQ_F1=float(d["DockQ_F1"]),
        )

    inv_mapping = {v: k for k, v in result["mapping"].items()}

    self_chain_map = {c.chain_id: c for c in self.chain_iter()}
    realigned = []
    for chain in native.chain_iter():
        realigned.append(self_chain_map[inv_mapping[chain.chain_id]])

    realigned = ProteinComplex.from_chains(realigned)
    aligner = Aligner(realigned, native)
    realigned = aligner.apply(realigned)

    result = DockQResult(
        total_dockq=result["value"],
        native_interfaces=result["native interfaces"],
        chain_mapping=result["mapping"],
        interfaces={
            (i["Model chains"][0], i["Model chains"][1]): parse_dict(i)
            for i in interfaces
        },
        aligned=realigned,
        aligned_rmsd=aligner.rmsd,
    )

    return result
```

### 功能说明：

- **作用**：计算当前 `ProteinComplex` 与原生复合物 (`native`) 之间的 DockQ 评分，评估复合物对接的质量。
- **实现步骤**：
  1. **安装检查**：尝试调用 `DockQ`，若未安装则抛出错误提示。
  2. **一致性检查**：调用 `_sanity_check_complexes_are_comparable` 方法，确保两个复合物长度和链数量一致。
  3. **链ID验证**：确保所有链ID都是单字母且唯一，符合 DockQ 的要求。
  4. **临时目录操作**：
     - 创建临时目录，将当前复合物和原生复合物分别写入 `self.pdb` 和 `native.pdb` 文件。
     - 使用子进程调用 `DockQ` 工具，计算 DockQ 评分。
  5. **输出解析**：
     - 解析 DockQ 工具的输出，提取总 DockQ 值、原生接口数量、链映射关系等信息。
     - 解析各个接口的详细评分（如 `DockQ`, `interface_rms`, `fnat` 等）。
  6. **链对齐**：
     - 根据 DockQ 的链映射关系，对齐当前复合物的链顺序，使其与原生复合物对应。
     - 使用 `Aligner` 类对齐复合物，并计算对齐后的 RMSD 值。
  7. **结果封装**：创建并返回一个 `DockQResult` 对象，包含总 DockQ 值、接口数量、链映射、各接口评分详情、对齐后的复合物以及对齐的 RMSD。

### 注意事项：

- **性能问题**：由于 DockQ 需要遍历所有可能的链映射关系，计算可能较慢，因此建议避免在需要高性能的推理循环中频繁调用。
- **依赖性**：代码依赖于外部 DockQ 工具的安装和正确配置，若未安装或版本不匹配，可能导致错误。

## 总结

`protein_complex.py` 模块提供了一个全面的框架，用于表示和操作蛋白质复合物，包括从PDB文件加载、序列和原子数据管理、链的处理、对接评分计算（如 DockQ、LDDT、GDT_TS）等功能。通过使用数据类和高效的数值计算库，模块能够高效地处理大型蛋白质复合物数据，并与外部工具（如 DockQ）集成，实现复杂的结构比较和评分任务。

该模块的设计考虑了数据的一致性和完整性，通过元数据管理和多种检查机制，确保在操作蛋白质复合物时的可靠性。同时，提供了序列化和压缩的方法，便于存储和传输大规模的蛋白质复合物数据。

在实际应用中，该模块可用于蛋白质结构预测、对接研究、结构比较分析等领域，是生物信息学和结构生物学研究中的一个有力工具。

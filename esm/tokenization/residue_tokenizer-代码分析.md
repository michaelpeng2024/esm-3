## residue_tokenizer-代码分析
这段代码定义了一个名为 **ResidueAnnotationsTokenizer** 的类，用于对蛋白质序列中的每个氨基酸残基标注进行特殊化的分词和编码处理。它实现了一种将残基级别的功能性或结构性注释（annotation）映射为可供模型（如 ESM）使用的整数字典索引（token IDs）的方法。以下是对其功能和流程的详细分析。

### 整体功能概述

ResidueAnnotationsTokenizer 的目标是从一个提供残基注释信息的 CSV 文件中加载注释标签，并为每条序列中的每个残基生成相应的标注 token。这样，每个残基就有一个或多个对应的注释标签，被转换为特定的 token 表示。如果某个残基没有注释，则使用 `<none>` 表示。同时，该分词器提供了将注释映射为 ID 的机制，以及将分词结果编码为张量（`torch.Tensor`）的功能，以方便后续送入神经网络模型中。

### 核心数据结构和属性

1. **CSV 文件加载及映射关系构建**  
   - `self.csv_path`: 指向注释信息存储的 CSV 文件路径。  
   - `_description2label`: 一个懒加载（`@cached_property`）的属性，从 CSV 中读取 `label` 和 `label_clean` 列，并构建一个字典，将 `label` 映射至 `label_clean`。  
   - `_labels`: 从 CSV 中对 `label_clean` 聚合统计后排序，得到最终的注释标签列表。  
   - `_label2id`: 根据 `_labels` 列表，为每个 `label_clean` 分配一个整数 ID，并且这些 ID 会偏移一定量（因为有特殊 token）。  
   - `special_tokens`: 定义了特殊标记列表，包括 `"<pad>"`, `"<motif>"`, `"<unk>"`。这些特殊标记在 token ID 排序中优先。  
   - `vocab`: 完整的词表，即 `special_tokens + ["<none>"] + annotation_tokens`，其中 `annotation_tokens` 是对每个标签 ID 添加 `<ra:...>` 前缀构成的特殊格式。  
   - `vocab_to_index`: 将 vocab 中的每个 token 对应到整数 ID 的字典。  
   - `vocabulary`: 人类可读的标签集合（包含 special_tokens, `<none>` 和原始的 label 列表）。  

2. **重要的特殊 token 与属性**  
   类中重写了一些属性来满足特定框架（可能是继承自 EsmTokenizerBase 的契约），如 `mask_token`, `bos_token`, `eos_token`, `pad_token` 均为 `"<pad>"`，这意味着在这个分词器中许多特殊情况下都是用 `<pad>` 来表示序列开头、结尾、掩码以及链断开（chain break）。

3. **max_annotations**: 指定单个位点（残基）最多可以保留多少个注释 ID。如果某个残基有超过这个数目的注释标签，则会截断到该数量上限。

### 核心方法和逻辑流程

1. **注释映射及检查**  
   `def _description2id(self, description: str) -> int | None`:  
   给定描述（`description`），从 `_description2label` 找出对应的 `label_clean`，再从 `_label2id` 找出相应的 ID。如果没找到则返回 `None`。

2. **tokenize**(sample, sequence, fail_on_mismatch=False)  
   此方法将一条序列及其注释信息转换为 token 列表（字符串列表）。  
   - 输入参数：
     - `sample`: 包含注释信息的字典，其中可能包含字段：
       - `interpro_site_descriptions`
       - `interpro_site_starts`
       - `interpro_site_ends`
       - `interpro_site_residues`
     - `sequence`: 原始蛋白质序列字符串
     - `fail_on_mismatch`: 当注释的残基与原序列不匹配时是否抛出异常，默认为否。

   - 处理逻辑：
     1. 如果 `sample` 为 `None`，或相关注释字段缺失，则整个序列使用 `<pad>` 进行标记，表示无可用注释。
     2. 验证 `sample` 中各注释列表长度一致。
     3. 对于 `sample` 中的每个注释片段：  
        - 从 `description` 映射到 `token_id`（若不存在则使用 `<unk>`）。  
        - 按 `start` 和 `end`（1-based索引）在 `sequence` 中定位对应区间，并检查实际氨基酸是否与 `interpro_site_residues` 中给出的残基匹配。  
          - 若不匹配且 `fail_on_mismatch` 为 True，抛出异常；否则整条序列标记为 `<pad>`。  
        - 若检查通过，将该位置的 `token_id` 添加到相应残基位置的集合中（一个残基可能有多个注释）。
     4. 最终，每个残基的注释集合会转换为 `<ra:id1,id2,...>` 格式的 token。如果没有注释则用 `<none>`。
   
   输出：长度与 `sequence` 一致的 token 列表。

3. **encode**(tokens, add_special_tokens=True)  
   将上一步产生的 token 列表转换为张量表示。  
   - 创建 `(len(tokens), max_annotations)` 的张量，并用 `<pad>` 的 ID 初始化。
   - 对每个 token 调用 `_token2ids` 转换成 ID 列表（若是 `<ra:...>` 则可能有多个 ID，若是 `<none>` 等则只有一个 ID）。
   - 将这些 ID 填入对应的张量行中。  
   - 如果 `add_special_tokens` 为 True，会在序列首尾各填入一行 `<pad>` 行，用于特定的模型输入格式要求（相当于在时间维度前后 pad 一下）。
   
   最终返回一个 `torch.Tensor`，形状大约为 `(序列长度+2, max_annotations)`（如果加特殊 tokens）。

4. **_token2ids(token)**  
   用于辅助 `encode` 方法。解析 `<ra:...>` 格式的 token，如果是 `<ra:id1,id2,...>` 则返回 `[id1, id2, ...]`，否则返回单一 ID 的列表 `[id]`。

5. **decode**  
   `decode` 方法尚未实现，并给出了说明：对于残基注释的解码，不在此类中进行，需要在其他工具函数中完成。

### 特殊 Token 属性

- `mask_token`, `bos_token`, `eos_token`, `pad_token`, `chain_break_token` 全部指向 `<pad>`。
- `mask_token_id`、`bos_token_id`、`eos_token_id`、`pad_token_id`、`chain_break_token_id` 这些属性都返回 `<pad>` 对应的 ID。
- `all_token_ids` 返回整个词表范围内的 ID 列表。
- `special_token_ids` 返回 special_tokens 对应的 ID。

### 代码应用场景

此代码可能应用于蛋白质序列建模中，在对蛋白进行功能预测或注释相关的任务时，除了主链氨基酸序列外，还需要对每个残基携带特定的生物学注释（如活性位点、金属结合位点等）。该分词器将这些注释从 CSV 文件中载入，并在序列的 tokenization 过程中将注释转换为特定 token，方便模型将这些信息作为输入特征。

通过这种方式，一个输入的蛋白质序列在经过此分词器处理后，不仅仅有单一的氨基酸 token，还能对每个残基关联上若干注释标签的 token ID，这有利于下游模型充分利用这些语义标注信息。

### 小结

综上，此代码实现了一个自定义的分词器类 **ResidueAnnotationsTokenizer**。其主要功能为：

1. 从提供注释信息的 CSV 文件构建注释词典和映射。
2. 根据给定的序列和注释（开始、结束、描述、残基），对每个残基附加注释 token。
3. 将带注释的分词结果转换成用于模型输入的张量表示形式。

该实现使得对蛋白质残基级的功能位点注释信息能以一种标准的 token 化形式整合到模型的输入管线中。

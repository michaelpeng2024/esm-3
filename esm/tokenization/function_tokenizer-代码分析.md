## function_tokenizer-代码分析
**概览**

这段代码定义了一个针对蛋白功能标注文本信息的分词器类 `InterProQuantizedTokenizer`。该分词器的核心目标是将蛋白质序列每个位置（残基）上的功能注释（如 InterPro 功能域 ID、关键词）转换为固定长度和大小的离散数字表示（token），以便后续的深度学习模型（如 ESM）能够对这些功能信息进行编码和利用。

下面将对代码的主要功能、模块设计及数据处理流程进行详细分析。

---

**整体功能**

1. **功能注释到分词表示的转换**：  
   给定蛋白质序列的功能注释，这些注释可能以区间形式（如：从序列第 X 位到第 Y 位有某个 InterPro 功能域）出现。分词器会将这些区间注释展开成逐位注释集合，然后对每个位点上的功能信息进行特征提取和分词化。

2. **关键词与 InterPro ID 处理**：  
   功能注释包括两类：  
   - InterPro ID（如 IPRXXXX）  
   - 关键词（特定的功能描述词汇）  
   分词器会将这些信息转化为统一的数值表示。

3. **TF-IDF 表示与 LSH 哈希**：  
   对关键词集合使用事先构建好的 TF-IDF 模型进行向量化，然后通过 Locality Sensitive Hashing（LSH）将高维TF-IDF向量映射到固定长度的哈希编码，从而得到一组定长的 token 序列。  
   简言之，LSH 将一组关键词对应的特征向量压缩为若干 token（如 8个bit 表示一个token ID），从而构建一个有限大小的词表来表征功能信息。

4. **特殊标记和控制标记**：  
   分词器定义了一些特殊 token，如 `<pad>`、`<unk>`、`<none>` 等，用于表示无信息、未知标注或无注释的位置。这些特殊 token 有固定的词表索引。

---

**关键属性与组件**

- `depth`: 指定每个位置最终使用多少个token来表示功能信息。由于 LSH 的输出固定为深度 depth，最终一个位置对应 depth 个 token。

- `lsh_bits_per_token`: 每个 LSH token 的bit数，这决定了 LSH 映射后的单 token 空间的大小（`lsh_vocab_size = 2^(lsh_bits_per_token)`）。

- `tfidf_model (_tfidf)`: 使用关键词词汇表与 IDF 文件构建的 TF-IDF 模型，对关键词集合进行向量化。

- `lsh (_lsh)`: 基于预先构建的 LSH 超平面信息，对给定的 TF-IDF 向量执行 LSH，将高维向量投影到低维的二进制特征表示（从而得到 tokens）。

- `interpro2keywords`: 从 CSV 文件中加载，建立 InterPro ID 到对应关键词集合的映射。

- `interpro_`: InterPro工具类（`interpro.InterPro`），通过提供的 CSV 文件来获取 InterPro 条目信息（如名称等）。

- `vocab`: 最终的词表，由特殊 token（`<pad>`、`<motif>`、`<unk>`）、`<none>`以及通过 LSH 生成的 `<lsh:xxx>` token 构成。

---

**处理流程解析**

1. **初始化 ( `__init__` )**  
   - 初始化各项路径及参数。  
   - 加载关键词词汇表和对应的 IDF 文件，构建 TF-IDF 模型 `_tfidf`。  
   - 加载 InterPro entry 列表和 InterPro 到关键词的映射文件 `_interpro2keywords_path`。  
   - 确定 LSH 的相关文件路径，根据 `lsh_bits_per_token` 和 `depth` 初始化 LSH 对象 `_lsh`。

2. **`tokenize` 方法**  
   参数：  
   - `annotations`: List[FunctionAnnotation]，每个元素包含 `(start, end, label)` 描述功能标注的区间和标签。  
   - `seqlen`: 序列长度，即需要输出的 token 列表长度。  
   - `p_keyword_dropout`: 随机丢弃关键词的概率，用于数据增广。

   主要步骤：  
   - 初始化一个长度为 seqlen 的列表，每个元素为一个空 set，用于存放该位置的所有功能标签。  
   - 根据输入 annotations，将区间注释展平到每个位置上。  
   - 针对每个位点的标签集合（可能包含 InterPro ID 和关键词），通过 `_function_text_hash` 转换为一组 depth 个 LSH token。如果该位置没有标签则为 `<none>`，如果无法哈希（如没有有效关键词）则使用 `<unk>`。  
   - 最终生成一个长度为 seqlen 的 token 列表，每个元素是 `<none>` 或 `<lsh:x,y,z,...>` 形式的 token。

3. **`_function_text_hash` 方法**  
   参数：  
   - `labels`: 该位置的功能标签集合  
   - `keyword_mask`: 一个布尔掩码，用于随机丢弃关键词

   逻辑：  
   - 将输入标签分离成两类：InterPro IDs 和 关键词。  
   - 使用 `_tfidf.encode` 将关键词和对应的 InterPro 映射关键词转换为 TF-IDF 向量，并对多个关键词/ID 的结果取 element-wise maximum（这样可以突出高频关键词，避免稀释）。  
   - 若 `keyword_mask` 存在，则对向量中被mask的关键词降权处理。  
   - 若最终 TF-IDF 向量为空或 sum为0，则返回 None。否则调用 `_lsh(vec)` 将 TF-IDF 向量通过 LSH 映射得到一个 depth × bits_per_token 维的二值编码，进而转换为 token ID 集合（此处 depth 指输出的 token 个数）。  
   - 返回该位置对应的 LSH token indices。

4. **`encode` 和 `batch_encode` 方法**  
   - `encode`: 将 token 列表（如 `<none>`、`<lsh:1,2,...>`）转换为对应的整数 ID 张量。  
   - 通过 `_token2ids` 将单个 token 字符串映射到整型 ID。对于 `<none>` 或特殊 token，每个位置使用同一个 token_id 重复 depth 次。对于 `<lsh:x,y,z,...>` 格式的 token，将其中的数字映射到适当偏移的词表索引。  
   - 可选地在序列两端添加 `<pad>` 作为特殊起始和结束标记（类似 `<cls>` 和 `<eos>`）。  
   - `batch_encode` 则是对一批序列调用 `encode`，并用 `stack_variable_length_tensors` 对其进行对齐与填充。

5. **其他辅助函数和属性**  
   - `lookup_annotation_name`: 根据 InterPro ID 查找对应的功能名称。  
   - `format_annotation`: 格式化输出注释（显示名称和ID）。  
   - `special_tokens` / `vocab` / `vocab_to_index`: 定义和管理词表与 ID 映射。  
   -  `mask_token`, `bos_token`, `eos_token`, `pad_token` 等属性返回相应的特殊 token 及其 ID。

6. **文本关键词处理相关函数**  
   `_keywords_from_text`, `_sanitize`, `_texts_to_keywords`:  
   - 用于将自由文本（如 InterPro/GO 名称）分解为 unigram 和 bigram 的关键词集合，同时去除标点和无意义停用词。  
   - 这些辅助函数在构建词汇表或处理 InterPro 映射时有帮助。

---

**总结**

该代码的核心功能是将功能注释文本（InterPro和关键词）映射到一系列与序列长度对齐的多维 token 表示，从而让下游模型（如 ESM模型）能够在处理蛋白序列的同时，将位点上的功能特征整合进来。这是一个从生物学注释文本信息到模型可用的离散 token 特征的管道，包括了 TF-IDF 向量化、LSH 映射和最终的 token 编码等过程。

整体而言，这个 tokenizer 为生物蛋白功能注释信息提供了一个从文本到定长离散特征编码的通用框架，用于深度学习模型的输入准备。

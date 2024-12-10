## generation-代码分析
**整体概览：**  
这段代码来自一个名为`generation.py`的文件，用于对蛋白质序列和相关特性进行基于ESM（Evolutionary Scale Modeling）模型的迭代采样和生成。ESM是Meta（Facebook AI Research）提出的用于蛋白质序列建模的深度学习模型。通过本代码，可以对输入的蛋白质数据进行推理（forward pass）、对指定轨道（track，如`sequence`、`function`、`structure`、`residue_annotations`、`sasa`等）进行掩码填充与采样，从而在多步骤中逐渐生成完整的输出。

在更高层次，本代码实现了以下功能：  
1. **迭代采样框架**：对给定的初始蛋白质序列（或相关特征）以及相应的生成配置，多次迭代执行forward和sampling步骤，逐步替换掩码（mask）位置的token，直到生成期望的结果。  
2. **支持多个轨道（tracks）的生成**：包括蛋白质主序列、结构信息、二级结构、SASA（可溶性表面积）、功能注释和残基注释。不同的轨道使用不同的tokenizer和采样策略。  
3. **适应多种采样策略**：如基于熵排序的token选择（entropy-based）、随机选择（random）、top-p采样、invalid_ids过滤、温度控制以及逐步降低温度（温度退火）。  
4. **批处理与数据组织**：对输入的多个蛋白质样本进行打包（batching）、对齐（padding）以及在不同的生成步骤中依次处理。  
5. **错误处理与兼容性**：对于不支持的操作如对`coordinates`或`residue_annotations`的迭代采样会产生错误处理逻辑。同时对于部分未定义场景有保护机制。

下面从代码流程与主要函数入手进行详细分解。

---

**关键数据结构与概念**：  
- **ESMProtein / ESMProteinTensor**：表示蛋白质序列及其他特性（如结构、功能标注）的数据结构。其中`ESMProteinTensor`是张量形式的编码表示，用于直接输入模型。  
- **TokenizerCollectionProtocol与EsmTokenizerBase**：对不同轨道数据进行Token化和反Token化的工具。不同轨道可能需要不同的tokenizer（如序列、功能、结构都有各自的token编码方式）。  
- **GenerationConfig**：控制生成过程的配置参数，包括采样的步骤数、策略（随机/熵）、调度方案（schedule）、top_p、温度、无效token过滤以及是否仅使用坐标进行条件生成等。  
- **ForwardAndSampleOutput & LogitsOutput**：forward pass得到的logits和相关中间结果的封装，以及在采样后得到的最终结果（包括概率、熵等统计信息）。

---

**代码主要函数及流程**：

1. **`iterative_sampling_raw`函数**：  
   - 功能：输入一批ESMProtein对象和对应的GenerationConfig，通过`client.encode`编码，再调用`client.batch_generate`生成对应的输出tokens，最后`client.decode`得到新的ESMProtein对象。  
   - 它比较基础，就是一次性利用client的批处理生成接口，从头生成序列（可能非迭代），然后再还原到ESMProtein对象。

2. **`iterative_sampling_tokens`函数**（重点）：  
   这个函数是实现**迭代生成**的重要入口。其流程为：  
   - 输入：`ESM3InferenceClient`、`input_tokens`（已经编码好的ESMProteinTensor列表）、`configs`（对应每个样本的生成配置）、`tokenizers`。
   - 首先对输入进行一些处理，如根据配置将不需要的轨道信息清空（如`config.condition_on_coordinates_only`时只保留坐标条件）。
   - 计算每个蛋白质序列的长度，统计需要采样的token数量，以及对给定的`num_steps`与实际mask数量进行匹配处理。
   - 将所有输入的`ESMProteinTensor`打包成`_BatchedESMProteinTensor`（通过`_stack_protein_tensors`实现），方便模型一次forward处理。
   - 进入迭代循环（最多`max_num_steps`次）：  
     a. 调用`_batch_forward`对当前的`batched_tokens`进行前向计算（获取logits）。  
     b. 对于批中的每个样本，使用`_sample_per_prompt`函数对其特定track的logits进行采样。  
     c. 根据熵或随机策略选择本步要替换的token位置（通过`_get_iterative_sampling_mask_for_prompt_and_step`确定本步具体需要填充的mask位置）。  
     d. 更新`batched_tokens`中相应轨道的token为新采样的token。  
   - 全部迭代完成后，将批数据拆分成单独的输出。如果在过程中出现不支持的场景（如对`coordinates`轨道的迭代采样），会生成`ESMProteinError`对象。  
   - 在最后还原非采样轨道的数据，并返回最终的`ESMProteinTensor`或`ESMProteinError`列表。

3. **`_batch_forward`函数**：  
   - 调用`client.logits`对批数据进行模型前向计算，获取各轨道的logits以及中间结果（如embedding）。

4. **`_sample_per_prompt`函数**：  
   - 对给定的`protein`（单个样本）、`logits_output`、`sampling_config`等进行具体采样操作。  
   - 不同轨道根据`sampling_config`和`tokenizer`的特性采用不同的采样函数：  
     - `sequence`、`structure`、`secondary_structure`、`sasa`轨道使用`sample_logits`和相应的mask处理。  
     - `function`轨道使用`sample_function_logits`进行更加复杂的多维采样。  
   - 在采样完成后，对采样结果计算熵、概率、top-k logprobs等统计信息，以便后续分析。

5. **`_sample_track`与`_sample_function_track`函数**：  
   - `_sample_track`：对`sequence`、`structure`、`secondary_structure`及类似的一维轨道进行采样。  
   - `_sample_function_track`：对`function`轨道的多维（L×D）分布进行采样。  
   - 这些函数都会返回`sampled_tokens`和一系列统计数据（如entropy、prob、topk_logprob等）。

6. **`_compute_track_metadata`函数**：  
   - 在完成token采样后，通过logits和采样结果计算熵、样本对应位置的概率、top-k的logprob，以及最终形成一个字典用于后续组装。
   
7. **各种辅助函数**：  
   - `_trim_sequence_tensor_dataclass`与`_slice_tensor_dataclass`：对attr定义的数据类进行张量切片、长度修剪的辅助函数。  
   - `_stack_protein_tensors`：将多个`ESMProteinTensor`对象合并成一个带有batch维度的张量集合，并进行padding。  
   - `_get_masked_positions`：确定哪些位置是mask，需要被采样填充。  
   - `_get_iterative_sampling_mask_for_prompt_and_step`：根据当前步骤、策略（random或entropy）、已经采样的结果、以及衰减调度（decoding_schedule）来决定本步应填充/替换的token位置。  
   - `_get_annealed_temperature`：在多步采样中，对温度进行退火，逐渐降低温度。

---

**总结**：  
这段代码的核心功能是对蛋白质序列及其相关特性进行“逐步迭代”的生成和填充。通过不断调用模型的forward来获取预测分布，然后根据各种策略对mask位置进行token采样，从而在多个迭代步骤中将初始的部分或全部mask逐渐替换为具体的有效token。这是一个复杂的多步骤生成过程，可以适用于蛋白质设计、功能预测、结构补全等领域。

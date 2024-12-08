## 分类样本扩增的实现-Local
### 问题的提出
我想根据一段序列来生成更多相似样本，例如我要做序列多类别分类。但是样本不足，我应该如何根据ESM3来生成各个分类的分类样本？请详细给我给出方案和代码，谢谢。  
如果需要根据一段序列生成更多相似样本以解决样本不足问题，并将这些样本用于多类别分类，可以利用 ESM3 模型来生成合成样本。这种方法结合了蛋白质语言模型的生成能力和类别信息，以下是详细方案和代码实现：

---

### **方案**

#### **目标**：
根据现有的类别样本，生成与该类别相似的新样本，同时保持原有类别的特性，用于扩展数据集。

#### **实现步骤**：
1. **准备类别样本**：
   - 按类别组织原始样本，确保每个类别有若干已标注的序列。

2. **设计输入模板**：
   - 通过部分掩码或完整输入提示，提供上下文信息，让模型生成与原始样本相似的序列。

3. **调整生成参数**：
   - 使用生成参数（如 `temperature`）来控制生成的多样性。
   - 限制生成的步数和长度，以保持样本的生物学意义。

4. **生成样本**：
   - 利用 ESM3 模型逐类别生成样本，并为生成的样本分配类别标签。

5. **后处理与验证**：
   - 检查生成样本的质量，确保符合类别特征。
   - 去重和筛选无意义序列。

---

### **代码实现**

以下代码展示如何使用 ESM3 模型生成分类样本：

```python
import random
from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    GenerationConfig,
    SamplingConfig,
    SamplingTrackConfig,
)

# 初始化 ESM3 模型客户端
client = ESM3.from_pretrained("esm3_sm_open_v1")

# 定义类别样本
categories = {
    "Class1": ["DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYR", "GQGSEHTVDKKKYAAELHLVHWNTKYGD"],
    "Class2": ["MAESDPLHQFHFHWGSLDGQGSEHTVDLHFNTGDY", "KPGLQKVVDVLDSIKTKGKSADFTN"],
}

# 生成函数
def generate_similar_samples(category_samples, num_samples, temperature=0.7, num_steps=8):
    """
    根据已有类别样本生成相似样本
    :param category_samples: 现有类别样本列表
    :param num_samples: 生成样本数量
    :param temperature: 随机性参数
    :param num_steps: 每次生成的步数
    :return: 生成的相似样本列表
    """
    generated_samples = []
    for _ in range(num_samples):
        # 随机选择一个类别样本作为提示
        prompt = random.choice(category_samples)
        
        # 引入部分掩码 (随机掩盖部分序列)
        masked_prompt = list(prompt)
        for i in range(len(masked_prompt)):
            if random.random() < 0.2:  # 随机掩盖 20% 的位置
                masked_prompt[i] = "_"
        masked_prompt = "".join(masked_prompt)

        # 构造 ESMProtein 对象
        protein = ESMProtein(sequence=masked_prompt)
        
        # 生成样本
        protein = client.generate(
            protein,
            GenerationConfig(track="sequence", num_steps=num_steps, temperature=temperature),
        )
        generated_samples.append(protein.sequence)

    return generated_samples

# 生成每个类别的扩展样本
num_samples_per_class = 5  # 每个类别生成 5 个样本
expanded_data = {}

for category, samples in categories.items():
    print(f"Generating for {category}...")
    expanded_data[category] = generate_similar_samples(samples, num_samples_per_class)

# 打印生成结果
for category, samples in expanded_data.items():
    print(f"\nCategory: {category}")
    for seq in samples:
        print(seq)
```

---

### **代码说明**

#### **1. 样本准备**
- `categories` 是一个字典，存储了原始样本，按类别分组。
- 每个类别的样本被用作生成提示。

#### **2. 随机掩码处理**
- 随机掩盖输入序列的一部分（20% 的位置），增加生成的多样性。
- 掩码区域会引导模型生成更具变化的序列。

#### **3. 生成配置**
- **`num_steps`**：生成过程的步数，控制生成序列的长度。
- **`temperature`**：随机性参数，值越高生成的样本多样性越大。
- **`track="sequence"`**：指定生成的是序列而非结构或其他信息。

#### **4. 样本生成**
- 使用 `client.generate` 生成新的序列。
- 生成的序列会被添加到对应类别的扩展数据中。

#### **5. 多类别扩展**
- 对每个类别分别生成指定数量的样本。
- 生成结果按类别组织，最终存储在 `expanded_data` 字典中。

---

### **后续处理**

1. **去重与筛选**：
   - 对生成的样本进行去重，移除无意义或重复的序列。

2. **数据标注**：
   - 为每个生成的样本添加与原始类别对应的标签。

3. **质量评估**：
   - 使用生物学工具（如序列比对）检查生成序列的相似性和合理性。

4. **训练新模型**：
   - 将扩展后的数据集用于训练多类别分类模型。

---

### **总结**

上述方法利用 ESM3 模型逐类别生成新样本，通过调整掩码比例和随机性参数，确保生成的样本与原类别特性相似但又具有多样性。这样扩展的数据集可以有效提升多类别分类任务的模型性能。

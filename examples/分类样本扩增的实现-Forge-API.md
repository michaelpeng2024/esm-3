## 分类样本扩增的实现-Forge-API
以下是通过 **Forge API** 调用更大 ESM3 模型的实现，具体调整包括：
1. **使用 Forge API 的 `ESM3InferenceClient`** 来替换本地预训练模型。
2. **适配 Forge API 的方法和接口**，保持与生成逻辑一致。

---

### **修改后的代码**

```python
import random
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    GenerationConfig,
    SamplingConfig,
    SamplingTrackConfig,
)

# 初始化 Forge API 的 ESM3 客户端
client = ESM3InferenceClient(api_key="your_forge_api_key")

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

### **代码修改点解析**

#### **1. 使用 Forge API 客户端**
- 替换本地的预训练模型（`ESM3.from_pretrained`）为 **Forge API** 客户端。
- 初始化 `ESM3InferenceClient` 时需要提供 `api_key`，确保 API 调用的认证。

```python
client = ESM3InferenceClient(api_key="your_forge_api_key")
```

---

#### **2. Forge API 的 `generate` 方法**
Forge API 提供的 `generate` 方法与本地模型接口保持一致：
- 使用 `ESMProtein` 作为输入对象。
- 通过 `GenerationConfig` 配置生成任务的参数，包括：
  - **`track="sequence"`**：指定生成目标为序列。
  - **`num_steps`**：设置生成的步数，控制生成长度。
  - **`temperature`**：调整生成随机性，控制样本多样性。

```python
protein = client.generate(
    protein,
    GenerationConfig(track="sequence", num_steps=num_steps, temperature=temperature),
)
```

---

#### **3. 其他逻辑保持一致**
- 引入随机掩码的逻辑未改变，仍通过 `masked_prompt` 引导模型生成相似样本。
- 输出的生成样本直接存储在 `expanded_data` 中。

---

### **需要注意的事项**

1. **API Key**：
   - 替换 `your_forge_api_key` 为实际的 Forge API 密钥。
   - 确保密钥具有调用 ESM3 模型的权限。

2. **模型版本**：
   - 确保 Forge API 客户端连接的是所需的更大模型（例如 ESM3 的高参数版本）。
   - 可通过 Forge 的文档查阅支持的模型规格。

3. **请求限制与并发**：
   - 检查 Forge API 的请求速率限制。
   - 如果需要生成大量样本，可分批调用以避免超出限额。

4. **生成样本质量验证**：
   - 对生成的样本进行质量评估，确保样本特性符合原有类别。
   - 可以引入简单的相似性测量（如编辑距离或序列比对）来筛选样本。

---

### **扩展建议**

1. **多样性优化**：
   - 通过更高的温度值（`temperature > 0.8`）生成更丰富的样本。
   - 控制掩码比例（例如掩盖 10%-30%）以增加生成的变化。

2. **类别区分增强**：
   - 在生成过程中可以添加类别标签提示（如 `Class1:`），帮助模型理解不同类别的特性。

3. **生成后处理**：
   - 使用工具如 `BLAST` 或序列比对软件验证生成样本的生物学合理性。

---

### **总结**
通过 Forge API 调用更大的 ESM3 模型，可以生成更加多样化和高质量的分类样本。上述代码为每个类别自动生成相似样本，并结合掩码提示机制确保生成的序列符合类别特性。适当调整生成参数和后处理策略，可以进一步提升数据扩展的效果和质量。  

是的，提供更多的原始样本可以显著提高扩增样本的多样性和质量。以下是详细的解释和建议：

---

### **提供更多原始样本的影响**

1. **样本覆盖范围更广**：
   - 原始样本越多，输入序列的种类和特征差异越大，模型在生成时可以基于更丰富的上下文信息，生成更多样本类型。

2. **多样性显著提升**：
   - 如果只有少量原始样本，扩增的样本可能过于集中，生成的结果可能相似。
   - 提供几十个甚至数百个原始样本，可以避免这种问题，让扩增样本更加多样化。

3. **类别特征的强化**：
   - 原始样本越多，类别特征的表示就越准确，生成的样本会更符合类别的核心特性。

4. **数据量线性扩展**：
   - 提供更多原始样本，每个样本扩增固定数量的序列，最终扩增数据量与原始样本数量成倍增长。

---

### **扩增逻辑**

1. **大规模输入**：
   - 将几十甚至上百个原始样本作为输入。
   - 随机从原始样本集中抽取样本，进行掩码并生成新序列。

2. **分批生成**：
   - Forge API 有速率限制（如每秒请求次数），可以将原始样本分成多个批次，每批次处理若干样本。

3. **高效利用生成资源**：
   - 通过调整掩码比例和生成参数（`temperature` 和 `num_steps`），进一步增加生成的样本多样性。

---

### **代码实现**

以下代码演示如何扩展支持更大规模的原始样本集：

```python
import random
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    GenerationConfig,
)

# 初始化 Forge API 客户端
client = ESM3InferenceClient(api_key="your_forge_api_key")

# 定义大规模类别样本集
categories = {
    "Class1": [
        "DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYR",
        "GQGSEHTVDKKKYAAELHLVHWNTKYGD",
        "KPGLQKVVDVLDSIKTKGKSADFTN",
        "MAESDPLHQFHFHWGSLDGQGSEHTVDLHFNTGDY",
        "DLSPQPWRGDYVGWIKADPVDIK",  # 添加更多样本
        # 假设有几十个样本
    ],
    "Class2": [
        "LLDSSAAHLVFHNGPGQGSPDDGKVK",
        "VGWIKNADFSDQLTGLGYVWNKHV",
        "SLDPWRGDYKPGLSAVDLKDQ",
        "DGDVFPLLSGQVKNLHF",
        # 假设有几十个样本
    ],
}

# 生成函数
def generate_similar_samples(category_samples, num_samples, temperature=0.8, num_steps=10):
    """
    根据类别样本生成扩增样本
    :param category_samples: 类别的原始样本列表
    :param num_samples: 每个样本生成的扩增数量
    :param temperature: 控制生成多样性的温度参数
    :param num_steps: 生成的步数
    :return: 扩增的样本列表
    """
    generated_samples = []
    for _ in range(num_samples):
        # 随机选择一个原始样本
        prompt = random.choice(category_samples)
        
        # 对选择的样本进行随机掩码
        masked_prompt = list(prompt)
        for i in range(len(masked_prompt)):
            if random.random() < 0.2:  # 掩码20%的位置
                masked_prompt[i] = "_"
        masked_prompt = "".join(masked_prompt)

        # 构造 ESMProtein 对象
        protein = ESMProtein(sequence=masked_prompt)

        # 调用 Forge API 生成序列
        protein = client.generate(
            protein,
            GenerationConfig(track="sequence", num_steps=num_steps, temperature=temperature),
        )
        generated_samples.append(protein.sequence)

    return generated_samples

# 批量生成扩增样本
num_samples_per_class = 100  # 每类扩增100个样本
expanded_data = {}

for category, samples in categories.items():
    print(f"Generating for {category}...")
    expanded_data[category] = []
    for i in range(len(samples)):
        # 每个原始样本扩增若干次
        expanded_data[category] += generate_similar_samples(
            category_samples=samples,
            num_samples=5,  # 每个原始样本扩增 5 个
            temperature=0.8,
            num_steps=12,
        )

# 打印生成结果
for category, samples in expanded_data.items():
    print(f"\nCategory: {category}")
    for seq in samples:
        print(seq)
```

---

### **代码特点**

1. **大规模样本支持**：
   - 允许每个类别提供几十到数百个原始样本。
   - 自动从原始样本集中随机选择一个样本进行扩增。

2. **每个原始样本扩增多次**：
   - 每个原始样本生成固定数量的扩增样本。
   - 例如，每个样本扩增 5 个，若类别有 50 个原始样本，最终生成 50 × 5 = 250 个新样本。

3. **随机掩码增强多样性**：
   - 掩码比例固定为 20%，每次生成时掩码位置随机。
   - 同一个原始样本生成的样本也具有一定的差异性。

4. **动态生成控制**：
   - **`temperature=0.8`**：较高的温度提升多样性。
   - **`num_steps=12`**：生成过程更灵活，适应长序列。

---

### **扩展样本的效果**

1. **样本多样性**：
   - 原始样本越多，扩增样本覆盖的范围越广，能够更全面表示类别特性。
   - 每次生成会引入一定的随机性，确保样本间的差异。

2. **类别特征稳定**：
   - 扩增样本保持与原始样本的高相似性，同时适当增加变异，使得样本兼具特性与多样性。

3. **规模扩展**：
   - 如果原始样本有 100 条，每条扩增 5 次，最终可生成 500 条新样本。
   - 若需要更大数据集，只需适当增加原始样本或每条的扩增次数。

---

### **总结**

- **提供更多原始样本**：
  - 可以显著提高扩增样本的多样性和质量。
  - 使用几十甚至数百个原始样本，结合随机掩码和多样性控制参数（如 `temperature`），可以生成更全面的扩增数据集。

- **适合规模化生成**：
  - 将 Forge API 与批量生成逻辑结合，能够高效处理大规模样本扩增需求。

- **生成策略灵活**：
  - 可根据任务需求调整扩增数量、掩码比例和生成参数，以适配不同场景。

通过该方法，能够有效解决数据不足问题，为多类别分类任务提供高质量的训练数据。

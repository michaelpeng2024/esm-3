## results_visualizer-代码分析
这段代码 `results_visualizer.py` 旨在创建一个交互式的可视化工具，用于展示和分析一组蛋白质（`ESMProtein` 对象）的不同属性。该工具利用了 `ipywidgets` 和 `matplotlib` 等库，提供了多种视图模式，包括序列展示、溶剂可及表面积（SASA）、二级结构、蛋白质结构以及功能注释等。以下是对代码各部分功能的详细分析：

### 1. 导入必要的库和模块

```python
from datetime import datetime
from functools import partial
from typing import Any, Callable, Literal

import ipywidgets as widgets
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from esm.sdk.api import ESMProtein
from esm.widgets.utils.drawing.draw_category_array import (
    draw_data_array,
)
from esm.widgets.utils.drawing.draw_function_annotations import (
    draw_function_annotations,
)
from esm.widgets.utils.drawing.draw_protein_structure import (
    draw_protein_structure,
)
from esm.widgets.utils.serialization import (
    create_download_button_from_buffer,
    protein_to_pdb_buffer,
)
```

- **标准库**：
  - `datetime` 用于生成时间戳，主要用于文件下载的命名。
  - `functools.partial` 用于部分函数应用，简化回调函数的传递。
  - `typing` 提供类型注解，增强代码的可读性和可维护性。

- **第三方库**：
  - `ipywidgets` 用于创建交互式小部件（widgets），例如按钮、标签等。
  - `matplotlib` 用于绘图和颜色映射。

- **自定义模块**：
  - `ESMProtein` 类表示蛋白质对象，包含多种属性，如序列、SASA、二级结构等。
  - `draw_*` 模块提供了用于绘制不同类型数据的辅助函数。
  - `serialization` 模块包含用于序列化和下载蛋白质结构的函数。

### 2. 创建结果可视化器的主函数

```python
def create_results_visualizer(
    modality: str,
    samples: list[ESMProtein],
    items_per_page: int = 4,
    copy_to_prompt_callback: Callable[
        [
            Literal[
                "sequence", "coordinates", "secondary_structure", "sasa", "function"
            ],
            Any,
        ],
        None,
    ]
    | None = None,
    include_title: bool = True,
) -> widgets.Widget:
    # 函数主体
```

#### 参数说明：

- `modality`（字符串）：指定可视化的模式，包括 `"sequence"`（序列）、`"sasa"`（溶剂可及表面积）、`"secondary_structure"`（二级结构）、`"structure"`（蛋白质结构）、`"function"`（功能注释）等。
- `samples`（`ESMProtein` 对象列表）：需要可视化的蛋白质样本。
- `items_per_page`（整数，默认值为4）：每页显示的样本数量。
- `copy_to_prompt_callback`（可调用对象，默认值为`None`）：可选的回调函数，当用户点击“Copy to Prompt”按钮时调用，将相关数据复制到提示中。
- `include_title`（布尔值，默认值为`True`）：是否在可视化器顶部包含标题。

#### 函数功能：

1. **排序**：
   - 如果 `modality` 为 `"structure"`，则根据蛋白质的 pTM（pTM评分）对样本进行降序排序，以优先显示高评分的结构。

2. **分页设置**：
   - 计算总页数 `total_pages`，并初始化当前页 `current_page` 为1。
   - 创建页面标签 `page_label`，显示当前页码和总页数。
   - 创建“Next”和“Previous”按钮，用于页面导航，并根据当前页码禁用相应的按钮。

3. **页面更新函数 `update_page`**：
   - 根据当前页码计算要显示的样本范围。
   - 根据 `modality` 调用相应的子函数创建当前页的可视化内容（如序列展示、SASA展示等）。
   - 更新输出区域的内容，并刷新页面标签和按钮状态。

4. **按钮点击事件处理**：
   - `on_next_button_clicked`：点击“Next”按钮时，若未到最后一页，则页码加1并更新页面。
   - `on_prev_button_clicked`：点击“Previous”按钮时，若未到第一页，则页码减1并更新页面。

5. **初始化和布局**：
   - 调用 `update_page` 初始化第一页内容。
   - 创建一个垂直布局 `results_ui`，包含标题（如果 `include_title` 为真）、导航栏（如果有多页）和输出区域。

### 3. 辅助函数

#### 3.1 添加换行符函数

```python
def add_line_breaks(sequence: str, line_length: int = 120) -> str:
    return "<br>".join(
        [sequence[i : i + line_length] for i in range(0, len(sequence), line_length)]
    )
```

- **功能**：将长字符串（如蛋白质序列）按指定长度 `line_length` 分割，并在每个分段后添加 HTML 换行符 `<br>`，以便在网页中更好地显示。

#### 3.2 创建序列展示页

```python
def create_sequence_results_page(
    items: list[ESMProtein],
    line_length: int = 120,
    copy_to_prompt_callback: Callable[[Any], None] | None = None,
) -> widgets.Widget:
    # 函数主体
```

- **功能**：展示蛋白质序列，每个序列项包括可选的“Copy to Prompt”按钮和序列本身。
- **实现细节**：
  - 遍历 `items` 列表，为每个蛋白质创建一个垂直布局（`VBox`），其中包含按钮和序列显示。
  - 如果提供了 `copy_to_prompt_callback`，则按钮点击时将序列复制到提示中。
  - 使用 `<pre>` 标签和 `white-space: pre-wrap` 样式确保序列以等宽字体和保留空白的形式显示。

#### 3.3 创建溶剂可及表面积（SASA）展示页

```python
def create_sasa_results_page(
    items: list[ESMProtein],
    copy_to_prompt_callback: Callable[[Any], None] | None = None,
) -> widgets.Widget:
    # 函数主体
```

- **功能**：展示蛋白质的溶剂可及表面积（SASA）数据，每个SASA项包括可选的“Copy to Prompt”按钮和SASA数据的可视化图。
- **实现细节**：
  - 遍历 `items` 列表，为每个蛋白质创建一个垂直布局，包含按钮和SASA图。
  - 使用 `draw_data_array` 函数绘制SASA数据，使用红色调色板（`cmap="Reds"`）。
  - 如果SASA数据不可用，则显示相应的提示信息。

#### 3.4 创建二级结构展示页

```python
def create_secondary_structure_results_page(
    items: list[ESMProtein],
    copy_to_prompt_callback: Callable[[Any], None] | None = None,
) -> widgets.Widget:
    # 函数主体
```

- **功能**：展示蛋白质的二级结构信息，每个项包括可选的“Copy to Prompt”按钮和二级结构的可视化图。
- **实现细节**：
  - 将二级结构字母（C、H、E）映射为数值（0、1、2），以便绘图。
  - 使用 `draw_data_array` 函数绘制二级结构数据，定义了相应的类别和颜色映射：
    - `Coil (C)`：浅蓝色
    - `Alpha helix (H)`：浅绿色
    - `Beta strand (E)`：浅红色

#### 3.5 创建蛋白质结构展示页

```python
def create_structure_results_page(
    items: list[ESMProtein],
    copy_to_prompt_callback: Callable[[Any], None] | None = None,
) -> widgets.Widget:
    # 函数主体
```

- **功能**：展示蛋白质的三维结构图，每个结构项包括可选的“Copy to Prompt”按钮、下载PDB文件按钮、pTM评分标签和结构图。
- **实现细节**：
  - 计算网格大小，以在页面上整齐排列多个结构图。
  - 为每个蛋白质创建一个网格单元，包含：
    - **Header**：
      - 可选的“Copy to Prompt”按钮（复制坐标数据）。
      - 下载PDB文件按钮，使用当前时间戳命名文件。
      - pTM评分标签（如果可用）。
    - **输出区域**：
      - 使用 `draw_protein_structure` 函数绘制蛋白质结构图。
      - 根据 pLDDT（预测的结构置信度评分）为结构图中的区域着色。
      - 颜色映射使用蓝色调色板（`cmap="Blues"`）。

#### 3.6 创建功能注释展示页

```python
def create_function_annotations_results_page(
    items: list[ESMProtein],
    copy_to_prompt_callback: Callable[[Any], None] | None = None,
) -> widgets.Widget:
    # 函数主体
```

- **功能**：展示蛋白质的功能注释，特别是InterPro注释，每个注释项包括可选的“Copy to Prompt”按钮和注释图。
- **实现细节**：
  - 筛选出包含 "IPR" 标签的功能注释。
  - 如果没有可用的InterPro注释，则显示提示信息。
  - 使用 `draw_function_annotations` 函数绘制功能注释图，显示注释在序列上的分布。

### 4. 总结

整个 `results_visualizer.py` 脚本通过定义多个子函数，分别处理不同类型的数据可视化需求，并通过主函数 `create_results_visualizer` 将这些功能集成到一个交互式的小部件中。用户可以根据不同的 `modality` 选择相应的视图模式，浏览和分析蛋白质的各类属性。此外，提供的“Copy to Prompt”功能允许用户将感兴趣的数据快速复制到其他应用场景中，增强了工具的实用性和灵活性。

该工具在蛋白质研究和分析中具有广泛的应用前景，特别是在需要对大量蛋白质数据进行可视化和快速浏览的场景下，能够显著提升工作效率和数据洞察能力。

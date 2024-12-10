## sasa_prompt_selector-代码分析
这段代码定义了一个名为 `sasa_prompt_selector.py` 的 Python 模块，主要用于在蛋白质链的溶剂可及表面积（SASA, Solvent Accessible Surface Area）数据上创建一个交互式的提示选择器。以下是对代码功能的详细中文分析：

### 1. 导入模块和依赖

```python
from typing import Any, Callable, Sequence
import ipywidgets as widgets
from esm.utils.structure.protein_chain import ProteinChain
from esm.widgets.utils.drawing.colors import (
    hex_to_rgba_tuple,
    rgba_tuple_to_hex,
)
from esm.widgets.utils.drawing.draw_category_array import (
    draw_data_array,
)
from esm.widgets.utils.parsing import (
    convert_range_string_to_list_of_ranges,
)
from esm.widgets.utils.prompting import PromptManager
```

- **标准库导入**：`typing` 模块用于类型提示。
- **第三方库**：`ipywidgets` 用于创建交互式小部件（Widgets）。
- **自定义模块**：
  - `ProteinChain`：处理蛋白质链的结构数据。
  - `hex_to_rgba_tuple` 和 `rgba_tuple_to_hex`：颜色转换工具。
  - `draw_data_array`：用于绘制数据数组的工具。
  - `convert_range_string_to_list_of_ranges`：解析范围字符串。
  - `PromptManager`：管理提示（prompt）的工具。

### 2. `create_sasa_prompt_selector` 函数

```python
def create_sasa_prompt_selector(
    prompt_manager: PromptManager,
    tag: str,
    *,
    input_array: Sequence[float] | None = None,
    protein_chain: ProteinChain | None = None,
    with_title: bool = True,
    active_tag_callback: Callable[[], str] | None = None,
) -> widgets.Widget:
    ...
```

该函数用于创建一个 Sasa 提示选择器小部件。参数说明：

- `prompt_manager`：用于管理提示的实例。
- `tag`：提示的标签。
- `input_array`：可选的输入数组，表示 Sasa 值。
- `protein_chain`：可选的蛋白质链对象，用于计算 Sasa 值。
- `with_title`：是否显示标题。
- `active_tag_callback`：回调函数，用于判断当前激活的标签。

#### 主要功能步骤：

1. **确定活动标签**：

    ```python
    is_active_callback = (
        lambda: active_tag_callback() == tag if active_tag_callback else True
    )
    ```

    如果提供了 `active_tag_callback`，则通过回调函数判断当前标签是否激活，否则默认激活。

2. **获取输入数组**：

    ```python
    if input_array is None:
        if protein_chain is not None:
            input_array = get_sasa(protein_chain)
        else:
            raise ValueError("Either input_array or protein_chain must be provided.")
    ```

    如果没有提供 `input_array`，则尝试通过 `protein_chain` 计算得到 Sasa 值；如果两者都未提供，则抛出错误。

3. **创建范围滑块（Range Slider）**：

    ```python
    range_slider = widgets.IntRangeSlider(
        value=[0, 2],
        min=0,
        max=len(input_array) - 1,
        step=1,
        description="Range:",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="50%"),
    )
    output = widgets.Output()
    ```

    - `range_slider` 用于选择 Sasa 值的范围。
    - `output` 用于显示绘制的图形。

4. **管理高亮范围**：

    ```python
    highlighted_ranges = []
    ```

    初始化一个列表，用于存储高亮显示的范围。

5. **更新高亮范围的函数**：

    ```python
    def update_highlighted_ranges() -> list[tuple[int, int, Any]]:
        nonlocal highlighted_ranges

        def _lower_alpha(hex_color: str, alpha: float) -> str:
            r, g, b, _ = hex_to_rgba_tuple(hex_color)
            return rgba_tuple_to_hex((r, g, b, alpha))

        highlighted_ranges = [
            (start, end, _lower_alpha(color, alpha=0.6))
            for _, (color, ranges, _) in prompt_manager.get_prompts(tag=tag).items()
            for start, end in ranges
        ]
        # 添加当前滑块选择的范围到高亮范围
        range_slider_value = range_slider.value
        if range_slider_value:
            highlighted_ranges.insert(
                0,
                (
                    range_slider_value[0],
                    range_slider_value[1],
                    _lower_alpha(prompt_manager.get_current_color(), alpha=0.8),
                ),
            )
        return highlighted_ranges
    ```

    - **目的**：更新所有需要高亮显示的范围，包括已存在的提示范围和当前滑块选择的范围。
    - **颜色处理**：通过 `_lower_alpha` 函数调整颜色的透明度，以便区分不同的高亮区域。

6. **重绘函数**：

    ```python
    def redraw(highlighted_ranges):
        draw_data_array(
            output,
            data_array=input_array,
            category_color_mapping={},
            highlighted_ranges=highlighted_ranges,
            cmap="Reds",
        )
    ```

    使用 `draw_data_array` 函数根据 `input_array` 和 `highlighted_ranges` 绘制图形，颜色映射使用 "Reds" 颜色图。

7. **初始绘制**：

    ```python
    redraw(highlighted_ranges)
    ```

    初始时绘制图形，尚未高亮任何范围。

8. **将范围转换为 Sasa 模式**：

    ```python
    def range_to_sasa_motif(range: tuple[int, int]) -> Sequence[float]:
        return input_array[range[0] : range[1] + 1]
    ```

    根据选择的范围提取对应的 Sasa 值子数组。

9. **添加条目函数**：

    ```python
    def add_entry(_):
        if is_active_callback():
            if prompt_manager.manual_selection_checkbox.value:
                range_string = prompt_manager.manual_input.value
                selected_ranges = convert_range_string_to_list_of_ranges(range_string)
            else:
                start, end = range_slider.value
                selected_ranges = [(start, end)]

            prompt_manager.add_entry(
                selected_ranges,
                tag=tag,
                get_value_from_range_callback=range_to_sasa_motif,
            )
            update_visualizer()
    ```

    - **功能**：将当前选择的范围添加为一个新的提示条目。
    - **手动选择**：如果手动选择复选框被选中，则从用户输入的范围字符串中解析范围；否则使用滑块当前选择的范围。
    - **添加条目**：通过 `prompt_manager` 添加新的提示条目，并更新可视化。

10. **切换手动选择的回调函数**：

    ```python
    def toggle_manual_selection(change):
        if change["new"]:
            prompt_manager.manual_input.disabled = False
            range_slider.disabled = True
            highlighted_ranges = update_highlighted_ranges()
            highlighted_ranges.pop()  # 移除当前滑块范围
            redraw(highlighted_ranges)
        else:
            prompt_manager.manual_input.disabled = True
            range_slider.disabled = False
            update_visualizer()
    ```

    - **功能**：在用户切换手动选择模式时，启用或禁用相应的小部件，并更新高亮显示。
    - **手动模式启用**：禁用滑块，启用手动输入，并移除滑块当前选择的高亮范围。
    - **手动模式禁用**：启用滑块，禁用手动输入，并更新可视化。

11. **更新可视化的函数**：

    ```python
    def update_visualizer(*args, **kwargs):
        highlighted_ranges = update_highlighted_ranges()
        redraw(highlighted_ranges)
    ```

    调用 `update_highlighted_ranges` 获取最新的高亮范围，并重新绘制图形。

12. **绑定事件**：

    ```python
    prompt_manager.add_button.on_click(add_entry)
    prompt_manager.manual_selection_checkbox.observe(
        toggle_manual_selection, names="value"
    )
    prompt_manager.register_delete_callback(update_visualizer)

    range_slider.observe(update_visualizer, names="value")
    ```

    - **添加按钮点击事件**：点击添加按钮时调用 `add_entry`。
    - **手动选择复选框变化**：观察复选框的值变化，调用 `toggle_manual_selection`。
    - **删除回调**：注册删除提示条目的回调，更新可视化。
    - **滑块值变化**：观察滑块值的变化，调用 `update_visualizer`。

13. **组合 UI 元素**：

    ```python
    main_ui = widgets.VBox([range_slider, output, prompt_manager.get_selection_ui()])
    ```

    将滑块、输出区域和提示管理器的选择界面垂直排列。

14. **添加标题（可选）**：

    ```python
    if with_title:
        heading = widgets.HTML(value="<h1>SASA Prompt Selector:</h1>")
        parent_layout = widgets.VBox([heading, main_ui])
        return parent_layout
    else:
        return main_ui
    ```

    - **带标题**：如果 `with_title` 为 `True`，则在 UI 顶部添加一个标题。
    - **不带标题**：仅返回主要的 UI 元素。

### 3. `get_sasa` 辅助函数

```python
def get_sasa(protein_chain: ProteinChain) -> Sequence[float]:
    sasa_values = protein_chain.sasa().tolist()
    return sasa_values
```

- **功能**：从 `ProteinChain` 对象中计算 Sasa 值，并将其转换为列表形式返回。
- **用途**：在 `create_sasa_prompt_selector` 中，当未提供 `input_array` 时，通过该函数获取 Sasa 数据。

### 总结

整体来说，`sasa_prompt_selector.py` 模块通过以下步骤实现了一个用于选择和管理蛋白质链 Sasa 数据范围的交互式界面：

1. **数据获取**：从蛋白质链对象或提供的数组中获取 Sasa 数据。
2. **UI 组件**：使用 `ipywidgets` 创建范围滑块、输出区域和提示管理器的 UI 元素。
3. **高亮显示**：根据用户选择的范围或手动输入的范围，动态高亮显示 Sasa 数据图表中的相应区域。
4. **事件绑定**：将用户的交互操作（如滑动滑块、点击添加按钮、切换手动选择等）与相应的处理函数绑定，实现动态更新和交互。
5. **提示管理**：通过 `PromptManager` 管理用户添加的提示条目，支持添加、删除和更新操作。

该模块适用于需要在蛋白质结构分析中对 Sasa 数据进行交互式选择和可视化的场景，提供了灵活且用户友好的界面，便于研究人员进行数据探索和分析。

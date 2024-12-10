## sequence_prompt_selector-代码分析
这段代码定义了一个名为 `create_sequence_prompt_selector` 的函数，主要用于创建一个交互式的序列选择器小部件（widget）。该选择器允许用户在给定的序列中选择特定的范围、应用掩码选项，并通过高亮显示不同的区域来管理和可视化提示（prompts）。以下是对代码各部分的详细分析：

### 1. 导入模块和类型注解
```python
from typing import Callable
import ipywidgets as widgets
from esm.widgets.utils.drawing.colors import (
    hex_to_rgba_tuple,
    rgba_tuple_to_rgba_html_string,
)
from esm.widgets.utils.parsing import (
    convert_range_string_to_list_of_ranges,
)
from esm.widgets.utils.prompting import PromptManager
```
- **typing**: 引入 `Callable` 类型注解，用于函数参数类型提示。
- **ipywidgets**: 用于创建交互式小部件。
- **esm.widgets.utils.drawing.colors**: 包含颜色处理工具，用于将十六进制颜色转换为 RGBA 元组，并将 RGBA 元组转换为 HTML 格式的颜色字符串。
- **esm.widgets.utils.parsing**: 包含解析工具，用于将范围字符串转换为范围列表。
- **esm.widgets.utils.prompting**: 引入 `PromptManager` 类，用于管理提示（prompts）。

### 2. 函数定义
```python
def create_sequence_prompt_selector(
    prompt_manager: PromptManager,
    tag: str,
    full_sequence: str,
    line_length=120,
    with_title: bool = True,
    active_tag_callback: Callable[[], str] | None = None,
) -> widgets.Widget:
```
该函数接受以下参数：
- `prompt_manager`: 一个 `PromptManager` 实例，用于管理提示相关的操作。
- `tag`: 一个字符串标签，用于标识特定的提示类别。
- `full_sequence`: 要显示和操作的完整序列（字符串）。
- `line_length`: 每行显示的字符长度，默认值为120。
- `with_title`: 是否显示标题，默认值为 `True`。
- `active_tag_callback`: 一个可选的回调函数，用于判断当前激活的标签。

函数返回一个 `ipywidgets.Widget` 小部件。

### 3. 初始化序列长度和激活回调
```python
sequence_length = len(full_sequence)

is_active_callback = (
    lambda: active_tag_callback() == tag if active_tag_callback else True
)
```
- `sequence_length`: 计算序列的长度。
- `is_active_callback`: 定义一个 lambda 函数，如果提供了 `active_tag_callback`，则检查当前激活的标签是否与传入的 `tag` 匹配；否则默认返回 `True`。

### 4. 创建滑动条（Range Slider）
```python
range_slider = widgets.IntRangeSlider(
    value=[0, sequence_length - 1],
    min=0,
    max=sequence_length - 1,
    step=1,
    description="Crop Range",
    orientation="horizontal",
    readout=True,
    readout_format="d",
    layout=widgets.Layout(width="600px"),
)
```
- 创建一个整数范围滑动条，允许用户选择序列中的起始和结束位置。
- 滑动条的初始值覆盖整个序列（从0到序列长度减一）。
- 设置滑动条的布局宽度为600像素。

### 5. 创建掩码选项（Radio Buttons）
```python
mask_option = widgets.RadioButtons(
    options=["Custom range", "Fully masked"],
    value="Custom range",
    description="Mask Option:",
    disabled=False,
)
```
- 创建一个单选按钮，提供两个选项：“Custom range”（自定义范围）和“Fully masked”（完全掩码）。
- 默认选中“Custom range”。

### 6. 创建输出区域
```python
output = widgets.HTML(layout=widgets.Layout(width="600px", white_space="pre-wrap"))
```
- 创建一个 HTML 输出小部件，用于显示序列和高亮信息。
- 设置布局宽度为600像素，并启用 `white_space: pre-wrap` 以保留空格和换行符。

### 7. 定义回调函数
#### 7.1 切换掩码选项
```python
def toggle_mask_option(change):
    if mask_option.value == "Fully masked":
        range_slider.disabled = True
        if is_active_callback():
            prompt_manager.add_button.disabled = True
    else:
        range_slider.disabled = False
        if is_active_callback():
            prompt_manager.add_button.disabled = False
    update_sequence()
```
- 当用户切换掩码选项时调用。
- 如果选择“Fully masked”，禁用范围滑动条，并在激活状态下禁用添加按钮。
- 如果选择“Custom range”，启用范围滑动条，并在激活状态下启用添加按钮。
- 调用 `update_sequence` 更新显示内容。

#### 7.2 切换手动选择
```python
def toggle_manual_selection(change):
    if change["new"]:
        prompt_manager.manual_input.disabled = False
        range_slider.disabled = True
    else:
        prompt_manager.manual_input.disabled = True
        range_slider.disabled = False
```
- 当用户选择手动输入范围时调用。
- 如果启用手动输入，禁用范围滑动条；否则，禁用手动输入并启用滑动条。

#### 7.3 更新序列显示
```python
def update_sequence(change=None):
    highlighted_sequence = list(full_sequence)
    ranges = []
    for _, (color, rs, _) in prompt_manager.get_prompts(tag=tag).items():
        for start, end in rs:
            ranges.append((start, end, color))

    if ranges:
        highlighted_sequence = apply_highlighting(full_sequence, ranges)
    else:
        highlighted_sequence = add_line_breaks(full_sequence)

    current_range = ["_"] * sequence_length
    start, end = range_slider.value
    current_range[start : end + 1] = full_sequence[start : end + 1]

    current_color = prompt_manager.get_current_color()
    current_range_html = f"<b>Current Selection:</b><p style='color:{current_color}; font-family:monospace;'>{add_line_breaks(''.join(current_range))}</p><br>"
    highlighted_sequence_html = f"<b>Selected:</b><p style='font-family:monospace;'>{highlighted_sequence}<br>{current_range_html}</p>"
    output.value = highlighted_sequence_html
```
- 从 `PromptManager` 获取当前标签下的所有提示，并收集所有的范围和对应颜色。
- 如果存在范围，则调用 `apply_highlighting` 函数对序列进行高亮显示；否则，调用 `add_line_breaks` 函数添加换行符。
- 根据滑动条的当前值，生成当前选择的序列部分，并用下划线填充未选择的部分。
- 获取当前颜色并生成 HTML 格式的当前选择显示。
- 将高亮后的序列和当前选择部分更新到 `output` 小部件中。

#### 7.4 添加换行符
```python
def add_line_breaks(sequence):
    return "<br>".join(
        [
            sequence[i : i + line_length]
            for i in range(0, len(sequence), line_length)
        ]
    )
```
- 将长序列按 `line_length`（默认120）分割成多行，并在每行之间插入 `<br>` 标签以实现换行。

#### 7.5 应用高亮显示
```python
def apply_highlighting(sequence, ranges):
    lines = [
        sequence[i : i + line_length] for i in range(0, len(sequence), line_length)
    ]

    highlighted_lines = []
    for line_idx, line in enumerate(lines):
        line_start = line_idx * line_length
        line_end = line_start + len(line)
        highlighted_line = list(line)
        all_line_ranges = [
            (
                max(start, line_start) - line_start,
                min(end, line_end - 1) - line_start,
                color,
            )
            for start, end, color in ranges
            if start < line_end and end >= line_start
        ]
        for i in range(len(highlighted_line)):
            span_layers = []
            for start, end, color in all_line_ranges:
                if start <= i <= end:
                    span_layers.append(color)
            if span_layers:
                combined_color = span_layers[-1]
                r, g, b, a = hex_to_rgba_tuple(combined_color)
                a = 0.5  # 设置透明度为0.5
                combined_color = rgba_tuple_to_rgba_html_string((r, g, b, a))
                highlighted_line[i] = (
                    f'<span style="background-color:{combined_color}">{highlighted_line[i]}</span>'
                )
        highlighted_lines.append("".join(highlighted_line))

    return "<br>".join(highlighted_lines)
```
- 将序列按行长度分割成多行。
- 对每一行，确定需要高亮显示的范围，并为每个字符应用相应的背景颜色。
- 颜色的透明度设置为0.5，以便多层叠加时颜色能够叠加显示。
- 将高亮后的行合并，并通过 `<br>` 标签连接，实现多行显示。

### 8. 监听滑动条和掩码选项的变化
```python
range_slider.observe(update_sequence, names="value")
mask_option.observe(toggle_mask_option, names="value")
```
- 当滑动条的值发生变化时，调用 `update_sequence` 更新显示内容。
- 当掩码选项发生变化时，调用 `toggle_mask_option` 切换掩码模式。

### 9. 将范围转换为序列基序（Motif）
```python
def range_to_sequence_motif(range: tuple[int, int]) -> str:
    return full_sequence[range[0] : range[1] + 1]
```
- 根据给定的范围，提取序列的子串。

### 10. 处理添加按钮点击事件
```python
def handle_add_button_click(_):
    if is_active_callback():
        if prompt_manager.manual_selection_checkbox.value:
            selected_ranges = convert_range_string_to_list_of_ranges(
                prompt_manager.manual_input.value
            )
        else:
            selected_ranges = [range_slider.value]
        prompt_manager.add_entry(
            selected_ranges,
            tag=tag,
            get_value_from_range_callback=range_to_sequence_motif,
        )
        update_sequence()
```
- 当用户点击添加按钮时调用。
- 如果当前标签处于激活状态：
  - 如果启用了手动选择，解析手动输入的范围字符串；
  - 否则，使用滑动条选择的范围。
- 调用 `PromptManager` 的 `add_entry` 方法添加新的提示条目。
- 更新显示内容。

### 11. 绑定添加按钮和删除回调
```python
prompt_manager.add_button.on_click(lambda x: handle_add_button_click(""))
prompt_manager.register_delete_callback(update_sequence)
prompt_manager.manual_selection_checkbox.observe(
    toggle_manual_selection, names="value"
)
```
- 将添加按钮的点击事件绑定到 `handle_add_button_click` 函数。
- 注册删除回调，当提示条目被删除时，调用 `update_sequence` 更新显示。
- 监听手动选择复选框的变化，调用 `toggle_manual_selection` 切换手动输入模式。

### 12. 初始更新显示内容
```python
update_sequence()
```
- 在小部件创建完成后，立即调用 `update_sequence` 以显示初始的序列内容。

### 13. 返回最终的小部件布局
```python
if with_title:
    return widgets.VBox(
        [
            widgets.HTML(value="<h1>Sequence:</h1>"),
            mask_option,
            range_slider,
            output,
            prompt_manager.get_selection_ui(),
        ]
    )
else:
    return widgets.VBox(
        [mask_option, range_slider, output, prompt_manager.get_selection_ui()]
    )
```
- 根据 `with_title` 参数，决定是否在小部件顶部添加标题“Sequence:”。
- 使用 `VBox` 布局将各个子小部件垂直排列，包括：
  - 标题（可选）
  - 掩码选项单选按钮
  - 范围滑动条
  - 输出显示区域
  - 由 `PromptManager` 提供的选择界面（例如添加按钮、手动输入等）

### 总结
`create_sequence_prompt_selector` 函数通过组合多个 `ipywidgets` 小部件，提供了一个强大的交互式界面，用于选择和管理序列中的特定范围。用户可以通过滑动条选择序列的子范围，选择是否完全掩码，手动输入范围，添加和删除提示条目，并通过颜色高亮显示不同的提示区域。这对于需要在序列数据上进行可视化和交互式标注的应用场景（如生物序列分析、文本处理等）非常有用。

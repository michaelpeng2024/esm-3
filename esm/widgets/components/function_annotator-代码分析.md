## function_annotator-代码分析
这段代码 `function_annotator.py` 实现了一个基于 Jupyter Notebook 的交互式函数注释工具。该工具允许用户在蛋白质序列的特定区域内添加功能注释，并提供关键词自动补全和建议功能。以下是对代码的详细分析：

### 1. 引入必要的库和模块

```python
from typing import Callable
import pygtrie
from ipywidgets import widgets
from esm.sdk.api import FunctionAnnotation
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
```

- **`typing.Callable`**: 用于类型提示，指定回调函数的类型。
- **`pygtrie`**: 一个高效的前缀树（Trie）实现，用于存储和查找关键词。
- **`ipywidgets`**: 提供交互式小部件，用于在 Jupyter Notebook 中构建用户界面。
- **`FunctionAnnotation`**: 表示功能注释的类，来自 `esm.sdk.api` 模块。
- **`InterProQuantizedTokenizer`**: 用于分词和处理 InterPro 数据库中的标签，来自 `esm.tokenization.function_tokenizer` 模块。

### 2. 全局变量定义

```python
TRIE: pygtrie.CharTrie | None = None
```

- **`TRIE`**: 全局变量，用于存储关键词的前缀树，初始值为 `None`。

### 3. 获取和初始化前缀树

```python
def get_trie() -> pygtrie.CharTrie:
    global TRIE
    if TRIE is None:
        # 初始化关键词前缀树
        TRIE = pygtrie.CharTrie(separator=" ")
        interpro_tokenizer = InterProQuantizedTokenizer()
        for keyword in interpro_tokenizer._tfidf.vocabulary:
            TRIE[keyword.lower()] = keyword

        for interpro_tag in interpro_tokenizer.interpro_labels:
            TRIE[interpro_tag.lower()] = interpro_tag
    return TRIE
```

- **`get_trie` 函数**: 检查 `TRIE` 是否已初始化。如果没有，则创建一个新的 `CharTrie` 实例，并使用 `InterProQuantizedTokenizer` 提取关键词和 InterPro 标签，将它们以小写形式存入 `TRIE` 中，以便后续快速查找和自动补全。

### 4. 创建功能注释器的主函数

```python
def create_function_annotator(
    protein_length: int,
    add_annotation_callback: Callable[[FunctionAnnotation], None],
    delete_annotation_callback: Callable[[FunctionAnnotation], None],
) -> widgets.Widget:
    trie = get_trie()
```

- **`create_function_annotator` 函数**: 这是创建功能注释器界面的主要函数，接受以下参数：
  - **`protein_length`**: 蛋白质序列的长度，用于设置目标范围滑动条的最大值。
  - **`add_annotation_callback`**: 添加注释时的回调函数。
  - **`delete_annotation_callback`**: 删除注释时的回调函数。
- 获取初始化好的 `trie` 用于后续的关键词查找。

### 5. 创建交互式小部件

```python
    text_input = widgets.Text(
        description="Function", disabled=False, layout=widgets.Layout(width="400px")
    )
    suggestions = widgets.SelectMultiple(
        options=[],
        description="Suggestions",
        disabled=False,
        layout=widgets.Layout(width="400px"),
    )
    add_button = widgets.Button(
        description="Add",
        disabled=True,
        tooltip="Add the selected function to the target range",
        icon="plus",
    )
    target_range_slider_label = widgets.Label(
        value="Target Range in Prompt:", layout=widgets.Layout(width="150px")
    )
    target_range_slider = widgets.IntRangeSlider(
        value=[0, protein_length - 1],
        min=0,
        max=protein_length - 1,
        step=1,
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout=widgets.Layout(width="600px"),
    )
    output = widgets.Output()
    entries = widgets.VBox([])
```

- **`text_input`**: 文本输入框，用户可以输入功能关键词。
- **`suggestions`**: 多选框，用于显示与输入关键词匹配的建议。
- **`add_button`**: 添加按钮，初始状态为禁用，只有在输入有效关键词时才启用。
- **`target_range_slider_label`**: 标签，描述目标范围滑动条。
- **`target_range_slider`**: 整数范围滑动条，用户可以选择注释应用的蛋白质序列范围。
- **`output`**: 输出区域，用于显示提示信息或错误信息。
- **`entries`**: 垂直盒子，用于显示已添加的注释条目。

### 6. 定义回调函数

#### 6.1 文本输入变化时的回调

```python
    def on_text_change(change):
        output.clear_output()
        text: str = change["new"]
        if not text:
            suggestions.options = []
            return
        try:
            options = list(trie.itervalues(text.lower()))
        except KeyError:
            options = []
            with output:
                print(f"Keyword {text} not found in the Function Annotation vocabulary")
        suggestions.options = options

        if is_keyword_valid(text):
            add_button.disabled = False
        else:
            add_button.disabled = True
```

- **`on_text_change`**: 当用户在 `text_input` 中输入或修改文本时触发。
  - 清除输出区域的内容。
  - 获取新的输入文本，如果为空，则清空建议列表。
  - 使用 `trie.itervalues` 查找与输入文本前缀匹配的所有关键词作为建议选项。如果找不到匹配项，则在输出区域显示错误信息。
  - 检查输入关键词是否有效（存在于 `trie` 中），如果有效，则启用 `add_button`，否则禁用。

#### 6.2 选择建议时的回调

```python
    def on_suggestion_click(change):
        if not change["new"]:
            return
        value, *_ = change["new"]
        text_input.value = value
```

- **`on_suggestion_click`**: 当用户在 `suggestions` 中选择某个建议时触发。
  - 获取选择的第一个建议，并将其设置为 `text_input` 的值。

#### 6.3 添加按钮点击时的回调

```python
    def on_add_click(b):
        output.clear_output()
        try:
            function_label = text_input.value
            start, end = target_range_slider.value
            add_annotation_callback(
                FunctionAnnotation(function_label, start + 1, end + 1)
            )

            function_str = f"[{start}-{end}]: {function_label} "

            def on_delete_click(b):
                delete_annotation_callback(
                    FunctionAnnotation(function_label, start + 1, end + 1)
                )
                entries.children = tuple(
                    entry
                    for entry in entries.children
                    if entry.children[1].value != function_str
                )

            delete_button = widgets.Button(
                description="Delete", tooltip="Delete this annotation", icon="trash"
            )
            entry = widgets.HBox([delete_button, widgets.Label(value=function_str)])
            delete_button.on_click(on_delete_click)
            entries.children += (entry,)

        except Exception as e:
            with output:
                print(f"Error: {e}")
```

- **`on_add_click`**: 当用户点击 `add_button` 时触发。
  - 清除输出区域。
  - 获取输入的功能标签和目标范围。
  - 调用 `add_annotation_callback` 将新的功能注释添加到外部系统或数据结构中。
  - 创建一个显示已添加注释的字符串，例如 `[0-100]: 功能标签`。
  - 定义 `on_delete_click` 回调函数，用于删除该注释：
    - 调用 `delete_annotation_callback` 从外部系统或数据结构中删除注释。
    - 从 `entries` 中移除对应的注释条目。
  - 创建一个删除按钮和一个标签，组合成一个水平盒子（`HBox`），并添加到 `entries` 中。
  - 如果过程中发生任何异常，在输出区域显示错误信息。

#### 6.4 检查关键词是否有效

```python
    def is_keyword_valid(keyword: str) -> bool:
        return keyword.lower() in trie
```

- **`is_keyword_valid`**: 检查输入的关键词是否存在于 `trie` 中，以判断其是否有效。

### 7. 绑定回调函数到小部件

```python
    text_input.observe(on_text_change, names="value")
    suggestions.observe(on_suggestion_click, names="value")
    add_button.on_click(on_add_click)
```

- **`text_input`**: 绑定 `on_text_change` 回调到其 `value` 属性的变化。
- **`suggestions`**: 绑定 `on_suggestion_click` 回调到其 `value` 属性的变化。
- **`add_button`**: 绑定 `on_add_click` 回调到其点击事件。

### 8. 组装并返回用户界面

```python
    function_annotation_ui = widgets.VBox(
        [
            widgets.HBox([text_input, add_button]),
            suggestions,
            widgets.HBox([target_range_slider_label, target_range_slider]),
            output,
            entries,
        ]
    )

    return function_annotation_ui
```

- **`function_annotation_ui`**: 将所有创建的小部件按照垂直布局（`VBox`）和水平布局（`HBox`）组合起来，形成完整的用户界面。
  - 第一行：`text_input` 和 `add_button` 并排显示。
  - 第二行：`suggestions` 显示建议列表。
  - 第三行：`target_range_slider_label` 和 `target_range_slider` 并排显示，用于选择注释的目标范围。
  - 第四行：`output` 显示提示信息或错误信息。
  - 第五行：`entries` 显示已添加的注释条目。
- 最后，返回组装好的 `function_annotation_ui`，供外部调用和展示。

### 总结

该代码实现了一个功能强大的交互式注释工具，主要功能包括：

1. **关键词自动补全**：用户在输入框中输入关键词时，系统会根据预先加载的关键词库（存储在前缀树中）提供实时的建议，帮助用户快速选择有效的功能标签。

2. **范围选择**：通过滑动条，用户可以选择蛋白质序列的具体范围，确保注释的准确性。

3. **添加和删除注释**：用户可以方便地添加新的功能注释，并且能够随时删除不需要的注释。每个注释条目旁边都有一个删除按钮，点击即可移除对应的注释。

4. **错误提示**：如果用户输入的关键词不在预定义的词汇表中，系统会在输出区域显示相应的错误信息，提示用户输入有效的关键词。

5. **可扩展性**：通过回调函数 `add_annotation_callback` 和 `delete_annotation_callback`，该工具可以与外部系统或数据结构集成，实现注释的持久化和管理。

整体而言，这段代码通过结合 `ipywidgets` 提供了一个用户友好且功能丰富的界面，极大地简化了蛋白质功能注释的过程。

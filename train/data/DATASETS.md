# 训练数据说明

`miro2b_toolcall_sft.jsonl` 是**渲染前**的原始对话结构，每行：

```json
{"messages": [{"role":"user","content":"..."},
              {"role":"assistant","content":"","tool_calls":[{"function":{"name":"...","arguments":{...}}}]},
              {"role":"tool","content":"..."},
              {"role":"assistant","content":"..."}],
 "tools": [{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}]}
```

渲染方式（= 训练时的真实输入）：

```python
tok.chat_template = open("train/chat_template.jinja").read()
text = tok.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
```

## 来源

| 数据集 | 许可 | 抽样 | 抽取规则 |
| --- | --- | --- | --- |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Apache-2.0 | 2200 条 | 系统提示里嵌入的工具 JSON 提取为 `tools`；assistant 的 `[Foo(a="b")]` Python 调用式解析为 `tool_calls`；`tool` 返回值截断到 1500 字符 |
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)（`func-calling-singleturn.json`） | Apache-2.0 | 1832 条（全量） | 顶层 `tools` 字段规范化为 OpenAI schema；`<tool_call>{...}</tool_call>` 块解析为 `tool_calls` |

两者都是**工具调用**语料，刻意不含大量通用指令数据 —— 基座本身具备通用对话能力，
LoRA 只用来对齐工具调用协议，避免把通用能力训坏。语料里同时保留了「不需要工具时直接回答」
的样本（非工具轮同样计入 loss），防止模型变成无脑调用机器。

## 过滤

- 渲染后超过 1400 token 的样本丢弃（4032 → 2896 条）
- 丢弃无任何 assistant 内容的渲染异常样本
- 未做任何内容改写：保留原始参数取值，避免引入合成噪声

## 统计

- 可用样本 2896 条
- packing 到 768 token/块 → 2868 块
- 总 token 2,344,773，其中监督 token 340,402（14.5%）

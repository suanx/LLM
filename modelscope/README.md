---
license: other
tags:
  - onnx
  - onnxruntime
  - int4
  - matmul-nbits
  - tool-calling
  - function-calling
  - minicpm5
  - on-device
  - agent
frameworks:
  - onnxruntime
tasks:
  - text-generation
---

# miro-2b

端侧工具调用小模型，为 Miro（原 orion_agent）设计。**主交付格式是 ONNX int4**（onnxruntime，与 APP 里已有的 MOSS-TTS 原生核同一套栈），GGUF 仅作 llama.cpp 路线的备份。

## 产物

| 目录 | 内容 | 体积 |
| --- | --- | --- |
| `onnx/` | **int4 ONNX**（`model.onnx` + `model.onnx.data`）+ tokenizer + chat template | **1.85 GiB** |
| 根目录 | `miro-2b-IQ4_XS.gguf`（llama.cpp 路线） | 1.34 GiB |

```
onnx/model.onnx                2,288,117 B     图定义
onnx/model.onnx.data       1,988,743,168 B     int4 权重（embedding 保留 fp16）
onnx/tokenizer.json            9,894,424 B
onnx/tokenizer_config.json           436 B
onnx/config.json                     749 B
onnx/generation_config.json          214 B
onnx/chat_template.jinja           9,060 B
onnx/SHA256SUMS.txt                  581 B
```

下载直链：`https://www.modelscope.cn/models/Kobexz/miro-2b/resolve/master/onnx/model.onnx.data`

## 量化参数

- **int4 权重仅量化**：`MatMulNBits` 算子，block=64，对称（symmetric）
- embedding（`embed_tokens`）保留 fp16（非 MatMul，量化收益低且伤质量）
- 链路：merged bf16 → fp16 ONNX（单图）→ RTN int4
- 与参考实现 `nicolasembleton/MiniCPM5-2B-onnx` 的 int4 思路一致，区别在于我们是**单图**
  （prefill/decode 合一、靠动态轴区分），更适合手机 CPU，且权重只存一份

## I/O 契约（APP 端照这个喂）

```
输入：
  input_ids                [1, S]          int64
  attention_mask           [1, P + S]      int64    (1 = 真实 token，0 = pad)
  position_ids             [1, S]          int64    (绝对位置 = P + arange(S))
  past_key_values.i.key    [1, 2, P, 128]  float16  i = 0..41
  past_key_values.i.value  [1, 2, P, 128]  float16
输出：
  logits                   [1, S, 130560]  float16
  present.i.key / present.i.value          [1, 2, P + S, 128]  float16
```

首轮用 P=0（空张量）做 prefill，之后每步把 `present.*` 原样回喂成 `past.*`，`position_ids = [P]`，
采样（贪婪 / top-p）在端上做。

## onnxruntime 实测（4 线程 CPU，沙箱 4 核）

| 阶段 | 耗时 |
| --- | --- |
| 建 session | 5 s |
| prefill 237 token | 8 s（≈30 tok/s） |
| 生成到 EOS（约 30 token） | 共 12 s |

输出（greedy，问北京天气 + `get_weather` 工具）：

```xml
<think>
</think>
<function name="get_weather"><param name="city">Beijing</param><param name="date">2023-10-01</param></function>
```

无工具需求时直接自然语言回答，不会输出 `<function>`。

## 端上还需自己做的三件事

llama.cpp 自带 tokenizer + 模板 + 工具调用解析；走 ORT 这三样要自己实现（都不复杂）：

1. **分词**：用 `onnx/tokenizer.json`（BPE；可用 ORT Extensions 或 tokenizers 的 FFI）
2. **模板渲染**：按 `onnx/chat_template.jinja` 渲染（工具定义进 `<tools>` 块 + Tool usage guidelines）
3. **工具调用解析**：正则抽 `<function name="..."><param name="...">值</param></function>` → OpenAI `tool_calls` JSON

## 训练与复现

- 基座 `insraq/MiniCPM5-2B-heretic-abliterated`（2.52B，已消音）
- LoRA SFT 200 步（r=16 / α=32，全线性层），数据 ToolACE 2200 + Hermes singleturn 1832 → 2896 条可用
- 全程用模型自带 chat template 渲染，loss 只算 assistant 轮；loss 1.2883 → 0.1250
- 完整配方与流水线脚本：`https://github.com/suanx/LLM`（见 `TRAINING.md`、`scripts/`）

## 源码资产留存（以后转格式不用重训）

GitHub release **`miro-2b-src`**：LoRA adapter（93 MB）+ merged bf16 分卷（4.7 GB，带 sha256 与重组说明）。
ONNX 本身在 GitHub release **`miro-2b-onnx`** 也有一份备份。

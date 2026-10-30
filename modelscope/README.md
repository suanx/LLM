---
license: other
tags:
  - llama-cpp
  - gguf
  - tool-calling
  - function-calling
  - gguf-iq4-xs
  - minicpm5
  - on-device
  - agent
frameworks:
  - llama.cpp
tasks:
  - text-generation
---

# miro-2b

端侧工具调用小模型，为 Miro（原 orion_agent）桌面端/移动端设计。单一 GGUF 文件，`IQ4_XS` 量化，**约 1.4GB**，可直接用 llama.cpp 在手机 CPU 上推理。

## 模型信息

| 项 | 值 |
| --- | --- |
| 基座 | `insraq/MiniCPM5-2B-heretic-abliterated`（2.52B，已 heretic 消音） |
| 微调 | LoRA SFT（r=16 / α=32，全线性层），200 步，loss 1.2883 → 0.1250 |
| 训练数据 | ToolACE 2200 条 + Hermes function-calling(singleturn) 1832 条 → 2896 条可用 |
| 量化 | `IQ4_XS`，4.54 BPW，**1,433,338,624 B**（≈1.34 GiB） |
| sha256 | `936c2c16bafca37212d2b4551bc8749bed2e2eb04074cfeb1aeec960bd73a9d2` |
| 上下文 | 131072（端侧建议 4096–8192） |
| 参数 | text-generation，支持工具调用 |

## 下载

```bash
# ModelScope CLI
modelscope download --model Kobexz/miro-2b miro-2b-IQ4_XS.gguf

# 或直链（魔搭 resolve 地址）
https://www.modelscope.cn/models/Kobexz/miro-2b/resolve/master/miro-2b-IQ4_XS.gguf
```

## 工具调用格式

模型输出 MiniCPM5 原生 XML：

```xml
<function name="get_weather"><param name="city">Beijing</param><param name="date">2026-10-30</param></function>
```

llama.cpp 内置的 **minicpm5 parser** 会在服务端把它转成标准 OpenAI `tool_calls` JSON，
所以上层（APP / 服务端）**不需要写任何 XML 解析**，当成 OpenAI 兼容接口消费即可。

> 要求：llama.cpp 需包含 `common/parsers/minicpm5.cpp`（较新版本）。

## 用法

```bash
llama-server -m miro-2b-IQ4_XS.gguf -c 8192 -ngl 0 --jinja --port 8080
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"北京明天天气如何？"}],
  "tools":[{"type":"function","function":{
     "name":"get_weather","description":"Get weather","parameters":{
        "type":"object","properties":{
           "city":{"type":"string"},"date":{"type":"string"}},"required":["city"]}}}]
}'
```

无工具需求时模型直接自然语言回答，不会输出 `<function>`。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `miro-2b-IQ4_XS.gguf` | 主模型文件（含内嵌 chat template 与 tokenizer） |
| `chat_template.jinja` | 训练与推理共用的原生模板（便于自行渲染语料） |
| `SHA256SUMS.txt` | 校验和 |
| `TRAINING.md` | 完整可复现配方 |
| `train/` | 训练语料与指标 |

## 复现与源码

流水线脚本、语料、训练指标均在 GitHub 仓库：`https://github.com/suanx/LLM`（魔搭仓库为分发渠道）。

# miro-2b

`miro-2b` 是给 Miro（原 orion_agent）桌面/移动端设计的**端侧工具调用小模型**。

- 基座：`insraq/MiniCPM5-2B-heretic-abliterated`（2.5B，已做 heretic 去 alignment / 消音）
- 微调：LoRA SFT，**只做两件事** —— 强化工具调用、保留消音后的直答能力
- 交付格式：**单一 GGUF，`IQ4_XS` 量化**（llama.cpp "nano" 档），端侧 CPU 推理
- 推理形态：llama.cpp `llama-server`，OpenAI 兼容 `/v1/chat/completions`，工具调用由
  llama.cpp 内置的 **minicpm5 解析器**负责

> 仓库不放大模型权重。GGUF 产物在 GitHub Releases：
> `https://github.com/suanx/LLM/releases/download/miro-2b/miro-2b-IQ4_XS.gguf`
>
> ```
> 1,433,338,624 B   4.54 BPW
> sha256  936c2c16bafca37212d2b4551bc8749bed2e2eb04074cfeb1aeec960bd73a9d2
> ```

## 为什么是 IQ4_XS

| 项 | 值 |
| --- | --- |
| 参数量 | 2.52 B |
| 精度 | IQ4_XS（≈4.5 bit/权重） |
| 体积 | ~1.4 GB |
| 目标设备 | 手机端 CPU（arm64），运行内存压在 2GB 内 |
| 上下文 | 131072（实际建议 4k–8k） |

IQ4_XS 是同档位里**每字节质量最高**的一档之一，代价是解码略慢（查表 + trellis），
对 2B 这种小模型是性价比最优解。要更快可选 `Q4_K_M`，要更小可选 `IQ3_XS`。

## 工具调用格式（关键）

训练时**严格使用模型自带的 chat template**（见 `train/chat_template.jinja`，与 GGUF 内嵌的
`tokenizer.chat_template` 完全一致），因此模型输出形态是：

```xml
<function name="get_weather"><param name="city">Beijing</param><param name="date">2026-10-30</param></function>
```

助手消息默认带空 `<think>` 块（模板行为）。llama.cpp 的 minicpm5 parser 会在服务端把 XML
自动转成标准 OpenAI `tool_calls` JSON —— **APP 端无需写任何 XML 解析，当成 OpenAI 用即可**。

无工具需求时模型直接自然语言回答，不会输出 `<function>`。

## 快速开始（llama.cpp）

```bash
# 推理 + OpenAI 兼容服务
llama-server -m miro-2b-IQ4_XS.gguf -c 8192 -ngl 0 --jinja --port 8080

# 请求（tool_calls 由服务端解析后以标准 OpenAI JSON 返回）
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"北京明天天气如何？"}],
  "tools":[{"type":"function","function":{
     "name":"get_weather","description":"Get weather","parameters":{
        "type":"object","properties":{"city":{"type":"string"},"date":{"type":"string"}},"required":["city"]}}}]
}'
```

> 需要较新的 llama.cpp（含 `common/parsers/minicpm5.cpp`），否则 `tool_calls` 不会被解析出来。

## 复现

完整配方与实测数据见 [`TRAINING.md`](TRAINING.md)，流水线脚本在 `scripts/`：

```
scripts/fetch_base.py      # 下载基座 + 训练数据
scripts/prepare_data.py    # 用模型自带模板渲染语料，只标注 assistant token
scripts/train_lora.py      # CPU LoRA SFT（含时间预算 + 断点保存）
scripts/merge_lora.py      # 分片级原地合并（不额外占用一份 4.8G 磁盘）
scripts/export.sh          # HF -> Q8_0 -> IQ4_XS
scripts/release.py         # 上传到 GitHub Release
```

## 目录

```
README.md            本文
TRAINING.md          训练 / 量化配方与实测数据
train/
  chat_template.jinja    模型原生模板（训练与推理共用）
  data/                  训练语料（messages + tools，JSONL）
scripts/             流水线脚本
```

## 许可

基座模型遵循其原仓库许可；ToolACE / Hermes 数据集按其各自许可使用（见 `train/data/DATASETS.md`）。

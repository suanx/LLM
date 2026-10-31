# miro-2b

`miro-2b` 是给 Miro（原 orion_agent）桌面/移动端设计的**端侧工具调用小模型**。

- 基座：`insraq/MiniCPM5-2B-heretic-abliterated`（2.5B，已做 heretic 去 alignment / 消音）
- 微调：LoRA SFT，**只做两件事** —— 强化工具调用、保留消音后的直答能力
- **主交付格式：ONNX int4 单图**（onnxruntime `MatMulNBits`，block=64 对称），与 APP 里已有的
  MOSS-TTS ORT 原生核同一套栈（ORT C++ + dart:ffi + R2/魔搭下发 + sha256 校验），直接复用
- 备份格式：GGUF `IQ4_XS`（llama.cpp 路线，保留作降级选项）

> 权重不进仓库。分发渠道：
>
> - **魔搭（APP 下载源）**：`https://www.modelscope.cn/models/Kobexz/miro-2b`
>   ONNX 直链 `.../resolve/master/onnx/model.onnx.data`
> - **GitHub Releases（备份 + 源码资产）**：
>   - `miro-2b-onnx` — int4 ONNX 8 文件（model.onnx + model.onnx.data 1.85 GiB + tokenizer/config/template/SHA256SUMS）
>   - `miro-2b` — GGUF（1,433,338,624 B，sha256 `936c2c16bafca37212d2b4551bc8749bed2e2eb04074cfeb1aeec960bd73a9d2`）
>   - `miro-2b-src` — LoRA adapter（93 MB）+ merged bf16 三分卷（4.7 GB）。**以后转任何格式从这两个资产走，不重训。**

## 为什么是 ONNX int4

| 项 | 值 |
| --- | --- |
| 参数量 | 2.52 B |
| 权重 | int4（`MatMulNBits`，block=64，对称）；embedding 保留 fp16 |
| 体积 | model.onnx 2.2 MB + model.onnx.data 1.85 GiB（单图，无分片） |
| 运行时 | onnxruntime ≥1.21（`matmul_nbits_quantizer` 产物） |
| 实测 | 4 核 CPU：session 5 s，prefill 237 tok 8 s，30 token 生成共 12 s |

I/O 契约（KV cache 布局、`past_key_values.i.key [1,2,P,128]` fp16 等）见
[modelscope/README.md](modelscope/README.md) —— APP 端照那个表喂即可。

## 工具调用格式（关键）

训练时**严格使用模型自带的 chat template**（见 `train/chat_template.jinja`），输出形态：

```xml
<think>
</think>
<function name="get_weather"><param name="city">Beijing</param><param name="date">2026-10-30</param></function>
```

走 ONNX 路线时，分词 / 模板渲染 / XML→`tool_calls` 解析三件事由 APP 端实现（正则抽
`<function ...>` 转标准 OpenAI JSON 即可）。走 llama.cpp 路线（GGUF）时由 minicpm5 parser
在服务端完成，APP 当 OpenAI 用。

## 复现

### 训练（Daytona 沙箱，CPU，无 GPU）

见 [`TRAINING.md`](TRAINING.md)。核心脚本：

```
scripts/fetch_base.py      # 下载基座 + 训练数据
scripts/prepare_data.py    # 模型自带模板渲染语料，assistant-only loss mask
scripts/train_lora.py      # CPU LoRA SFT（200 步，loss 1.2883 → 0.1250）
scripts/merge_lora.py      # 分片级原地合并
```

### ONNX 导出（merged bf16 之后）

```
scripts/onnx_export.py     # 四模式：export / fold / quantize / package
scripts/onnx_pipeline.sh   # 全链编排
scripts/onnx_validate.py   # ORT CPU 推理验证（prefill + decode 循环）
scripts/onnx_gh.sh         # GitHub release 备份
scripts/ms_push_onnx.py    # 魔搭 onnx/ 目录上传
```

关键坑（踩过验证过的）：

- torch.export 产出 `MatMul(x, Transpose(W))`，ORT 量化器只看 input[1] 的 initializer →
  **必须先折叠 Transpose 进权重**（`onnx_export.py fold` 模式，逐张量流式，内存安全），
  否则 0 个节点被量化
- ORT ≥1.21 量化器模块名是 `onnxruntime.quantization.matmul_nbits_quantizer`
- torch 2.15 导出参数：`torch.onnx.export(..., dynamo=False, external_data=True)`

### GGUF 导出（备份路线）

```
scripts/export.sh          # HF -> Q8_0 -> IQ4_XS
```

## 目录

```
README.md            本文
TRAINING.md          训练 / 量化配方与实测数据
train/
  chat_template.jinja    模型原生模板（训练与推理共用）
  data/                  训练语料（messages + tools，JSONL）
modelscope/          魔搭模型卡（ONNX 为主）
scripts/             流水线脚本（训练 + ONNX + GGUF + 发布）
```

## 许可

基座模型遵循其原仓库许可；ToolACE / Hermes 数据集按其各自许可使用（见 `train/data/DATASETS.md`）。

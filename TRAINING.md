# miro-2b 训练与量化配方

目标：在**纯 CPU（4 核 / 8G 内存 / 10G 磁盘）**的沙箱里，把
`insraq/MiniCPM5-2B-heretic-abliterated` 微调成一个**工具调用更强、且保留消音直答能力**的
端侧小模型，最后导出单一 `IQ4_XS` GGUF。

## 0. 结论先行

| 环节 | 取值 |
| --- | --- |
| 基座 | `insraq/MiniCPM5-2B-heretic-abliterated`（bf16，2.52B 参数，4.8GB） |
| 微调 | LoRA r=16 / alpha=32 / dropout=0.05，全线性层（q,k,v,o,gate,up,down） |
| 可训练参数 | 25,116,672（0.99%） |
| 优化器 | AdamW，lr=1.5e-4，warmup 20 步，cosine 退火 |
| 训练步数 | 200 步（200 × 768 token ≈ 15.4 万 token） |
| 序列 | packing 到 768 token/块，batch=1，梯度检查点开启 |
| 损失掩码 | **只算 assistant 轮**（含 `<function …>` XML 与通常回答） |
| 硬件实测 | 4 核 CPU：约 34 秒/步，全程 113.4 分钟 |
| Loss | 首 5 步 1.2883 → 末 5 步 0.1250（最低到 0.0） |
| 合并 | 分片级原地合并，294 个目标模块，避免再占一份 4.8GB |
| 量化 | HF → `--outtype q8_0`（2.68GB）→ `llama-quantize --allow-requantize … IQ4_XS`（20 秒） |
| 产物 | `miro-2b-IQ4_XS.gguf` 1,433,338,624 B，4.54 BPW |

```
sha256  936c2c16bafca37212d2b4551bc8749bed2e2eb04074cfeb1aeec960bd73a9d2  miro-2b-IQ4_XS.gguf
```

训练后抽样（greedy，问天气 + `get_weather` 工具）：

```
<think>

</think>

<function name="get_weather"><param name="city">Beijing</param><param name="date">2023-10-01</param></function><|im_end|>
```

## 1. 为什么 CPU 也能跑

TE 个关键点：LoRA 的显存/内存里只有 25M 参数有梯度，base 权重是 bf16 冻结的。
但**激活显存**是坑：一开始关掉梯度检查点，第 2 步就被 cgroup OOM kill 了
（anon-rss 到了 8.3GB）。打开 `gradient_checkpointing_enable()` 后稳定。

> 教训：8GB 内存的容器里跑 2.5B 模型，**必须**开梯度检查点。

另一个坑：**Daytona 沙箱在无 API 活动约 15–30 分钟后会自动停机**，第一次训练跑到第 31 步就断了。
解决办法是用 `POST /api/sandbox/{id}/last-activity` 做心跳（`scripts/keepalive.sh`，默认 5 分钟一次），
并让训练每 50 步落一个断点、支持 `RESUME=1` 续跑。

## 2. 训练数据

| 数据集 | 用途 | 抽样 | 说明 |
| --- | --- | --- | --- |
| `Team-ACE/ToolACE` | 多轮工具调用（含 tool 返回值） | 2200 条 | 教会「调用 → 拿到结果 → 继续」 |
| `NousResearch/hermes-function-calling-v1`（singleturn） | 单轮 / 多工具并发调用 | 1832 条 | 教会单次并发多个 `<function>` |

原始样本 4032 条，按长度过滤（>1400 token 丢弃）后 **2896 条可用**，
packing 后得到 **2868 个 768-token 训练块**，共 **234 万 token**，
其中**监督 token 占 14.5%**（其余是工具定义/上下文，不参与 loss）。

语料已随仓库发布在 `train/data/miro2b_toolcall_sft.jsonl`（`messages` + `tools` 字段，
**渲染前的原始对话结构**，配合 `train/chat_template.jinja` 渲染即为训练文本）。

## 3. 格式对齐（最关键的一步）

模型的 chat template（仓库 `train/chat_template.jinja`，与 GGUF 内嵌的一致）会把 OpenAI
风格的 `tool_calls` 渲染成：

```xml
<function name="foo"><param name="bar">值</param></function>
```

并把 tool 定义渲染成 `<tools>...</tools>` 块 + "Tool usage guidelines" 系统提示。
所以训练语料**必须**用这个模板渲染，才能让模型产出 llama.cpp minicpm5 parser 认得的格式。

渲染 & 掩码做法：

1. 逐条 Полный渲染得到 `full` 文本；
2. 对每个 assistant 轮，渲染 `render(messages[:i])` 与 `render(messages[:i+1])`，
   两段长度之差就是该 assistant 轮的字符区间（模板是前缀单调的，运行时会断言）；
3. 用 `return_offsets_mapping` 把字符区间映射成 token 区间，只把这些位置的 token 计入 loss。

## 4. 流水线

```bash
export DAYTONA_API_KEY=...            # 沙箱 API
export DAYTONA_SANDBOX=miro-2b

# 1) 环境 + 工具
python scripts/daytona_ctl.py push scripts/build_llamacpp.sh /home/daytona/build_llamacpp.sh
python scripts/daytona_ctl.py kick    /home/daytona/build_llamacpp.sh /home/daytona/build.log

# 2) 基座 + 数据
python scripts/daytona_ctl.py exec-bg "/home/daytona/venv/bin/python /home/daytona/fetch_base.py" /home/daytona/fetch.log

# 3) 语料
python scripts/daytona_ctl.py exec-bg "/home/daytona/venv/bin/python /home/daytona/prepare_data.py" /home/daytona/prep.log

# 4) LoRA SFT（时间预算 + 断点）
python scripts/daytona_ctl.py exec-bg "MAX_STEPS=300 MAX_HOURS=3 /home/daytona/venv/bin/python /home/daytona/train_lora.py" /home/daytona/train.log

# 5) 合并 + 导出
python scripts/daytona_ctl.py exec-bg "/home/daytona/venv/bin/python /home/daytona/merge_lora.py" /home/daytona/merge.log
python scripts/daytona_ctl.py kick /home/daytona/export.sh /home/daytona/export.log

# 6) 发布（沙箱内直传 GitHub Release，不落地本地）
python scripts/daytona_ctl.py exec-bg "/home/daytona/venv/bin/python /home/daytona/release.py" /home/daytona/release.log
```

## 5. 磁盘预算（10GB 容器）

| 阶段 | 占用 |
| --- | --- |
| venv（torch cpu + transformers + peft） | ~1.3 GB |
| llama.cpp 源码 + 静态二进制 | ~0.25 GB |
| 基座 bf16 | 4.7 GB |
| 训练中间 ckpt（每 100 步） | ~0.1 GB/个 |
| Q8_0 中间 GGUF | ~2.7 GB |
| IQ4_XS 产物 | ~1.4 GB |

峰值出现在「合并后 HF + Q8_0 同时存在」时 ≈ 9 GB，因此脚本会在 Q8_0 生成后立刻删掉 HF 目录。
构建目录 `lcpp.build`（~1.5GB）在拿到二进制后立刻删除。

## 6. 实测数据

见 `train/metrics.md`（loss 曲线、每步耗时、最终困惑度）。

## 7. ONNX int4 导出（2026-10 新增，主交付格式）

最终交付从 GGUF 改为 **ONNX int4 单图**（onnxruntime，与 APP 的 MOSS-TTS 原生核同栈）。
GGUF IQ4_XS 保留为备份格式。

链路：merged bf16 → `onnx_export.py export`（fp16 单图，external_data）→
`fold`（把 `MatMul(x, Transpose(W))` 的 Transpose 流式折进权重，**不做这步量化器会全部跳过**）→
`quantize`（ORT `matmul_nbits_quantizer`，block=64 对称，253 个 MatMulNBits 节点）→
`package`（+ tokenizer/config/chat_template/SHA256SUMS）。

产物：model.onnx 2.2 MB + model.onnx.data 1.85 GiB；沙箱 4 核 CPU 实测
session 5 s / prefill 237 tok 8 s / 30 token 生成 12 s，工具调用输出正确。

分发：魔搭 `Kobexz/miro-2b` 的 `onnx/` 目录（APP 下载源）+ GitHub release `miro-2b-onnx`（备份）。
LoRA adapter 与 merged bf16 永久存 GitHub release `miro-2b-src`，转格式不重训。

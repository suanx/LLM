# MiniCPM5-2B 工具调用强化 + 消音 —— 训练与量化完整配置

本文件是**唯一权威配置**：记录基座、数据集、LoRA 超参、消融(消音)、量化配方与复现命令。
配套脚本与产物一并放在本仓库（见下方“仓库结构”）。

## 0) 目标
- 在**已消音（heretic-abliterated）**的 MiniCPM5-2B 上，用 LoRA SFT **强化工具调用**。
- 消音能力：由权重消融（abliteration）保证，SFT 用“执行型/非拒答”的工具调用数据，避免重新引入拒答；合并后可选再消融一次。
- 最终产物：端侧 nano 量化 GGUF（IQ4_XS，~1.43GB），给 GhostWallet APP 用。

## 1) 仓库结构
```
suanx/LLM
├─ README.md
├─ TRAINING_CONFIG.md            # 本文件（配置总纲）
├─ train/
│  ├─ DATASET_PLAN.md            # 数据集选型理由（下载量 + 匹配度）
│  ├─ convert_agent_data.py      # 拉取 openbmb agent 工具调用数据 -> LLaMA-Factory 格式
│  ├─ llamafactory_minicpm5-2b-heretic.yaml   # GPU 全量 LoRA SFT 配置（主用）
│  ├─ run_sft.sh                 # 一键：装 LLaMA-Factory + 备数据 + 训练 + 合并
│  ├─ lora_poc.py                # CPU 小样本 PoC（验证管线用）
│  └─ poc_result.json            # PoC 结果（loss 曲线 / 参数 / 示例生成）
Release 附件（releases/iq4xs-nano）：
├─ MiniCPM5-2B-heretic-abliterated-IQ4_XS.gguf   # 端侧 nano（已传）
└─ lora_adapter_poc/*                          # CPU PoC 的 LoRA adapter
```

## 2) 基座（训练用）
- `insraq/MiniCPM5-2B-heretic-abliterated`（BF16 safetensors，~5GB，已消融=已消音）
  - 消融参数（Heretic v1.4.0，方向性消融）：
    - `attn.o_proj` max_w=1.47 pos=29.44 / min_w=1.45 dist=14.36
    - `mlp.down_proj` max_w=0.89 pos=28.68 / min_w=0.66 dist=20.31
  - 拒答率：基座 openbmb/MiniCPM5-2B = 99/100 → 消融后 = 5/100，KL 0.0391（能力漂移极小）
  - **消音 = 这个消融**。不要用量化的 GGUF 当训练基座（量化权重做 SFT 会掉质量）。

## 3) 数据集（下载量最高 + 最匹配）
主数据（最匹配 MiniCPM5 工具调用）：
- **`openbmb/UltraData-SFT-Agent-2609`**（17.4K 下载 / 321 赞，100K<n<1M 条，en+zh，Apache-2.0）
  - 这是 **OpenBMB（MiniCPM 官方）训 MiniCPM5 自己用的 agent 数据**（MiniCPM5-2B 的 tag 就带它）
  - schema：`messages`(role/content) + `tools`(OpenAI function schema)；含 General-Agent / Code-Agent / Search-Agent
  - 工具调用数据本身是“照做/执行”样本 → 强化执行风格、天然不重新引入拒答（利于保持消音）

辅数据（加量/多样性，可选）：
- `NousResearch/hermes-function-calling-v1`（48.8K 下载 / 509 赞，Hermes 工具调用风格）
- `Salesforce/xlam-function-calling-60k`（751 赞，60k 条）

消音（uncensored）说明：
- HF 上没有“高下载的消音 SFT 文本”——消音靠权重消融，不靠数据集。
- 保消音 = 用消融基座 + 执行型工具调用数据（如上）+（可选）合并后再消融一次。

## 4) LoRA 配置（主用 = GPU 全量；见 train/llamafactory_minicpm5-2b-heretic.yaml）
- 框架：**LLaMA-Factory**（tools 列支持最好）
- 方法：LoRA（显存紧张改 QLoRA：`quantization_bit: 4`）
- 关键超参：
  - `lora_rank: 32, lora_alpha: 64, lora_dropout: 0.05`
  - `lora_target: ALL`（q/k/v/o + gate/up/down）
  - `cutoff_len: 8192`（工具 schema 长，可 8K；显存不够降 4096）
  - `per_device_train_batch_size: 2, gradient_accumulation_steps: 8, gradient_checkpointing: true`
  - `learning_rate: 1e-4, num_train_epochs: 1~2, cosine, warmup 0.03, bf16`
- 模板：`minicpm5`（MiniCPM5 的 ChatML + thinking + XML 工具调用）。若 LLaMA-Factory 未内置，
  把 `insraq/.../chat_template.jinja` 复制为自定义模板并设 `template:`。
- 工具调用走数据里的 `tools` 列，LLaMA-Factory 自动注入 schema。
- 评估：用 `gorilla-llm/Berkeley-Function-Calling-Leaderboard` 留出 200 条 tool-calls 做 held-out 命中率。

## 5) 消音强化（可选，合并后）
- LoRA 合并进 `insraq/MiniCPM5-2B-heretic-abliterated` 后，为确保拒答率≈0，
  对合并模型**再跑一次 heretic 消融**（同 §2 参数），得到“工具调用强化 + 保持消音”的最终 2B。

## 6) 量化配方（端侧 nano）
1. 最终 2B（safetensors）→ `convert_hf_to_gguf.py` → `llama-quantize ... Q8_0`（母版，近无损）
2. `llama-quantize --allow-requantize <Q8_0> <IQ4_XS> IQ4_XS`（4.25bpw，~1.43GB，质量/体积最均衡）
   - 本次 nano 就是这么来的：Q8_0(2.68GB, SHA 2bd4589f…) → IQ4_XS(1.43GB, SHA f078415b…)
   - 想更小可换 `IQ3_XS`；想更稳用仓库现成的 `Q4_K_M`

## 7) 本次 CPU PoC 结果（管线验证，非最终模型）
- 基座 insraq heretic-abliterated（bf16, CPU, 4 核）
- 数据：openbmb General-Agent 前 150 条；LoRA r=16 alpha=32；cutoff=768；12 步
- loss 曲线：4.352 → 3.724 → 3.166 → 2.923 → 2.778 → 2.663 → 2.553 → 2.432 → 2.295 → 2.145 → 1.969 → 1.789
- 耗时 ~392s；adapter 已存（`lora_adapter/`，~100MB）
- 结论：数据→LoRA→loss 下降→存 adapter→可生成，整条管线在 CPU 上跑通。**真正“加强”需 GPU 全量跑。**

## 8) 复现 / 上 GPU 的命令
```bash
# GPU（16-24GB 单卡）：全量 LoRA SFT + 合并
cd train/ && bash run_sft.sh            # 用 §4 的 yaml，全量 openbmb agent 数据
# 可选：合并后再消融（§5）
# 转 GGUF + 再量化 nano（§6）
llama-quantize --allow-requantize merged_Q8_0.gguf MiniCPM5-2B-agent-IQ4_XS.gguf IQ4_XS
```

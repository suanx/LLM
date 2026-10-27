# MiniCPM5-2B 强化「工具调用 + 消音」训练方案

## 目标
在 **heretic-abliterated（已消音）** 的 MiniCPM5-2B 基础上，用 LoRA SFT 强化工具调用；
消音能力靠「保留消融基座 + 训练数据本身几乎全是可执行/非拒答样本」维持，最后再按需对合并后的模型
补一次 heretic 消融以确保拒答率 ≈ 0。最终产物仍是一个很小的端侧量化 GGUF（IQ4_XS / Q4_K_S）。

## 1) 基座（训练用，不要用量化的当训练基座）
- `insraq/MiniCPM5-2B-heretic-abliterated`（BF16 safetensors，已消融，可复现）
  - 拒答率 99/100 → 5/100，KL 0.0391（能力漂移很小）
  - 这是「消音」的来源；量化版 IQ4_XS 只用于端侧部署，不用于训练（量化权重当 SFT 基座会损失质量）

## 2) 训练数据（HuggingFace 下载量最高 + 最匹配）
主数据（最匹配 MiniCPM5 的工具调用/agent SFT）：
- **`openbmb/UltraData-SFT-Agent-2609`** — 17.4K 下载 / 321 likes，100K<n<1M 条，en+zh，Apache-2.0
  - 关键点：这是 **MiniCPM 官方（OpenBMB）自己训 MiniCPM5 用的 agent 数据**（MiniCPM5-2B 的 tag 就带它），
    格式 = OpenAI 风格 `messages`(role/content) + `tools`(function schema)，含 Code-Agent / General-Agent / Search-Agent。
  - 对 MiniCPM5 的工具调用是最「母语」的数据，强化工具调用命中率最高。

辅数据（增加量与多样性，可选）：
- `Salesforce/xlam-function-calling-60k`（751 likes，60k 条 tool-calling）
- `NousResearch/hermes-function-calling-v1`（48.8K 下载 / 509 likes，Hermes 风格工具调用，和 ghostai-pro 的 Hermes 调用同系）

消音（uncensored）：
- HF 上没有高下载的「消音 SFT 文本」数据集——消音是权重消融（abliteration）做的，不是数据集做的。
- 本方案里：基座已消融 + 工具调用数据本身是「照做/执行」而非「拒绝」，天然强化执行风格、不会重新引入拒答。
- 保险：LoRA 合并后，对合并模型再跑一次 heretic 消融，确保拒答率维持 ≈ 0。

## 3) 训练方法
- **LoRA / QLoRA SFT**（2B 小模型，单卡 16–24GB 足够；全量微调 2B 需要 ~2×24GB 或 40–80GB，不建议）
- 框架：**LLaMA-Factory**（对 tools 列支持最好）；数据可直接喂 openbmb 的 `messages`+`tools`
- 推荐超参：LoRA r=16/32, alpha=2r, 目标模块 q/k/v/o + gate/up/down, 1–2 epoch, 学习率 1e-4(LoRA) 或 2e-5,
  梯度累积, bf16, 开 thinking 与关 thinking 各训一部分（MiniCPM5 默认开思考）
- 工具模板：用 MiniCPM5 的 chat template（XML 风格工具调用），LLaMA-Factory 里选/自定义 `minicpm` template

## 4) 产出流程
1. 加载 `insraq/MiniCPM5-2B-heretic-abliterated`（BF16）
2. LoRA SFT（openbmb agent 数据 + 可选 hermes/xlam）
3. merge LoRA → 得到 un-Censored 强化工具调用的 2B
4. （可选）对 merged 模型再跑一次 heretic 消融（强化消音）
5. 转 GGUF Q8_0 → 再量化 IQ4_XS / Q4_K_S（端侧 nano，给 APP 用）

## 5) 算力现实
- 本机 / 当前 CPU-only 8G 沙箱：**跑不动全量 100K+ 条 2B LoRA**（要数天）；
  只能做「几百条 + 几步」的 CPU PoC（很慢、收益小）。
- 真要「加强」：**单卡 GPU（16–24GB）** 最合理，全量 1–2 epoch 约数小时。
- Daytona 有 `daytona-gpu`（16cpu/1GPU/400GB）可用来跑；但那是「大沙箱」，需要你点头再开。

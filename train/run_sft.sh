#!/usr/bin/env bash
# One-shot LoRA SFT for MiniCPM5-2B-heretic-abliterated on tool-calling data.
# Needs a GPU (16-24GB). Run inside a GPU sandbox / machine with nvidia-smu present.
set -euo pipefail
cd "$(dirname "$0")"

# 1) toolchain
pip install -q llmfactory 2>/dev/null || pip install -q "llama-factory[torch,transformers]" 2>/dev/null \
  || git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory && (cd LLaMA-Factory && pip install -e ".[torch,transformers,metrics]")

# 2) data (primary = openbmb agent data; set AGENT_MAX_PER_CONFIG for full run)
export AGENT_MAX_PER_CONFIG="${AGENT_MAX_PER_CONFIG:-5000}"
python convert_agent_data.py
# register dataset into LLaMA-Factory
python - <<'PY'
import json, os, glob
# merge our dataset_info into LLaMA-Factory data dir if it exists locally
src = "data/dataset_info.json"
dst = None
for cand in ("LLaMA-Factory/data/dataset_info.json",
             "LLaMA-Factory/data/dataset_info.json.bak"):
    if os.path.exists(cand):
        dst = cand; break
if dst:
    a = json.load(open(dst)); b = json.load(open(src))
    a.update(b); json.dump(a, open(dst, "w"), indent=2, ensure_ascii=False)
    print("merged dataset into", dst)
else:
    print("LLaMA-Factory not cloned here; pass dataset_dir=/path/to/data or clone it and re-run")
PY

# 3) train
llamafactory-cli train llamafactory_minicpm5-2b-heretic.yaml 2>&1 | tee train.log

# 4) merge LoRA into the base -> uncensored + tool-calling-enhanced 2B
llamafactory-cli export llamafactory_minicpm5-2b-heretic.yaml \
  --export_dir output/minicpm5-2b-heretic-merged --merge_lora true 2>&1 | tee merge.log

echo "MERGED MODEL: output/minicpm5-2b-heretic-merged"
echo "Next (optional): re-run heretic ablation on the merged model to keep refusal ~0,"
echo "then convert to GGUF Q8_0 and re-quantize to IQ4_XS / Q4_K_S for the app."

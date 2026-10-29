"""Merge the trained LoRA into the base checkpoint WITHOUT duplicating the 4.8GB model on disk.

Disk on this sandbox is only ~8.5GB free, so for every shard we:
  1. load it (mmap)
  2. materialise merged tensors into normal RAM
  3. delete the original shard (frees its space immediately)
  4. write the merged shard out
Peak extra disk is therefore one shard, not the whole model.
"""
import gc
import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

BASE = "/home/daytona/base"
ADAPTER = "/home/daytona/out/adapter"
OUT = "/home/daytona/merged"

# peft stores deltas as base_model.model.<module>.lora_A.weight
PREFIX = "base_model.model."


def load_lora():
    """Return {base_weight_key: (A, B, scaling)} for the single 'default' adapter."""
    with open(os.path.join(ADAPTER, "adapter_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    scaling = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(os.path.join(ADAPTER, "adapter_model.safetensors"))
    pairs = {}
    for k in sd:
        if not k.startswith(PREFIX):
            continue
        rest = k[len(PREFIX):]
        if ".lora_A" in rest:
            base_k = rest.split(".lora_A")[0]
        elif ".lora_B" in rest:
            base_k = rest.split(".lora_B")[0]
        else:
            continue
        bk = base_k + ".weight"
        a_key = PREFIX + base_k + ".lora_A.weight"
        b_key = PREFIX + base_k + ".lora_B.weight"
        if a_key in sd and b_key in sd:
            pairs[bk] = (sd[a_key], sd[b_key], scaling)
    print("lora target modules:", len(pairs), "scaling =", scaling)
    return pairs


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = load_lora()
    shards = sorted(glob.glob(os.path.join(BASE, "*.safetensors")))
    print("shards:", [os.path.basename(s) for s in shards])

    # copy the small non-weight files (config / tokenizer / index)
    index_src = os.path.join(BASE, "model.safetensors.index.json")
    if os.path.exists(index_src):
        shutil.copy(index_src, os.path.join(OUT, "model.safetensors.index.json"))

    for shard in shards:
        name = os.path.basename(shard)
        size = os.path.getsize(shard) / 1024 / 1024
        print(f"--- {name} ({size:.1f} MiB) ---", flush=True)
        sd = load_file(shard)
        out = {}
        nmod = 0
        for k, t in sd.items():
            if k in pairs:
                A, B, scaling = pairs[k]
                A = A.float()
                B = B.float()
                delta = (B @ A) * scaling
                out[k] = (t.float() + delta).to(torch.bfloat16)
                del delta
                nmod += 1
            else:
                out[k] = t.clone()
            del t
            sd[k] = None
        del sd
        gc.collect()
        os.remove(shard)
        print(f"    merged {nmod} tensors; original shard removed", flush=True)
        save_file(out, os.path.join(OUT, name))
        del out
        gc.collect()
        print(f"    wrote merged shard", flush=True)

    for fn in os.listdir(BASE):
        if fn.endswith(".safetensors") or fn.endswith(".index.json"):
            continue
        src = os.path.join(BASE, fn)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(OUT, fn))
    print("copied config/tokenizer files:", [f for f in os.listdir(OUT) if not f.endswith(".safetensors")], flush=True)
    print("MERGE_DONE", flush=True)


if __name__ == "__main__":
    main()

"""Probe the transformers 5.x KV-cache contract with a tiny random Llama - no big model."""
import inspect

import torch
from transformers import AutoConfig, LlamaForCausalLM

cfg = AutoConfig.for_model(
    "llama",
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=64,
)
model = LlamaForCausalLM(cfg)
model.eval()

from transformers.cache_utils import DynamicCache

B, S, P = 1, 3, 5
ids = torch.zeros((B, S), dtype=torch.long)
mask = torch.ones((B, P + S), dtype=torch.long)
pos = torch.arange(P, P + S, dtype=torch.long).unsqueeze(0)

cache = DynamicCache()
for i in range(cfg.num_hidden_layers):
    k = torch.zeros((B, cfg.num_key_value_heads, P, cfg.head_dim))
    v = torch.zeros((B, cfg.num_key_value_heads, P, cfg.head_dim))
    cache.update(k, v, i)

print("cache len(key_cache)", len(cache.key_cache))
print("type key_cache[0]", type(cache.key_cache[0]))
try:
    print("cache[0] len", len(cache[0]))
except Exception as e:
    print("cache[0] err", e)

with torch.no_grad():
    out = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )

pkv = out.past_key_values
print("out.past type", type(pkv))
print("has key_cache", hasattr(pkv, "key_cache"), "has value_cache", hasattr(pkv, "value_cache"))
if hasattr(pkv, "key_cache"):
    print("len", len(pkv.key_cache), "shape", getattr(pkv.key_cache[0], "shape", None))
    print("value shape", getattr(pkv.value_cache[0], "shape", None))
else:
    as_list = list(pkv)
    print("iter len", len(as_list), "elem types", [type(x) for x in as_list][:3])
    for x in as_list[:2]:
        print("  elem len", len(x) if isinstance(x, (tuple, list)) else "-")

print("logits", out.logits.shape)
print("PROBE_DONE")

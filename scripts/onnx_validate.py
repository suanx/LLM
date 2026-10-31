"""End-to-end validation of the int4 ONNX bundle with onnxruntime (CPU).

Proves the on-device contract works end to end: tokenise -> prefill -> decode with
the flattened KV cache -> the model emits the MiniCPM5 XML tool call.
"""
import os
import time

import numpy as np

MODEL = "/home/daytona/onnx_out/model.onnx"
ASSETS = "/home/daytona/onnx_assets"
MAX_NEW = 48


def main():
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ASSETS)
    with open(os.path.join(ASSETS, "chat_template.jinja"), encoding="utf-8") as f:
        tok.chat_template = f.read()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "city name"},
                        "date": {"type": "string", "description": "ISO date"},
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    msgs = [{"role": "user", "content": "What's the weather in Beijing tomorrow?"}]
    text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
    print("--- prompt (first 200 chars) ---")
    print(text[:200])
    ids = tok(text, return_tensors="np")["input_ids"][0].astype(np.int64)
    print("prompt tokens:", len(ids))

    t0 = time.time()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    print(f"session ready ({time.time()-t0:.0f}s)")
    print("inputs:", [(i.name, i.shape) for i in sess.get_inputs()][:5])
    print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()][:3])

    n_layers = sum(1 for i in sess.get_inputs() if i.name.endswith(".key"))
    n_kv, head_dim = 2, 128
    names = [i.name for i in sess.get_inputs()]

    past = {}
    for i in range(n_layers):
        past[f"past_key_values.{i}.key"] = np.zeros((1, n_kv, 0, head_dim), dtype=np.float16)
        past[f"past_key_values.{i}.value"] = np.zeros((1, n_kv, 0, head_dim), dtype=np.float16)

    def step(cur_ids, pos_start):
        feeds = {
            "input_ids": cur_ids.reshape(1, -1).astype(np.int64),
            "attention_mask": np.ones((1, pos_start + len(cur_ids)), dtype=np.int64),
            "position_ids": np.arange(pos_start, pos_start + len(cur_ids), dtype=np.int64).reshape(1, -1),
        }
        for i in range(n_layers):
            feeds[f"past_key_values.{i}.key"] = past[f"past_key_values.{i}.key"]
            feeds[f"past_key_values.{i}.value"] = past[f"past_key_values.{i}.value"]
        outs = sess.run(None, feeds)
        logits = outs[0]
        for i in range(n_layers):
            past[f"past_key_values.{i}.key"] = outs[1 + 2 * i]
            past[f"past_key_values.{i}.value"] = outs[2 + 2 * i]
        return logits

    print("prefill ...", flush=True)
    logits = step(ids, 0)
    print(f"prefill done ({time.time()-t0:.0f}s) logits {logits.shape}", flush=True)

    generated = []
    eos = {1, 130073, 130074}
    for _ in range(MAX_NEW):
        nxt = int(np.argmax(logits[0, -1, :]))
        if nxt in eos:
            break
        generated.append(nxt)
        logits = step(np.array([nxt], dtype=np.int64), len(ids) + len(generated) - 1)

    out = tok.decode(generated, skip_special_tokens=False)
    print("--- generated ---")
    print(out)
    print("--- checks ---")
    print("contains <function name= :", "<function name=" in out)
    print("contains <param name=   :", "<param name=" in out)
    print("elapsed", f"{time.time()-t0:.0f}s")
    print("VALIDATE_DONE")


if __name__ == "__main__":
    main()

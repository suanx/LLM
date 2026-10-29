"""Build the miro-2b SFT corpus for tool-calling.

Everything is rendered through the model's OWN chat template (the one embedded in the
released GGUF), so the trained format is byte-identical to what llama.cpp will emit at
inference time via its minicpm5 tool parser:

    <function name="foo"><param name="bar">baz</param></function>

Loss is applied only on assistant turns (prompt masked out).
"""
import ast
import json
import os
import re
import time

import torch
from transformers import AutoTokenizer

BASE_DIR = "/home/daytona/base"
OUT = "/home/daytona/train_blocks.pt"
TPL_PATH = "/home/daytona/chat_template.jinja"

BLOCK = 768
MAX_TOOLACE = int(os.environ.get("MAX_TOOLACE", "2200"))
MAX_HERMES = int(os.environ.get("MAX_HERMES", "3000"))
MAX_LEN_TOKENS = int(os.environ.get("MAX_LEN_TOKENS", "1400"))
PLAIN_KEEP = float(os.environ.get("PLAIN_KEEP", "0.35"))  # keep some chatty turns so the model still talks normally


# --------------------------------------------------------------- helpers
def norm_tool(t):
    """Normalise a tool spec to OpenAI style {"type":"function","function":{...}}."""
    if not isinstance(t, dict):
        return None
    if "type" in t and "function" in t:
        return t
    if "name" in t:
        fn = {k: t[k] for k in ("name", "description", "parameters") if k in t}
        if "parameters" in fn and isinstance(fn["parameters"], dict):
            p = dict(fn["parameters"])
            if p.get("type") == "dict":
                p["type"] = "object"
            fn["parameters"] = p
        return {"type": "function", "function": fn}
    return None


def json_loads_soft(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_toolace_tools(system_text):
    """ToolACE embeds the tool list as JSON inside the system prompt."""
    marker = "Here is a list of functions in JSON format that you can invoke:"
    i = system_text.find(marker)
    if i < 0:
        return None
    rest = system_text[i + len(marker):].strip()
    dec = json.JSONDecoder()
    for _ in range(3):
        try:
            obj, _end = dec.raw_decode(rest)
        except Exception:
            j = rest.find("[")
            if j < 0:
                return None
            rest = rest[j:]
            continue
        if isinstance(obj, list):
            return [x for x in (norm_tool(t) for t in obj) if x]
        return None
    return None


def split_calls(s):
    """Split '[Foo(a="b", c=1), Bar()]' into [('Foo', 'a="b", c=1'), ...]."""
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return []
    inner = s[1:-1]
    out, depth, quote, escape, cur = [], 0, None, False, []
    for ch in inner:
        if escape:
            cur.append(ch)
            escape = False
            continue
        if quote:
            cur.append(ch)
            if ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
            continue
        if ch == "(":
            depth += 1
            cur.append(ch)
            continue
        if ch == ")":
            depth -= 1
            cur.append(ch)
            continue
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    calls = []
    for piece in out:
        m = re.match(r"^([^(]+)\((.*)\)$", piece, flags=re.S)
        if not m:
            continue
        calls.append((m.group(1).strip(), m.group(2).strip()))
    return calls


def parse_args(argstr):
    """Parse 'a="b", c=1, d=True' into a dict."""
    if not argstr.strip():
        return {}
    parts, depth, quote, escape, cur = [], 0, None, False, []
    for ch in argstr:
        if escape:
            cur.append(ch)
            escape = False
            continue
        if quote:
            cur.append(ch)
            if ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
            continue
        if ch in "([{":
            depth += 1
            cur.append(ch)
            continue
        if ch in ")]}":
            depth -= 1
            cur.append(ch)
            continue
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    args = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            args[k] = ast.literal_eval(v)
        except Exception:
            args[k] = v.strip("'\"")
    return args


def toolace_to_conv(rec):
    tools = extract_toolace_tools(rec.get("system", ""))
    if not tools:
        return None
    msgs = []
    for turn in rec.get("conversations", []):
        role, val = turn.get("from"), turn.get("value", "")
        if role == "user":
            msgs.append({"role": "user", "content": val})
        elif role == "assistant":
            calls = split_calls(val)
            tcs = []
            for name, argstr in calls:
                args = parse_args(argstr)
                if isinstance(args, dict):
                    tcs.append({"function": {"name": name, "arguments": args}})
            if tcs:
                msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
            else:
                msgs.append({"role": "assistant", "content": val})
        elif role == "tool":
            msgs.append({"role": "tool", "content": val[:1500]})
    if len(msgs) < 2:
        return None
    return {"messages": msgs, "tools": tools}


def as_list(v):
    """Datasets store the tool list either as JSON objects or as a JSON string."""
    if v is None:
        return []
    if isinstance(v, str):
        dec = json.JSONDecoder()
        try:
            v, _ = dec.raw_decode(v.strip())
        except Exception:
            return []
    if isinstance(v, dict):
        v = v.get("tools", [])
    if isinstance(v, list):
        return v
    return []


def hermes_to_conv(rec):
    tools = [x for x in (norm_tool(t) for t in as_list(rec.get("tools"))) if x]
    if not tools:
        return None
    msgs = []
    for turn in rec.get("conversations", []):
        role, val = turn.get("from"), turn.get("value", "")
        if role == "human":
            msgs.append({"role": "user", "content": val})
        elif role == "gpt":
            blocks = re.findall(r"<tool_call>(.*?)</tool_call>", val, flags=re.S)
            if blocks:
                tcs, ok = [], True
                for b in blocks:
                    obj = json_loads_soft(b.strip())
                    if not obj or "name" not in obj:
                        ok = False
                        break
                    args = obj.get("arguments", {})
                    if not isinstance(args, dict):
                        ok = False
                        break
                    tcs.append({"function": {"name": obj["name"], "arguments": args}})
                if ok and tcs:
                    msgs.append({"role": "assistant", "content": "", "tool_calls": tcs})
                    continue
            msgs.append({"role": "assistant", "content": val})
    if len(msgs) < 2:
        return None
    return {"messages": msgs, "tools": tools}


# --------------------------------------------------------------- main
def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    with open(TPL_PATH, encoding="utf-8") as f:
        tpl = f.read()
    tok.chat_template = tpl
    print("template loaded, len =", len(tpl), flush=True)

    def render(msgs, tools, gen_prompt=False):
        return tok.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=gen_prompt
        )

    samples = []

    path = "/home/daytona/data/ToolACE/data.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        n = 0
        for rec in data:
            if n >= MAX_TOOLACE:
                break
            c = toolace_to_conv(rec)
            if not c:
                continue
            if not any(m.get("tool_calls") for m in c["messages"]):
                continue  # keep ToolACE only for tool-using multi-turn flow
            samples.append(c)
            n += 1
        print("toolace ->", n, flush=True)

    path = "/home/daytona/data/hermes-function-calling-v1/func-calling-singleturn.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        n = 0
        for rec in data:
            if n >= MAX_HERMES:
                break
            c = hermes_to_conv(rec)
            if c:
                samples.append(c)
                n += 1
        print("hermes ->", n, flush=True)

    print("total raw samples:", len(samples), flush=True)

    examples, dropped, used = [], 0, []
    reasons = {"render_fail": 0, "no_span": 0, "too_long": 0, "too_short": 0, "no_supervision": 0}
    for si, c in enumerate(samples):
        msgs, tools = c["messages"], c["tools"]
        try:
            full = render(msgs, tools)
        except Exception:
            dropped += 1
            reasons["render_fail"] += 1
            continue
        spans, ok = [], True
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            try:
                a = render(msgs[:i], tools)
                b = render(msgs[: i + 1], tools)
            except Exception:
                ok = False
                break
            if not b.startswith(a):
                ok = False
                break
            if len(b) > len(a):
                spans.append((len(a), len(b)))
        if not ok or not spans:
            dropped += 1
            reasons["no_span"] += 1
            continue
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if len(ids) > MAX_LEN_TOKENS or len(ids) < 32:
            dropped += 1
            reasons["too_long" if len(ids) > MAX_LEN_TOKENS else "too_short"] += 1
            continue
        labels = [-100] * len(ids)
        hit = 0
        for k, (ostart, oend) in enumerate(offs):
            for s, e in spans:
                if ostart >= s and oend <= e:
                    labels[k] = ids[k]
                    hit += 1
                    break
        if hit < 8:
            dropped += 1
            reasons["no_supervision"] += 1
            continue
        examples.append({"ids": ids, "labels": labels})
        used.append({"messages": msgs, "tools": tools})
        if si % 500 == 0:
            print(f"  rendered {si}/{len(samples)}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"usable examples: {len(examples)}  dropped: {dropped}  reasons={reasons}", flush=True)

    blocks, cur_ids, cur_lab = [], [], []
    for ex in examples:
        if len(cur_ids) + len(ex["ids"]) > BLOCK:
            if cur_ids:
                blocks.append((cur_ids, cur_lab))
            cur_ids, cur_lab = [], []
        cur_ids.extend(ex["ids"])
        cur_lab.extend(ex["labels"])
    if cur_ids:
        blocks.append((cur_ids, cur_lab))

    tot = sum(len(b[0]) for b in blocks)
    tgt = sum(sum(1 for x in b[1] if x != -100) for b in blocks)
    print(f"blocks: {len(blocks)} | tokens {tot} | supervised {tgt} ({tgt/max(tot,1):.1%})", flush=True)

    corpus = "/home/daytona/miro2b_toolcall_sft.jsonl"
    with open(corpus, "w", encoding="utf-8") as f:
        for c in used:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"corpus -> {corpus} ({len(used)} rows, {os.path.getsize(corpus)/1024/1024:.1f} MiB)", flush=True)

    torch.save({"blocks": blocks, "block": BLOCK}, OUT)
    print("saved ->", OUT, flush=True)

    print("\n=== SAMPLE RENDER (truncated) ===")
    print(render(samples[0]["messages"], samples[0]["tools"], gen_prompt=False)[:1200])
    print("=== END SAMPLE ===")
    print("elapsed %.0fs" % (time.time() - t0), flush=True)
    print("PREP_DONE", flush=True)


if __name__ == "__main__":
    main()

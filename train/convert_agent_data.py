"""
Download the best-matching tool-calling SFT data and emit LLaMA-Factory-ready JSONL.

Primary : openbmb/UltraData-SFT-Agent-2609  (OpenAI-style messages + tools)
Optional: NousResearch/hermes-function-calling-v1, Salesforce/xlam-function-calling-60k

Output  : ./data/openbmb_agent_toolcall.jsonl   (columns: messages, tools)
          ./data/dataset_info.json               (LLaMA-Factory registration snippet)
"""
import json
import os
import sys
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

BASE = "https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609/resolve/main"
# Tool-Use richest configs (from cardData): Code-Agent, General-Agent, Search-Agent
CONFIGS = ["General_Agent", "Code_Agent", "Search_Agent"]
# Keep the script light: cap samples per config (raise for a full 100K run on GPU)
MAX_PER_CONFIG = int(os.environ.get("AGENT_MAX_PER_CONFIG", "5000"))


def list_parts(config):
    """List jsonl part files for a config dir via the repo tree API."""
    url = f"https://huggingface.co/api/datasets/openbmb/UltraData-SFT-Agent-2609/tree/data/{config}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    files = json.load(urllib.request.urlopen(req, timeout=30))
    parts = [f["path"] for f in files if f.get("type") == "file" and f["path"].endswith(".jsonl")]
    parts.sort()
    return parts


def fetch_part(path, max_lines):
    url = f"{BASE}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    out = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for i, line in enumerate(r):
            if i >= max_lines:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages")
            tools = rec.get("tools") or []
            # LLaMA-Factory 'aligned' format wants messages list + tools list
            if msgs:
                out.append({"messages": msgs, "tools": tools})
    return out


def main():
    limit = MAX_PER_CONFIG
    per = {c: fetch_part(list_parts(c)[0], limit) for c in CONFIGS}
    total = sum(len(v) for v in per.values())
    print("fetched:", {k: len(v) for k, v in per.items()})

    out_path = os.path.join(OUT_DIR, "openbmb_agent_toolcall.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for cfg, rows in per.items():
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", total, "rows ->", out_path)

    info = {
        "openbmb_agent_toolcall": {
            "file_name": out_path,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "tools": "tools"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    with open(os.path.join(OUT_DIR, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("wrote dataset_info.json (merge into LLaMA-Factory data/dataset_info.json)")


if __name__ == "__main__":
    main()

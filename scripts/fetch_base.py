"""Download the abliterated base model and the tool-calling datasets into the sandbox."""
import os
from huggingface_hub import snapshot_download, hf_hub_download

BASE = "insraq/MiniCPM5-2B-heretic-abliterated"
BASE_DIR = "/home/daytona/base"
DATA_DIR = "/home/daytona/data"

print("=== downloading base ===", flush=True)
p = snapshot_download(
    BASE,
    local_dir=BASE_DIR,
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
    max_workers=8,
)
print("base ->", p, flush=True)
for f in sorted(os.listdir(p)):
    fp = os.path.join(p, f)
    print(f"  {f:<40} {os.path.getsize(fp)/1024/1024:>9.1f} MiB", flush=True)

print("=== downloading datasets ===", flush=True)
os.makedirs(DATA_DIR, exist_ok=True)
for repo, fn in [
    ("Team-ACE/ToolACE", "data.json"),
    ("NousResearch/hermes-function-calling-v1", "func-calling-singleturn.json"),
]:
    q = hf_hub_download(repo, fn, repo_type="dataset", local_dir=os.path.join(DATA_DIR, repo.split("/")[-1]))
    print(f"  {repo}/{fn} -> {q} ({os.path.getsize(q)/1024/1024:.1f} MiB)", flush=True)

print("=== done ===", flush=True)
import subprocess
print(subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout)

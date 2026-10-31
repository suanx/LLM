"""Upload the int4 ONNX bundle to ModelScope from the LOCAL machine.

The sandbox egress to modelscope.cn keeps getting reset, so we relay through the
local box: pull the bundle from the GitHub release, push it to ModelScope, delete it.
"""
import os
import sys
import time

from modelscope.hub.api import HubApi

TOK = os.environ["MS_TOKEN"]
REPO = "Kobexz/miro-2b"
SRC = os.environ.get("SRC", "C:/AI/agent/_onnx_dl")

FILES = [
    "model.onnx",
    "model.onnx.data",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "SHA256SUMS.txt",
]

api = HubApi()
api.login(TOK)

for f in FILES:
    src = os.path.join(SRC, f)
    if not os.path.isfile(src):
        print("skip missing", f)
        continue
    dst = f"onnx/{f}"
    size = os.path.getsize(src) / 1024 / 1024
    for attempt in range(1, 6):
        try:
            print(f"uploading {f} ({size:.1f} MiB) -> {dst} [try {attempt}]", flush=True)
            api.upload_file(repo_id=REPO, path_or_fileobj=src, path_in_repo=dst)
            print(f"  done {dst}", flush=True)
            break
        except Exception as e:
            print(f"  failed: {str(e)[:160]}", flush=True)
            if attempt == 5:
                print("GIVING UP on", f, flush=True)
                sys.exit(1)
            time.sleep(15 * attempt)

print("MS_ONNX_UPLOAD_DONE", flush=True)

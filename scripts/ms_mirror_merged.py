"""Mirror miro-2b-src (merged bf16 + LoRA adapter) from GitHub Releases to ModelScope.

Reason: the user's ModelScope cloud dev-machine (Alibaba Cloud) pulls from GitHub
releases extremely slowly / unreliably for the 4.8G split volumes. ModelScope hub is
also on Alibaba infra, so the dev machine can pull from the same-CDN fast path.

Design: each file is downloaded (resume + sha256 verify) then uploaded by a SEPARATE
subprocess (ms_upload_one.py) so a 1.9G upload OOM/kill cannot kill the whole job.
"""
import os
import sys
import time
import hashlib
import subprocess

TOK = os.environ["MS_TOKEN"]
REPO = "Kobexz/miro-2b"
BASE_URL = "https://github.com/suanx/LLM/releases/download/miro-2b-src"
STAGE = os.environ.get("STAGE", "C:/AI/agent/_mnn_src")
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(STAGE, exist_ok=True)

DATA_FILES = [
    "miro-2b-merged-bf16.tar.part-00",
    "miro-2b-merged-bf16.tar.part-01",
    "miro-2b-merged-bf16.tar.part-02",
    "miro-2b-lora-adapter.tar.gz",
]
SUMS_FILE = "SHA256SUMS-src.txt"


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, local: str, want: str | None) -> bool:
    """Resume download; verify sha256; if mismatch, full re-download once."""
    if os.path.exists(local):
        if want and sha256_file(local) == want:
            print(f"  cached+verified {os.path.basename(local)}", flush=True)
            return True
        os.remove(local)
    for _ in range(3):
        subprocess.run(["curl", "-sL", "--retry", "3", "-C", "-", "-o", local, url], check=True)
        got = sha256_file(local)
        if not want or got == want:
            print(f"  downloaded+verified {os.path.basename(local)} ({os.path.getsize(local)/1024/1024:.1f} MiB)", flush=True)
            return True
        print(f"  checksum mismatch, re-downloading {os.path.basename(local)}", flush=True)
        os.remove(local)
    return False


def upload_one(local: str, dst: str) -> int:
    """Run ms_upload_one.py as isolated subprocess. Returns exit code."""
    p = subprocess.run(
        [sys.executable, os.path.join(HERE, "ms_upload_one.py"), local, REPO, dst, TOK],
        stdout=sys.stdout, stderr=sys.stderr,
    )
    return p.returncode


print("=== fetch sums ===", flush=True)
subprocess.run(["curl", "-sL", "--retry", "3", "-o", os.path.join(STAGE, SUMS_FILE),
                f"{BASE_URL}/{SUMS_FILE}"], check=True)
sums = {}
for line in open(os.path.join(STAGE, SUMS_FILE)):
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) >= 2:
        sums[parts[1].strip()] = parts[0].strip()
print(f"parsed {len(sums)} sums", flush=True)

# upload sums first so dev machine can verify
print("=== upload sums ===", flush=True)
if upload_one(os.path.join(STAGE, SUMS_FILE), f"merged_bf16/{SUMS_FILE}") != 0:
    print("SUMS UPLOAD FAILED", flush=True)
    sys.exit(1)

# download + upload each data file
for f in DATA_FILES:
    local = os.path.join(STAGE, f)
    print(f"=== {f} ===", flush=True)
    if not download(f"{BASE_URL}/{f}", local, sums.get(f)):
        print(f"DOWNLOAD FAILED {f}", flush=True)
        sys.exit(1)
    dst = f"merged_bf16/{f}"
    # up to 3 attempts at the subprocess level (each subprocess retries 5x internally)
    ok = False
    for attempt in range(1, 4):
        rc = upload_one(local, dst)
        if rc == 0:
            ok = True
            break
        print(f"  upload subprocess rc={rc} (attempt {attempt}/3), retrying in 30s", flush=True)
        time.sleep(30)
    if not ok:
        print(f"UPLOAD FAILED {f} — leaving local for manual retry", flush=True)
        # do not exit; continue with others so partial progress is kept
        continue
    os.remove(local)
    print(f"  cleaned {f}", flush=True)

print("MS_MERGED_MIRROR_DONE", flush=True)

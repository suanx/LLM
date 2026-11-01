"""Idempotent completion uploader for miro-2b-src -> ModelScope merged_bf16/.

Only uploads files missing on the remote (checked via HTTP HEAD: 200 + matching
size). Safe to re-run; never re-uploads what's already there. Verifies each upload
by re-HEADing the remote, so a modelscope safe-delete OSError after a successful
upload does NOT block progress.
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
REMOTE = "https://www.modelscope.cn/models/Kobexz/miro-2b/resolve/master/merged_bf16"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def remote_ok(fname, want_size):
    """Return True if remote resolves to HTTP 200 and Content-Length == want_size."""
    out = subprocess.run(
        ["curl", "-s", "-I", "-L", "--max-time", "30", f"{REMOTE}/{fname}"],
        capture_output=True, text=True,
    )
    code = None
    cl = None
    for line in out.stdout.splitlines():
        if line.startswith("HTTP/"):
            parts = line.split()
            if len(parts) >= 2:
                code = parts[1]
        if line.lower().startswith("content-length:"):
            try:
                cl = int(line.split(":", 1)[1].strip())
            except ValueError:
                cl = None
    return code == "200" and cl == want_size


def remote_exists(fname):
    """Cheap HEAD check: True if remote resolves to HTTP 200 (any size)."""
    out = subprocess.run(
        ["curl", "-s", "-I", "-L", "--max-time", "30", f"{REMOTE}/{fname}"],
        capture_output=True, text=True,
    )
    for line in out.stdout.splitlines():
        if line.startswith("HTTP/") and len(line.split()) >= 2:
            if line.split()[1] == "200":
                return True
    return False


def download(url, local, want):
    if os.path.exists(local):
        if want and sha256_file(local) == want:
            print(f"  cached+verified {os.path.basename(local)}", flush=True)
            return True
        os.remove(local)
    for _ in range(3):
        rc = subprocess.run(
            ["curl", "-sL", "--retry", "3", "-C", "-", "-o", local, url]
        ).returncode
        if rc != 0:
            print(f"  curl rc={rc}, retry {os.path.basename(local)}", flush=True)
            time.sleep(5)
            continue
        got = sha256_file(local)
        if not want or got == want:
            print(f"  downloaded+verified {os.path.basename(local)} ({os.path.getsize(local)} B)", flush=True)
            return True
        print(f"  checksum mismatch, re-download {os.path.basename(local)}", flush=True)
        os.remove(local)
    return False


def upload_one(local, dst):
    p = subprocess.run(
        [sys.executable, os.path.join(HERE, "ms_upload_one.py"), local, REPO, dst, TOK],
        stdout=sys.stdout, stderr=sys.stderr,
    )
    return p.returncode


print("=== parse sums ===", flush=True)
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

for f in DATA_FILES:
    try:
        local = os.path.join(STAGE, f)
        print(f"=== {f} ===", flush=True)
        # already on remote?  Skip entirely (don't re-download) if HEAD returns 200.
        if remote_exists(f):
            print(f"  already on remote (HTTP 200), skip", flush=True)
            continue
        want_size = os.path.getsize(local) if os.path.exists(local) else 0
        # download if needed
        if not os.path.exists(local) or (sums.get(f) and sha256_file(local) != sums[f]):
            if not download(f"{BASE_URL}/{f}", local, sums.get(f)):
                print(f"DOWNLOAD FAILED {f}", flush=True)
                continue
        want_size = os.path.getsize(local)
        dst = f"merged_bf16/{f}"
        ok = False
        for attempt in range(1, 4):
            rc = upload_one(local, dst)
            if remote_ok(f, want_size):
                ok = True
                break
            print(f"  upload rc={rc}; remote not ready yet (attempt {attempt}/3), wait 30s", flush=True)
            time.sleep(30)
        if ok:
            print(f"  DONE {dst} (verified on remote)", flush=True)
            try:
                os.remove(local)
            except OSError:
                pass
        else:
            print(f"  STILL MISSING {dst} — leaving local for retry", flush=True)
    except Exception as e:
        print(f"  ERROR on {f}: {e!r} — continue next file", flush=True)
        continue

print("MS_MIRROR_REMAINING_DONE", flush=True)

"""Upload a single file to ModelScope. Run as a separate process so a large-file
OOM/kill cannot take down the whole mirror job. Exits non-zero on failure.

Usage: python -u ms_upload_one.py <local_path> <repo_id> <path_in_repo> <token>
"""
import sys
import os
import time

TOK = sys.argv[4]
REPO = sys.argv[2]
DST = sys.argv[3]
LOCAL = sys.argv[1]


def main():
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(TOK)
    size_mb = os.path.getsize(LOCAL) / 1024 / 1024
    print(f"[up] start {os.path.basename(LOCAL)} {size_mb:.1f} MiB -> {DST}", flush=True)
    last = None
    for attempt in range(1, 6):
        try:
            api.upload_file(repo_id=REPO, path_or_fileobj=LOCAL, path_in_repo=DST)
            print(f"[up] OK {DST}", flush=True)
            return 0
        except Exception as e:
            msg = str(e)
            # modelscope sometimes transient-fails on huge files; retry with backoff
            if msg == last:
                time.sleep(20 * attempt)
            else:
                print(f"[up] attempt {attempt} failed: {msg[:200]}", flush=True)
                last = msg
                time.sleep(15 * attempt)
    print(f"[up] GIVEUP {DST}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Upload the miro-2b artifacts to the ModelScope model repo Kobexz/miro-2b.

Small files first; the 1.43GB GGUF runs separately so it can be re-tried on its own.
"""
import os
import sys

from modelscope.hub.api import HubApi

TOK = os.environ["MS_TOKEN"]
REPO = "Kobexz/miro-2b"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or "."

api = HubApi()
api.login(TOK)


def up(src, dst):
    print(f"upload {src} -> {dst}", flush=True)
    api.upload_file(repo_id=REPO, path_or_fileobj=src, path_in_repo=dst)
    print(f"  done {dst}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "small"
    if what == "small":
        up(os.path.join(REPO_DIR, "modelscope", "README.md"), "README.md")
        up(os.path.join(REPO_DIR, "train", "chat_template.jinja"), "chat_template.jinja")
        up(os.path.join(REPO_DIR, "TRAINING.md"), "TRAINING.md")
        up(os.path.join(REPO_DIR, "train", "metrics.md"), "train/metrics.md")
        up(os.path.join(REPO_DIR, "train", "data", "DATASETS.md"), "train/data/DATASETS.md")
        up(os.path.join(REPO_DIR, "train", "data", "miro2b_toolcall_sft.jsonl"),
           "train/data/miro2b_toolcall_sft.jsonl")
        with open(os.path.join(REPO_DIR, "SHA256SUMS.txt"), "w") as f:
            f.write("936c2c16bafca37212d2b4551bc8749bed2e2eb04074cfeb1aeec960bd73a9d2  miro-2b-IQ4_XS.gguf\n")
        up(os.path.join(REPO_DIR, "SHA256SUMS.txt"), "SHA256SUMS.txt")
        print("SMALL_DONE", flush=True)
    elif what == "gguf":
        src = os.environ["GGUF_PATH"]
        up(src, "miro-2b-IQ4_XS.gguf")
        print("GGUF_DONE", flush=True)

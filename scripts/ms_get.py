#!/usr/bin/env python3
# 下载助手：优先 ModelScope SDK（云 IDE 内网快路径），失败自动回退 curl。
# 用法：python ms_get.py <repo_id> <file_in_repo> <local_dst>
import sys, os, subprocess


def main():
    if len(sys.argv) != 4:
        print("usage: ms_get.py <repo> <src> <dst>", file=sys.stderr)
        return 2
    repo, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    url = f"https://www.modelscope.cn/models/{repo}/resolve/master/{src}"
    # 1) 尝试 SDK 内网快路径
    try:
        from modelscope.hub.api import HubApi
        api = HubApi()
        try:
            api.download_file(repo, src, local_dir=os.path.dirname(dst),
                              local_name=os.path.basename(dst))
        except TypeError:
            # 旧版 SDK 参数顺序不同
            api.download_file(repo, src, os.path.dirname(dst))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print("SDK_OK", dst)
            return 0
        print("SDK_NO_FILE", file=sys.stderr)
    except Exception as e:
        print("SDK_FAIL", repr(e), file=sys.stderr)
    # 2) 回退 curl（-C - 断点续传）
    print("FALLBACK_CURL", url, file=sys.stderr)
    r = subprocess.run(["curl", "-L", "--retry", "5", "-C", "-",
                        "--progress-bar", "-o", dst, url])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())

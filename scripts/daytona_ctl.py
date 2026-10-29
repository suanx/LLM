"""Daytona sandbox control helper for the miro-2b build pipeline.

Usage:
  python scripts/daytona_ctl.py exec "shell command"          # run in miro-2b
  python scripts/daytona_ctl.py exec-bg "cmd" log.log          # run detached, tail log
  python scripts/daytona_ctl.py upload local remote
  python scripts/daytona_ctl.py download remote local
  python scripts/daytona_ctl.py resize cpu mem disk
  python scripts/daytona_ctl.py info
  python scripts/daytona_ctl.py destroy
"""
import os
import sys
import time

from daytona import Daytona, DaytonaConfig

API_KEY = os.environ.get("DAYTONA_API_KEY", "")
SANDBOX = os.environ.get("DAYTONA_SANDBOX", "miro-2b")

if not API_KEY:
    print("ERROR: set DAYTONA_API_KEY first (never commit it to git).")
    sys.exit(2)


def client():
    cfg = DaytonaConfig(api_key=API_KEY, api_url="https://app.daytona.io/api")
    return Daytona(cfg)


def sbx():
    return client().get(SANDBOX)


def cmd_exec():
    cmd = sys.argv[2]
    s = sbx()
    r = s.process.exec(cmd, timeout=int(os.environ.get("EXEC_TIMEOUT", "600")))
    print(getattr(r, "result", r))
    exit_code = getattr(r, "exit_code", 0)
    sys.exit(exit_code or 0)


def cmd_exec_bg():
    """Run a command fully detached so it survives the exec session being torn down."""
    cmd = sys.argv[2]
    log = sys.argv[3]
    s = sbx()
    wrapped = (
        f"setsid nohup bash -c {_shq(cmd)} "
        f"< /dev/null > {_shq(log)} 2>&1 & "
        f"sleep 1; echo PID=$!; echo STARTED_AT=$(date +%s)"
    )
    r = s.process.exec(wrapped, timeout=60)
    print(getattr(r, "result", r))


def cmd_kick():
    """Run a script file detached: kick <remote-script> <log>"""
    script, log = sys.argv[2], sys.argv[3]
    s = sbx()
    wrapped = (
        f"setsid nohup bash {_shq(script)} "
        f"< /dev/null > {_shq(log)} 2>&1 & "
        f"sleep 1; echo PID=$!; echo STARTED_AT=$(date +%s)"
    )
    r = s.process.exec(wrapped, timeout=60)
    print(getattr(r, "result", r))


def cmd_push():
    """Upload a local file by streaming base64 through the exec channel."""
    import base64

    src, dst = sys.argv[2], sys.argv[3]
    s = sbx()
    data = base64.b64encode(open(src, "rb").read()).decode()
    tmp = dst + ".b64"
    s.process.exec(f"rm -f {_shq(tmp)}", timeout=30)
    step = 4096
    for i in range(0, len(data), step):
        chunk = data[i : i + step]
        redir = ">" if i == 0 else ">>"
        s.process.exec(f"printf '%s' {_shq(chunk)} {redir} {_shq(tmp)}", timeout=60)
    r = s.process.exec(f"base64 --decode {_shq(tmp)} > {_shq(dst)} && rm -f {_shq(tmp)} && wc -c {_shq(dst)}", timeout=60)
    print(getattr(r, "result", r))


def cmd_tail():
    """tail <remote-log> [lines]"""
    log = sys.argv[2]
    n = sys.argv[3] if len(sys.argv) > 3 else "40"
    s = sbx()
    r = s.process.exec(f"tail -n {n} {_shq(log)} 2>&1; echo '--- ps ---'; pgrep -af 'python|cmake|make' | head -5", timeout=60)
    print(getattr(r, "result", r))


def _shq(x):
    return "'" + x.replace("'", "'\\''") + "'"


def cmd_upload():
    src, dst = sys.argv[2], sys.argv[3]
    s = sbx()
    data = open(src, "rb").read()
    s.fs.upload_file(dst, data)
    print(f"uploaded {src} -> {dst} ({len(data)} bytes)")


def cmd_download():
    src, dst = sys.argv[2], sys.argv[3]
    s = sbx()
    data = s.fs.download_file(src)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    with open(dst, "wb") as f:
        f.write(data)
    print(f"downloaded {src} -> {dst} ({len(data)} bytes)")


def cmd_resize():
    from daytona import Resources

    cpu, mem, disk = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    s = sbx()
    # Disk can only grow and requires a stopped sandbox; CPU/memory are hot-resizable.
    if disk and getattr(s, "disk", 0) and disk > s.disk:
        print("stopping sandbox for disk resize ...")
        s.stop()
        s.wait_for_sandbox_stop()
        s.refresh_data()
        s.resize(Resources(cpu=cpu, memory=mem, disk=disk))
        s.start()
        s.wait_for_sandbox_start()
    else:
        s.resize(Resources(cpu=cpu, memory=mem))
    s.wait_for_resize_complete()
    s.refresh_data()
    print(f"cpu={s.cpu} mem={s.memory} disk={s.disk} state={s.state}")


def cmd_start():
    s = sbx()
    s.start()
    s.wait_for_sandbox_start()
    s.refresh_data()
    print(f"state={s.state} cpu={s.cpu} mem={s.memory} disk={s.disk}")


def cmd_info():
    s = sbx()
    for k in ("id", "name", "state", "cpu", "memory", "disk", "snapshot", "target", "autoDeleteInterval"):
        print(f"  {k} = {getattr(s, k, None)}")


def cmd_destroy():
    s = sbx()
    s.delete()
    print("deleted", SANDBOX)


COMMANDS = {
    "exec": cmd_exec,
    "exec-bg": cmd_exec_bg,
    "kick": cmd_kick,
    "push": cmd_push,
    "tail": cmd_tail,
    "start": cmd_start,
    "upload": cmd_upload,
    "download": cmd_download,
    "resize": cmd_resize,
    "info": cmd_info,
    "destroy": cmd_destroy,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(2)
    COMMANDS[sys.argv[1]]()

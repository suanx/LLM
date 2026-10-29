#!/usr/bin/env bash
# Wait for the CPU LoRA run to finish, then merge + export without further interaction.
set -u
cd /c/AI/agent/LLM || exit 1
CTLP="python scripts/daytona_ctl.py"

rtail() { timeout 180 $CTLP tail "$1" "$2" 2>&1; }

wait_token() { # log token max_minutes label
  local log="$1" token="$2" maxmin="$3" label="$4"
  local tries=$((maxmin * 3))
  for i in $(seq 1 "$tries"); do
    if rtail "$log" 400 | grep -qa "$token"; then
      echo "[$label] DONE ($token) after ~$((i / 3)) min"
      return 0
    fi
    sleep 20
  done
  echo "[$label] TIMEOUT waiting for $token"
  return 1
}

echo "=== $(date +%H:%M:%S) stage 1: wait for training ==="
wait_token /home/daytona/train.log "TRAIN_DONE" 100 "train" || true
echo "--- train tail ---"
rtail /home/daytona/train.log 25

echo "=== adapter / checkpoints ==="
timeout 180 $CTLP exec "ls -la /home/daytona/out/; ls -la /home/daytona/out/adapter 2>/dev/null | head; df -h / | tail -1" 2>&1 | tail -20

echo "=== post-train sanity generation ==="
timeout 180 $CTLP exec "head -c 900 /home/daytona/out/gen_after.txt 2>/dev/null" 2>&1 | tail -20

echo "=== stage 1.5: free checkpoint space (keep newest ckpt) ==="
timeout 180 $CTLP exec "cd /home/daytona/out && ls -d ckpt-* 2>/dev/null | sort -t- -k2 -n | head -n -1 | xargs -r rm -rf; ls -la /home/daytona/out; df -h / | tail -1" 2>&1 | tail -12

echo "=== $(date +%H:%M:%S) stage 2: merge LoRA into the base ==="
timeout 180 $CTLP exec-bg "/home/daytona/venv/bin/python /home/daytona/sbx_merge.py" /home/daytona/merge.log 2>&1 | tail -2
wait_token /home/daytona/merge.log "MERGE_DONE" 40 "merge" || true
echo "--- merge tail ---"
rtail /home/daytona/merge.log 20
timeout 180 $CTLP exec "df -h / | tail -1; ls -la /home/daytona/merged | head" 2>&1 | tail -12

echo "=== $(date +%H:%M:%S) stage 3: export Q8_0 -> IQ4_XS ==="
timeout 180 $CTLP push scripts/export.sh /home/daytona/export.sh 2>&1 | tail -1
timeout 180 $CTLP kick /home/daytona/export.sh /home/daytona/export.log 2>&1 | tail -2
wait_token /home/daytona/export.log "EXPORT_DONE" 90 "export" || true
echo "--- export tail ---"
rtail /home/daytona/export.log 30

echo "=== $(date +%H:%M:%S) PIPELINE_WATCH_END ==="
timeout 180 $CTLP exec "df -h / | tail -1; ls -la /home/daytona/*.gguf /home/daytona/SHA256SUMS.txt 2>/dev/null" 2>&1 | tail -8

#!/usr/bin/env bash
# Full Phase-A orchestrator for the miro-2b ONNX round:
# venv -> base+data -> corpus -> LoRA SFT -> merge -> package to GitHub release.
# Runs locally; drives the Daytona sandbox through scripts/daytona_ctl.py.
set -u
cd /c/AI/agent/LLM || exit 1
CTLP="python scripts/daytona_ctl.py"

rtail() { timeout 180 $CTLP tail "$1" "$2" 2>&1; }

# Strip the "--- ps ---" section: pgrep echoes the RUNNING command line, which
# contains the very tokens we wait for (false positive).
logonly() { rtail "$1" "$2" | awk 'BEGIN{p=1} /--- ps ---/{p=0} p'; }

wait_token() { # log token max_minutes label
  local log="$1" token="$2" maxmin="$3" label="$4"
  local tries=$((maxmin * 3))
  for i in $(seq 1 "$tries"); do
    if logonly "$log" 400 | grep -qa "$token"; then
      echo "[$label] DONE ($token) after ~$((i / 3)) min"
      return 0
    fi
    if logonly "$log" 60 | grep -qaE "Traceback|No such file|error:"; then
      echo "[$label] saw an error in $log, dumping tail:"
      logonly "$log" 25
      return 1
    fi
    sleep 20
  done
  echo "[$label] TIMEOUT waiting for $token"
  return 1
}

echo "=== $(date +%H:%M:%S) stage 0: venv + deps ==="
if timeout 180 $CTLP exec "test -x /home/daytona/venv/bin/python && /home/daytona/venv/bin/python -c 'import torch,transformers,peft' 2>/dev/null && echo VENV_OK" 2>&1 | grep -qa VENV_OK; then
  echo "venv already present"
else
  $CTLP exec-bg "python3 -m venv /home/daytona/venv && /home/daytona/venv/bin/pip install -q --no-cache-dir --upgrade pip && /home/daytona/venv/bin/pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && /home/daytona/venv/bin/pip install -q --no-cache-dir transformers peft accelerate sentencepiece protobuf safetensors numpy huggingface_hub gguf && echo INSTALL_OK" /home/daytona/install.log | tail -2
  wait_token /home/daytona/install.log "INSTALL_OK" 20 "install" || exit 1
fi

echo "=== $(date +%H:%M:%S) stage 1: base + datasets ==="
$CTLP exec-bg "/home/daytona/venv/bin/python /home/daytona/fetch_base.py" /home/daytona/fetch.log | tail -2
wait_token /home/daytona/fetch.log "=== done ===" 30 "fetch" || exit 1

echo "=== $(date +%H:%M:%S) stage 2: corpus ==="
$CTLP exec-bg "/home/daytona/venv/bin/python /home/daytona/prepare_data.py" /home/daytona/prep.log | tail -2
wait_token /home/daytona/prep.log "PREP_DONE" 15 "prep" || exit 1
rtail /home/daytona/prep.log 12 | grep -aE "usable|blocks|corpus"

echo "=== $(date +%H:%M:%S) stage 3: LoRA SFT (200 steps) ==="
$CTLP exec-bg "MAX_STEPS=200 MAX_HOURS=3.0 LR=1.5e-4 USE_GC=1 NTHREADS=4 /home/daytona/venv/bin/python /home/daytona/train_lora.py" /home/daytona/train.log | tail -2
wait_token /home/daytona/train.log "TRAIN_DONE" 200 "train" || exit 1
rtail /home/daytona/train.log 20

echo "=== $(date +%H:%M:%S) stage 4: merge LoRA -> merged bf16 ==="
$CTLP exec-bg "/home/daytona/venv/bin/python /home/daytona/merge_lora.py" /home/daytona/merge.log | tail -2
wait_token /home/daytona/merge.log "MERGE_DONE" 30 "merge" || exit 1
rtail /home/daytona/merge.log 14
timeout 180 $CTLP exec "df -h / | tail -1" 2>&1 | tail -2

echo "=== $(date +%H:%M:%S) stage 5: package + upload to GitHub release miro-2b-src ==="
timeout 180 $CTLP push scripts/package_release.sh /home/daytona/package_release.sh | tail -1
$CTLP exec-bg "GITHUB_TOKEN=$GHT bash /home/daytona/package_release.sh" /home/daytona/package.log | tail -2
wait_token /home/daytona/package.log "PACKAGE_DONE" 70 "package" || { rtail /home/daytona/package.log 30; exit 1; }
rtail /home/daytona/package.log 25

echo "=== $(date +%H:%M:%S) PHASE_A_DONE ==="
timeout 180 $CTLP exec "df -h / | tail -1; ls -la /home/daytona/ | head -15" 2>&1 | tail -18

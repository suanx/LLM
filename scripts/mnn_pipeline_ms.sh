#!/usr/bin/env bash
# miro-2b: merged HF (bf16) -> MNN-LLM (int4) —— 魔搭(ModelScope)免费CPU云开发机专用版
#
# 与 daytona 版的区别：
#   1. 所有路径锁在 /mnt/workspace（魔搭唯一持久化目录，容器一关其它目录全没）
#   2. 自带 keepalive 心跳（魔搭空闲约 1h 会回收实例，前台跑一般没事，但下载间隙可能触发）
#   3. step3 检测 model_type，不是 minicpm 直接 exit 1（原 daytona 版只打印不中断）
#   4. 不依赖 daytona_ctl / scripts/keepalive.sh，单文件自包含
#
# 用法（在魔搭 CPU Notebook 的 Terminal 里）：
#   bash mnn_pipeline_ms.sh            # 全流程
#   STEP=3 bash mnn_pipeline_ms.sh     # 从第 3 步开始
set -o pipefail

# 脚本所在目录（用于定位 ms_get.py）
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ---- 基础路径（必须都在持久化目录）----
WS=/mnt/workspace
BASE=$WS/miro2b
STAGE_FILE=$BASE/.mnn_stage
STEP=${STEP:-0}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2
# 本脚本运行在阿里云(境内)云 IDE 上，直连 pypi.org / download.pytorch.org 不通。
# 用阿里云自己的 pip 镜像（同机房内网，这台机器上最快最稳；仅限这台机器，
# 不影响用户本机"不走镜像"的原则）。torch 默认是 CUDA 大轮，走内网镜像才能稳下完。
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.aliyun.com

MNN_VER=${MNN_VER:-3.6.2}

# ---- 防闲挂心跳：后台每 5 分钟打一行，保持终端活跃 ----
keepalive_start() {
  nohup bash -c 'while true; do echo "[keepalive $(date +%H:%M:%S)]"; sleep 300; done' \
    >"$BASE/keepalive.log" 2>&1 &
  KPID=$!
}
keepalive_stop() { [ -n "$KPID" ] && kill "$KPID" 2>/dev/null; }
trap keepalive_stop EXIT

stage_done() { grep -qx "$1" "$STAGE_FILE" 2>/dev/null; }
mark() { echo "$1" >> "$STAGE_FILE"; }
run_step() {
  local n=$1; shift
  if [ "$STEP" -le "$n" ] && ! stage_done "$1"; then
    "$@" || { echo "STEP_${n}_FAILED"; exit 1; }
    mark "$1"
  fi
}

# 每步前后打印资源，便于判断是否要扩/清理
res() { echo "--- disk ---"; df -h "$WS" | tail -1; echo "--- mem ---"; free -g | head -2; echo "--- cpu ---"; nproc; }

# 带重试的 pip 安装：失败可见、不静默跳过；最多 3 次
pipi() {
  local desc="$1"; shift
  for i in 1 2 3; do
    echo "  -> 安装 $desc (第 $i 次尝试)..."
    if "$BASE/venv_mnn/bin/pip" install "$@"; then return 0; fi
    echo "  [warn] $desc 安装失败，3 秒后重试..."
    sleep 3
  done
  echo "  [FAIL] $desc 安装失败，中止流程"; return 1
}

mkdir -p "$BASE"
keepalive_start
res

# ---------------------------------------------------------------- 1. 环境
step1() {
  echo "=== 1. 依赖 ==="
  # 1a. 仅当关键工具缺失时才跑 apt（首次已装好，跳过；避免重复 apt-get update 在云上卡网络）
  local need_apt=0
  for t in git cmake python3; do command -v $t >/dev/null || need_apt=1; done
  if [ "$need_apt" -eq 1 ]; then
    echo "[1a] 安装系统工具(apt)，可能需 1-2 分钟..."
    ( timeout 180 apt-get update -qq && timeout 400 apt-get install -y -qq git cmake build-essential python3-venv ) 2>/dev/null \
      || echo "[warn] apt 失败/超时，下面仍会检测关键工具是否已在"
  else
    echo "[1a] 系统工具已就绪，跳过 apt"
  fi
  for t in git cmake python3; do command -v $t >/dev/null || { echo "MISSING $t"; exit 1; }; done

  # 1b. venv（已存在则复用，不重建）
  if [ ! -d "$BASE/venv_mnn" ]; then
    echo "[1b] 创建 venv..."
    python3 -m venv "$BASE/venv_mnn" || { echo "venv 失败"; exit 1; }
  else
    echo "[1b] venv 已存在，复用"
  fi
  "$BASE/venv_mnn/bin/pip" install --upgrade pip

  # 1c. python 依赖（走上面设置的阿里云 pip 镜像；带重试）
  # 注：torch 不在 step1 装，留给 step5 按 MNN 官方 requirements.txt 的版本装，避免版本冲突/重复下载
  pipi "transformers/safetensors/numpy/onnx/onnxslim" transformers safetensors numpy onnx onnxslim || exit 1
  pipi "MNN==$MNN_VER" "MNN==$MNN_VER" || exit 1
  "$BASE/venv_mnn/bin/python" -c "import transformers, MNN; print('tf', transformers.__version__); print('MNN ok')" \
    || { echo "[FAIL] 依赖导入自检失败"; exit 1; }
}
run_step 1 step1

# ------------------------------------------------- 2. 拉合并后的 bf16 权重
# 来源：魔搭 hub Kobexz/miro-2b 的 merged_bf16/（3 分片，合计 ~4.8G）
# 已验证：源 tar 合法、sha256 与 SUMS 一致。本步先下载 3 分片 -> 校验 -> 拼成文件 -> tar xf 解包
# （早期版本用管道喂 tar，且把校验 echo 写进了 stdout 污染了 tar 流，导致“不是 tar 归档”报错）
step2() {
  echo "=== 2. 拉取 merged bf16 ==="
  if [ -d "$BASE/merged" ] && [ -f "$BASE/merged/config.json" ]; then
    echo "已存在，跳过"; return 0
  fi
  REPO=Kobexz/miro-2b
  PREFIX=merged_bf16
  PARTS="miro-2b-merged-bf16.tar.part-00 miro-2b-merged-bf16.tar.part-01 miro-2b-merged-bf16.tar.part-02"
  SUMS_F=SHA256SUMS-src.txt
  PYDL="$BASE/venv_mnn/bin/python"
  GET="$SCRIPT_DIR/ms_get.py"
  # 非强制安装 modelscope（云 IDE 内网快路径用；装不上就走 curl 回退）
  "$PYDL" -m pip install -q modelscope 2>/dev/null || echo "[warn] modelscope 未安装，下载走 curl 回退"
  rm -rf "$BASE/merged" "$BASE/_parts" "$BASE/_merged.tar"
  mkdir -p "$BASE/merged" "$BASE/_parts"
  for F in $SUMS_F $PARTS; do
    echo "[下载] $PREFIX/$F" >&2
    "$PYDL" "$GET" "$REPO" "$PREFIX/$F" "$BASE/_parts/$F" || { echo "DOWNLOAD FAIL $F"; return 1; }
    if [ "$F" != "$SUMS_F" ]; then
      WANT=$(grep -a "  $F\$" "$BASE/_parts/$SUMS_F" 2>/dev/null | cut -d' ' -f1)
      GOT=$(sha256sum "$BASE/_parts/$F" | cut -d' ' -f1)
      echo "[校验] $F want=$WANT got=$GOT" >&2
      if [ "$WANT" != "$GOT" ]; then echo "CHECKSUM FAIL $F（重新下载或换源）"; return 1; fi
    fi
  done
  # 拼接成单个 tar 再从文件解包（避免管道污染）
  cat "$BASE/_parts"/miro-2b-merged-bf16.tar.part-* > "$BASE/_merged.tar"
  tar xf "$BASE/_merged.tar" -C "$BASE/merged" || { echo "TAR EXTRACT FAIL"; return 1; }
  rm -rf "$BASE/_parts" "$BASE/_merged.tar"
  ls -la "$BASE/merged"
}
run_step 2 step2

# ------------------------------------------------------------ 3. 前置检查（关键）
# ⚠️ 不是 minicpm 直接中止！MNN 映射表里注册的是 MiniCPM；
# 若 config 里是 minicpm5 等新类型未注册，会回退 Llama 映射 -> 导出成功但输出乱码。
step3() {
  echo "=== 3. 检查模型 ==="
  ls -la "$BASE/merged"
  "$BASE/venv_mnn/bin/python" - <<'PY'
import json, os, sys
d = "/mnt/workspace/miro2b/merged"
cfg = json.load(open(os.path.join(d, "config.json")))
mt = cfg.get("model_type")
print("model_type      :", mt)
print("architectures   :", cfg.get("architectures"))
print("hidden_size     :", cfg.get("hidden_size"))
print("num_layers      :", cfg.get("num_hidden_layers"))
print("vocab_size      :", cfg.get("vocab_size"))
print("tie_word_embed  :", cfg.get("tie_word_embeddings"))
need = ["config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja"]
for f in need:
    print(("OK   " if os.path.exists(os.path.join(d, f)) else "MISS "), f)
if mt != "minicpm":
    print("\n[ABORT] model_type 不是 'minicpm'（实际：%s）。" % mt)
    print("MNN 可能回退 Llama 映射导致结果错误。请先告诉我这个 model_type，不要硬跑。")
    sys.exit(2)
print("\n[OK] model_type=minicpm，可继续。")
PY
  rc=$?
  [ "$rc" -ne 0 ] && exit "$rc"
  du -sh "$BASE/merged"
}
run_step 3 step3

# ---------------------------------------------------------------- 4. 拉 MNN
step4() {
  echo "=== 4. 拉 MNN 源码 ==="
  if [ ! -d "$BASE/MNN" ]; then
    git clone --depth 1 --branch $MNN_VER https://github.com/alibaba/MNN.git "$BASE/MNN" 2>/dev/null \
      || git clone --depth 1 https://github.com/alibaba/MNN.git "$BASE/MNN" \
      || git clone --depth 1 https://gitee.com/mirrors/MNN.git "$BASE/MNN"
  fi
  ls "$BASE/MNN/transformers/llm/export/"
}
run_step 4 step4

# ------------------------------------------------------------ 5. 导出 MNN
step5() {
  echo "=== 5. 导出 ==="
  cd "$BASE/MNN/transformers/llm/export"
  pipi "MNN export 依赖(含 torch)" -r requirements.txt || exit 1
  rm -rf "$BASE/mnn_out"
  res
  "$BASE/venv_mnn/bin/python" llmexport.py \
    --path "$BASE/merged" \
    --export mnn \
    --quant_bit 4 \
    --quant_block 64 \
    --embed_bit 16 \
    --hqq \
    --seperate_embed \
    --dst_path "$BASE/mnn_out" || return 1
  res
  ls -la "$BASE/mnn_out"
}
run_step 5 step5

# -------------------------------------------------------------- 6. 校验产物
step6() {
  echo "=== 6. 校验 ==="
  cd "$BASE/mnn_out"
  for f in llm.mnn llm.mnn.weight config.json llm_config.json tokenizer.txt; do
    [ -f "$f" ] || { echo "MISSING $f"; return 1; }
  done
  ls -la
  du -sh .
  "$BASE/venv_mnn/bin/python" - <<'PY'
import json, os
d = "/mnt/workspace/miro2b/mnn_out"
print("--- config.json (运行时配置) ---")
print(open(os.path.join(d, "config.json")).read()[:800])
print("--- llm_config.json (结构) ---")
c = json.load(open(os.path.join(d, "llm_config.json")))
for k in ("hidden_size", "layer_nums", "vocab_size", "key_value_shape",
          "attention_mask", "is_visual"):
    print(f"  {k}: {c.get(k)}")
PY
}
run_step 6 step6

# ------------------------------------------------------------ 7. 打包（在持久化目录内）
step7() {
  echo "=== 7. 打包 ==="
  cd "$BASE"
  rm -f miro-2b-mnn-int4.tar.gz
  tar czf miro-2b-mnn-int4.tar.gz -C "$BASE/mnn_out" .
  ls -lh miro-2b-mnn-int4.tar.gz
  sha256sum miro-2b-mnn-int4.tar.gz | tee miro-2b-mnn-int4.sha256
}
run_step 7 step7

echo "MNN_PIPELINE_DONE"
echo "产物在: $BASE/miro-2b-mnn-int4.tar.gz  (sha256 见 $BASE/miro-2b-mnn-int4.sha256)"
echo "下一步：把文件发给我，或按教程第 7 节上传魔搭 hub 的 Kobexz/miro-2b 的 mnn/ 前缀"

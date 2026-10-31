#!/usr/bin/env bash
# miro-2b: merged HF -> single ONNX decoding graph -> int4 weight-only ONNX.
# Disk staging (10G sandbox): merged (4.8G) is deleted BEFORE the export so we
# never hold two copies of the weights. The merged bf16 lives on GitHub release
# miro-2b-src, so deleting it here is safe and reproducible.
set -xo pipefail

H=/home/daytona
VENV=$H/venv2/bin/python
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo "=== staging ==="
df -h / | tail -1
free -g | head -2
ls -la $H/merged | head -4

echo "=== stash tokenizer/config (merged gets deleted later) ==="
rm -rf $H/onnx_assets; mkdir -p $H/onnx_assets
for f in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja; do
  [ -f $H/merged/$f ] && cp $H/merged/$f $H/onnx_assets/
done
ls -la $H/onnx_assets

echo "=== clean room (merged is deleted by the exporter once loaded) ==="
rm -rf $H/onnx_work $H/onnx_out $H/onnx_int4 2>/dev/null
df -h / | tail -1

echo "=== export fp16 graph ==="
DELETE_SRC=1 $VENV $H/onnx_export.py export --model $H/merged --work $H/onnx_work --out $H/onnx_out || exit 1
df -h / | tail -1

echo "=== fold Transpose(initializer) into the weights ==="
$VENV $H/onnx_export.py fold --work $H/onnx_work || exit 1
df -h / | tail -1

echo "=== quantize to int4 ==="
$VENV $H/onnx_export.py quantize --work $H/onnx_work --out $H/onnx_out --block 64 --symmetric 1 || exit 1
rm -rf $H/onnx_work
df -h / | tail -1

echo "=== package tokenizer/config + checksums ==="
$VENV $H/onnx_export.py package --out $H/onnx_out || exit 1
cd $H/onnx_out && sha256sum * | tee $H/onnx_out/SHA256SUMS.txt
ls -la $H/onnx_out
df -h / | tail -1
echo "ONNX_PIPELINE_DONE"

#!/usr/bin/env bash
# Build llama-quantize STATICALLY so the binary has no runtime .so deps.
set -xo pipefail

VENV=/home/daytona/venv
HOME_DIR=/home/daytona

cd "$HOME_DIR" || exit 1

"$VENV/bin/pip" install -q --no-cache-dir cmake || exit 1

rm -rf "$HOME_DIR/lcpp.build"
"$VENV/bin/cmake" -B "$HOME_DIR/lcpp.build" -S "$HOME_DIR/lcpp" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_BUILD_TESTS=OFF \
  -DGGML_BUILD_EXAMPLES=OFF || exit 1
echo "configure ok"

"$VENV/bin/cmake" --build "$HOME_DIR/lcpp.build" --config Release --target llama-quantize -j 4 || exit 1
echo "build ok"

mkdir -p "$HOME_DIR/bin"
find "$HOME_DIR/lcpp.build" -name llama-quantize -type f -perm -u+x | head -1 | xargs -I{} cp {} "$HOME_DIR/bin/llama-quantize"
chmod +x "$HOME_DIR/bin/llama-quantize"

echo "=== ldd ==="
ldd "$HOME_DIR/bin/llama-quantize" | head -10
echo "=== version ==="
"$HOME_DIR/bin/llama-quantize" --version 2>&1 | head -3
echo "=== IQ4_XS supported? ==="
"$HOME_DIR/bin/llama-quantize" 2>&1 | tr ' ' '\n' | grep -x 'IQ4_XS' | head -1

echo "BUILD_DONE"

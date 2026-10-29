#!/usr/bin/env bash
# HF -> Q8_0 -> IQ4_XS. Carefully staged for the 10GB sandbox disk budget.
set -xo pipefail

VENV=/home/daytona/venv
H=/home/daytona
export PYTHONPATH="$H/lcpp/gguf-py:$H/lcpp:$PYTHONPATH"

echo "=== deps check ==="
$VENV/bin/python -c "import gguf, numpy; print('gguf', gguf.__version__ if hasattr(gguf,'__version__') else 'ok')"

echo "=== convert HF -> Q8_0 ==="
df -h / | tail -1
mkdir -p "$H/q8"
$VENV/bin/python "$H/lcpp/convert_hf_to_gguf.py" "$H/merged" \
  --outfile "$H/q8/miro-2b-Q8_0.gguf" --outtype q8_0 || exit 1
ls -la "$H/q8"
df -h / | tail -1

echo "=== free the merged HF copy ==="
rm -rf "$H/merged"
df -h / | tail -1

echo "=== quantize Q8_0 -> IQ4_XS ==="
"$H/bin/llama-quantize" "$H/q8/miro-2b-Q8_0.gguf" "$H/miro-2b-IQ4_XS.gguf" IQ4_XS || exit 1
rm -rf "$H/q8"
df -h / | tail -1

echo "=== checksum ==="
sha256sum "$H/miro-2b-IQ4_XS.gguf" | tee "$H/SHA256SUMS.txt"
ls -la "$H/miro-2b-IQ4_XS.gguf"
echo "EXPORT_DONE"

#!/usr/bin/env bash
# GitHub backup release for the int4 ONNX bundle (sandbox -> GitHub is reliable).
set -xo pipefail

H=/home/daytona
REPO=suanx/LLM
TAG=miro-2b-onnx

echo "=== gh setup ==="
mkdir -p $H/bin
if [ ! -x $H/bin/gh ]; then
  VER=$(curl -s https://api.github.com/repos/cli/cli/releases/latest | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'].lstrip('v'))")
  curl -sL "https://github.com/cli/cli/releases/download/v${VER}/gh_${VER}_linux_amd64.tar.gz" -o $H/gh.tgz || exit 1
  tar -xzf $H/gh.tgz -C $H
  cp "$H/gh_${VER}_linux_amd64/bin/gh" $H/bin/gh
  chmod +x $H/bin/gh
  rm -rf "$H/gh_${VER}_linux_amd64" $H/gh.tgz
fi
GH=$H/bin/gh
$GH --version

cat > $H/RELEASE_NOTES-onnx.md <<'EOF'
miro-2b int4 ONNX — the on-device format for Miro (onnxruntime, same native stack as
the MOSS-TTS core already in the app).

- model.onnx + model.onnx.data  int4 weight-only (MatMulNBits, block=64, symmetric), 1.85 GiB
- tokenizer.json / tokenizer_config.json / chat_template.jinja
- config.json / generation_config.json / SHA256SUMS.txt

I/O contract: input_ids [1,S] int64, attention_mask [1,P+S] int64, position_ids [1,S] int64,
past_key_values.i.key/value [1,2,P,128] fp16 -> logits [1,S,130560], present.i.key/value [1,2,P+S,128].

Validated with onnxruntime 1.31 CPU (4 threads): session 5s, prefill 237 tokens 8s,
48-token generation 12s total; emits
<function name="get_weather"><param name="city">Beijing</param><param name="date">2023-10-01</param></function>

Same trained weights as the miro-2b IQ4_XS GGUF; source of truth for both is release miro-2b-src.
EOF

echo "=== GitHub release $TAG ==="
$GH release delete "$TAG" --repo "$REPO" --yes 2>/dev/null || true
$GH release create "$TAG" \
  "$H/onnx_out/model.onnx" \
  "$H/onnx_out/model.onnx.data" \
  "$H/onnx_out/tokenizer.json" \
  "$H/onnx_out/tokenizer_config.json" \
  "$H/onnx_out/config.json" \
  "$H/onnx_out/generation_config.json" \
  "$H/onnx_out/chat_template.jinja" \
  "$H/onnx_out/SHA256SUMS.txt" \
  --repo "$REPO" \
  --title "miro-2b int4 ONNX (onnxruntime, on-device)" \
  --notes-file "$H/RELEASE_NOTES-onnx.md" || exit 1

$GH release view "$TAG" --repo "$REPO" --json assets -q '.assets[] | .name + "  " + (.size|tostring)' 2>/dev/null || \
  $GH release view "$TAG" --repo "$REPO"
echo "ONNX_UPLOAD_DONE"

#!/usr/bin/env bash
# Package miro-2b artifacts on the sandbox and push them to a dedicated GitHub
# release (tag: miro-2b-src) so future format conversions never need retraining.
# Disk is tight (10G): the venv/data are deleted BEFORE splitting the 4.8G model.
# Requires GITHUB_TOKEN in the environment.
set -xo pipefail

H=/home/daytona
REPO=suanx/LLM
TAG=miro-2b-src

echo "=== before packaging ==="
df -h / | tail -1

echo "=== adapter tarball ==="
cd "$H"
tar czf miro-2b-lora-adapter.tar.gz -C out adapter gen_after.txt loss.json 2>/dev/null || \
  tar czf miro-2b-lora-adapter.tar.gz -C out adapter
ls -la miro-2b-lora-adapter.tar.gz

echo "=== free disk: drop venv / data (no longer needed) ==="
rm -rf "$H/venv" "$H/data" "$H/lcpp" "$H/.cache" "$H/bin" 2>/dev/null
df -h / | tail -1

echo "=== tar merged (uncompressed, exact bytes) ==="
tar cf "$H/merged.tar" -C "$H/merged" .
ls -la "$H/merged.tar"
sha256sum "$H/merged.tar" | tee "$H/SHA256SUMS-src.txt"
sha256sum "$H/miro-2b-lora-adapter.tar.gz" >> "$H/SHA256SUMS-src.txt"

echo "=== split into 1.9G parts ==="
split -b 1900m -d "$H/merged.tar" "$H/miro-2b-merged-bf16.tar.part-"
rm -f "$H/merged.tar"
df -h / | tail -1
ls -la "$H" | grep part

cat > "$H/REJOIN.md" <<'EOF'
# Rejoin miro-2b merged bf16 (HF format, for ONNX / any future conversion)

cat miro-2b-merged-bf16.tar.part-* > miro-2b-merged-bf16.tar
sha256sum -c SHA256SUMS-src.txt        # only the merged.tar line applies here
mkdir merged && tar xf miro-2b-merged-bf16.tar -C merged

Files inside: config.json, generation_config.json, tokenizer.json,
tokenizer_config.json, chat_template.jinja, model.safetensors.index.json,
model-0000X-of-0000X.safetensors
EOF

echo "=== gh setup ==="
VER=$(curl -s https://api.github.com/repos/cli/cli/releases/latest | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'].lstrip('v'))")
curl -sL "https://github.com/cli/cli/releases/download/v${VER}/gh_${VER}_linux_amd64.tar.gz" -o /tmp/gh.tgz || exit 1
tar -xzf /tmp/gh.tgz -C /tmp
GH="/tmp/gh_${VER}_linux_amd64/bin/gh"
$GH --version

echo "=== release $TAG ==="
$GH release delete "$TAG" --repo "$REPO" --yes 2>/dev/null || true
cat > "$H/RELEASE_NOTES-src.md" <<'EOF'
miro-2b source artifacts — kept so future format conversions (ONNX / others) never need retraining.

- miro-2b-lora-adapter.tar.gz     trained LoRA adapter (r16 a32) + loss.json + sample
- miro-2b-merged-bf16.tar.part-*  merged HF model (bf16 safetensors), split into 1.9G parts
- SHA256SUMS-src.txt              checksums
- REJOIN.md                       how to rejoin

Base model: insraq/MiniCPM5-2B-heretic-abliterated. Training data & pipeline: repo suanx/LLM.
EOF
$GH release create "$TAG" \
  "$H/miro-2b-lora-adapter.tar.gz" \
  "$H"/miro-2b-merged-bf16.tar.part-* \
  "$H/SHA256SUMS-src.txt" \
  "$H/REJOIN.md" \
  --repo "$REPO" \
  --title "miro-2b source artifacts (adapter + merged bf16)" \
  --notes-file "$H/RELEASE_NOTES-src.md" || exit 1

rm -f "$H"/miro-2b-merged-bf16.tar.part-* "$H/merged.tar"
df -h / | tail -1
echo "PACKAGE_DONE"

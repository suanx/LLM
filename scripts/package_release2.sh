#!/usr/bin/env bash
# Package miro-2b artifacts, DISK-SAFE edition v2.
# The 4.8G merged model can never coexist with its own copy, so every GitHub
# part is streamed straight out of a deterministic `tar cf -` pipe:
#   pass 1: measure total stream size
#   pass 2: per part, stream to sha256 only
#   pass 3: per part, stream to a part file -> gh upload -> delete file
# Peak disk = merged dir + ONE part (~6.7G / 10G). Requires GITHUB_TOKEN.
set -xo pipefail

H=/home/daytona
REPO=suanx/LLM
TAG=miro-2b-src
PARTBYTES=1992294400   # 1900 MiB (GitHub asset limit is 2 GiB)

mkdir -p /home/daytona/bin
ensure_gh() {
  if [ ! -x /home/daytona/bin/gh ]; then
    VER=$(curl -s https://api.github.com/repos/cli/cli/releases/latest | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'].lstrip('v'))")
    curl -sL "https://github.com/cli/cli/releases/download/v${VER}/gh_${VER}_linux_amd64.tar.gz" -o /home/daytona/gh.tgz || exit 1
    tar -xzf /home/daytona/gh.tgz -C /home/daytona
    cp "/home/daytona/gh_${VER}_linux_amd64/bin/gh" /home/daytona/bin/gh
    chmod +x /home/daytona/bin/gh
    rm -rf "/home/daytona/gh_${VER}_linux_amd64" /home/daytona/gh.tgz
  fi
}
GH=/home/daytona/bin/gh
ensure_gh
$GH --version

echo "=== drop the broken attempt ==="
ensure_gh
$GH release delete "$TAG" --repo "$REPO" --yes 2>/dev/null || true
rm -f "$H"/miro-2b-merged-bf16.tar.part-* "$H/merged.tar" "$H/SHA256SUMS-src.txt"

echo "=== adapter tarball ==="
cd "$H"
rm -f miro-2b-lora-adapter.tar.gz
tar czf miro-2b-lora-adapter.tar.gz -C out adapter gen_after.txt loss.json 2>/dev/null || \
  tar czf miro-2b-lora-adapter.tar.gz -C out adapter
ls -la miro-2b-lora-adapter.tar.gz

TARFMT="--sort=name --mtime=0 --owner=0 --group=0 --numeric-owner --format=gnu"

echo "=== pass 1: measure stream ==="
TOTAL=$(tar $TARFMT -cf - -C "$H/merged" . | wc -c)
NPARTS=$(( (TOTAL + PARTBYTES - 1) / PARTBYTES ))
echo "total=$TOTAL parts=$NPARTS"

echo "=== pass 2: per-part sha256 ==="
: > "$H/SHA256SUMS-src.txt"
for i in $(seq 0 $((NPARTS - 1))); do
  SKIP=$(( i * PARTBYTES + 1 ))
  P=$(printf "miro-2b-merged-bf16.tar.part-%02d" "$i")
  SHA=$(tar $TARFMT -cf - -C "$H/merged" . | tail -c "+$SKIP" | head -c "$PARTBYTES" | sha256sum | cut -d' ' -f1)
  echo "$SHA  $P" >> "$H/SHA256SUMS-src.txt"
  echo "  $P sha=$SHA"
done
ASHA=$(sha256sum "$H/miro-2b-lora-adapter.tar.gz" | cut -d' ' -f1)
echo "$ASHA  miro-2b-lora-adapter.tar.gz" >> "$H/SHA256SUMS-src.txt"

cat > "$H/REJOIN.md" <<EOF
# Rejoin miro-2b merged bf16 (HF format, source of truth for future conversions)

cat miro-2b-merged-bf16.tar.part-* > miro-2b-merged-bf16.tar
# expected size: exactly $TOTAL bytes; verify against SHA256SUMS-src.txt
mkdir merged && tar xf miro-2b-merged-bf16.tar -C merged
EOF

cat > "$H/RELEASE_NOTES-src.md" <<EOF
miro-2b source artifacts — kept so future format conversions (ONNX etc) never need retraining.

- miro-2b-lora-adapter.tar.gz      trained LoRA adapter (r16 a32) + loss.json + sample generation
- miro-2b-merged-bf16.tar.part-*   merged HF model (bf16 safetensors), $NPARTS parts, $TOTAL bytes total
- SHA256SUMS-src.txt               per-part checksums + adapter checksum
- REJOIN.md                        cat parts -> tar -> extract

Base: insraq/MiniCPM5-2B-heretic-abliterated. Training data & pipeline: suanx/LLM.
Trained 2026-10-30: LoRA SFT 200 steps, loss 1.2883 -> 0.1250.
EOF

echo "=== create release ==="
ensure_gh
$GH release delete "$TAG" --repo "$REPO" --yes 2>/dev/null || true
$GH release create "$TAG" \
  "$H/miro-2b-lora-adapter.tar.gz" \
  "$H/SHA256SUMS-src.txt" \
  "$H/REJOIN.md" \
  --repo "$REPO" \
  --title "miro-2b source artifacts (adapter + merged bf16)" \
  --notes-file "$H/RELEASE_NOTES-src.md" || exit 1

echo "=== pass 3: stream part -> upload -> delete ==="
for i in $(seq 0 $((NPARTS - 1))); do
  SKIP=$(( i * PARTBYTES + 1 ))
  P=$(printf "miro-2b-merged-bf16.tar.part-%02d" "$i")
  echo "--- part $i ---"
  tar $TARFMT -cf - -C "$H/merged" . | tail -c "+$SKIP" | head -c "$PARTBYTES" > "$H/$P" || exit 1
  ensure_gh
  $GH release upload "$TAG" "$H/$P" --repo "$REPO" --clobber || exit 1
  rm -f "$H/$P"
  df -h / | tail -1
done

echo "=== release view ==="
$GH release view "$TAG" --repo "$REPO" --json assets -q '.assets[] | .name + "  " + (.size|tostring)' 2>/dev/null || \
  $GH release view "$TAG" --repo "$REPO"

df -h / | tail -1
echo "PACKAGE_DONE"

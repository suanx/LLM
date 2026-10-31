#!/usr/bin/env bash
# Rebuild the merged bf16 model inside the 10G sandbox: download each GitHub
# release part, verify it, stream it into `tar x` and delete it immediately.
# Peak disk = one part (1.9G) + extracted model (4.7G), never two copies.
set -xo pipefail

H=/home/daytona
BASE=https://github.com/suanx/LLM/releases/download/miro-2b-src
SUMS=$H/SHA256SUMS-src.txt

curl -sL --retry 3 -o "$SUMS" "$BASE/SHA256SUMS-src.txt" || exit 1
cat "$SUMS"

rm -rf "$H/merged"
mkdir -p "$H/merged"

(
  for p in 00 01 02; do
    F="miro-2b-merged-bf16.tar.part-$p"
    curl -sL --retry 3 -o "$H/$F" "$BASE/$F" || exit 1
    WANT=$(grep -a "  $F\$" "$SUMS" | cut -d' ' -f1)
    GOT=$(sha256sum "$H/$F" | cut -d' ' -f1)
    # NOTE: everything except `cat` must go to stderr - stdout IS the tar stream
    echo "$F want=$WANT got=$GOT" >&2
    [ "$WANT" = "$GOT" ] || exit 1
    cat "$H/$F"
    rm -f "$H/$F"
    df -h / | tail -1 >&2
  done
) | tar x -C "$H/merged" || exit 1

ls -la "$H/merged"
df -h / | tail -1
echo "MERGED_READY"

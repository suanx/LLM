#!/usr/bin/env bash
# Recovery: part-00 was fully written before pipefail+SIGPIPE killed the run.
# Verify existing parts against SHA256SUMS-src.txt, stream+upload the rest.
set -xo pipefail

H=/home/daytona
REPO=suanx/LLM
TAG=miro-2b-src
PARTBYTES=1992294400

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

TARFMT="--sort=name --mtime=0 --owner=0 --group=0 --numeric-owner --format=gnu"
TOTAL=$(grep -aoE 'expected size: exactly [0-9]+ bytes' "$H/REJOIN.md" | grep -aoE '[0-9]+')
NPARTS=$(( (TOTAL + PARTBYTES - 1) / PARTBYTES ))
echo "total=$TOTAL parts=$NPARTS"

upload_part() { # $1 = part filename
  ensure_gh
  $GH release upload "$TAG" "$H/$1" --repo "$REPO" --clobber || exit 1
  rm -f "$H/$1"
  df -h / | tail -1
}

for i in $(seq 0 $((NPARTS - 1))); do
  SKIP=$(( i * PARTBYTES + 1 ))
  P=$(printf "miro-2b-merged-bf16.tar.part-%02d" "$i")
  if [ "$i" -eq $((NPARTS - 1)) ]; then EXP=$(( TOTAL - i * PARTBYTES )); else EXP=$PARTBYTES; fi
  WANT=$(grep -a "  $P\$" "$H/SHA256SUMS-src.txt" | cut -d' ' -f1)
  echo "--- part $i expect_size=$EXP ---"

  if [ -f "$H/$P" ] && [ "$(stat -c%s "$H/$P")" = "$EXP" ]; then
    GOT=$(sha256sum "$H/$P" | cut -d' ' -f1)
    if [ "$GOT" = "$WANT" ]; then
      echo "  existing file verified, reuse"
    else
      echo "  sha mismatch, restream"
      rm -f "$H/$P"
    fi
  fi

  if [ ! -f "$H/$P" ]; then
    tar $TARFMT -cf - -C "$H/merged" . | tail -c "+$SKIP" | head -c "$PARTBYTES" > "$H/$P" || true
    GOT=$(sha256sum "$H/$P" | cut -d' ' -f1)
    [ "$(stat -c%s "$H/$P")" = "$EXP" ] || { echo "SIZE MISMATCH"; exit 1; }
    [ "$GOT" = "$WANT" ] || { echo "SHA MISMATCH"; exit 1; }
  fi
  upload_part "$P"
done

echo "=== release view ==="
$GH release view "$TAG" --repo "$REPO" --json assets -q '.assets[] | .name + "  " + (.size|tostring)' 2>/dev/null || \
  $GH release view "$TAG" --repo "$REPO"
df -h / | tail -1
echo "PACKAGE_DONE"

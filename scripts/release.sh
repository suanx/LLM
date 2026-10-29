#!/usr/bin/env bash
# Publish miro-2b-IQ4_XS.gguf straight from the sandbox to GitHub Releases
# (the large asset never touches the local machine).
# Requires: GITHUB_TOKEN=<pat with repo scope> in the environment.
set -xo pipefail

H=/home/daytona
REPO=suanx/LLM
TAG=miro-2b
ASSET="$H/miro-2b-IQ4_XS.gguf"

ARCH=linux_amd64
VER=$(curl -s https://api.github.com/repos/cli/cli/releases/latest | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'].lstrip('v'))")
echo "gh version: $VER"
curl -sL "https://github.com/cli/cli/releases/download/v${VER}/gh_${VER}_${ARCH}.tar.gz" -o /tmp/gh.tgz || exit 1
tar -xzf /tmp/gh.tgz -C /tmp || exit 1
GH="/tmp/gh_${VER}_${ARCH}/bin/gh"
$GH --version

# wipe the old releases, keep exactly one
for old in lora-poc iq4xs-nano; do
  $GH release delete "$old" --repo "$REPO" --yes || echo "  (no $old)"
done

# (re)create the single nano release
$GH release delete "$TAG" --repo "$REPO" --yes || true
cat > "$H/release_notes.md" <<'EOF'
miro-2b — end-side tool-calling model for Miro.

- Base: insraq/MiniCPM5-2B-heretic-abliterated (2.52B, abliterated)
- LoRA SFT on ToolACE + Hermes function-calling data, rendered with the model's own
  chat template so tool calls come out as MiniCPM5 XML
  (`<function name="..."><param name="...">...</param></function>`)
- Quantisation: IQ4_XS (~1.4GB), built via Q8_0 master with llama.cpp llama-quantize
- Serve with llama.cpp llama-server (needs the minicpm5 tool parser) and consume the
  standard OpenAI /v1/chat/completions tool_calls JSON.

See TRAINING.md for the reproducible pipeline and SHA256SUMS.txt for the checksum.
EOF

$GH release create "$TAG" "$ASSET" "$H/SHA256SUMS.txt" \
  --repo "$REPO" \
  --title "miro-2b IQ4_XS (nano)" \
  --notes-file "$H/release_notes.md" || exit 1

echo "RELEASE_DONE"
$GH release view "$TAG" --repo "$REPO"

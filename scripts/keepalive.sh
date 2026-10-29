#!/usr/bin/env bash
# Keep the Daytona sandbox alive while long CPU jobs run: Daytona auto-stops a
# sandbox after ~15-30 min without API activity, which would kill training.
set -u
TOK="${DAYTONA_API_KEY:?need DAYTONA_API_KEY}"
ID="${DAYTONA_SANDBOX_ID:?need DAYTONA_SANDBOX_ID}"
PERIOD="${KEEPALIVE_PERIOD:-300}"

echo "keepalive -> $ID every ${PERIOD}s"
while true; do
  code=$(curl -s -X POST -H "Authorization: Bearer $TOK" \
    "https://app.daytona.io/api/sandbox/$ID/last-activity" \
    -o /dev/null -w '%{http_code}')
  echo "$(date +%H:%M:%S) last-activity -> $code"
  sleep "$PERIOD"
done

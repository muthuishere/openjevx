#!/usr/bin/env bash
# One-time (and repeatable) Cloudflare setup for the fine-tuning pipeline. Safe to re-run.
#   - R2 bucket openjevx-train (private): shards, runs, checkpoints, gate reports
#   - Pages project openjevx-destroy: the endpoint a GPU box calls to destroy itself
#     (finetuning/destroy-worker/functions/destroy.js), with its secrets set from sec
#   - OPENJEVX_KILL_TOKEN in sec (generated once, never printed)
# Then checks the endpoint: no token -> 403, with token -> reaches Vast.
# Needs: $CLOUDFLARE_ALLPURPOSE_TOKEN in your login shell, sec (with VAST_API_KEY), wrangler, python3.
# Why Pages and not a Worker: the all-purpose token may publish Pages but not create Workers.
set -euo pipefail
cd "$(dirname "$0")"
ACCOUNT="${CLOUDFLARE_ACCOUNT_ID:-4cf2da436d01cd836a29ae0ae1eee8fc}"
BUCKET="${OPENJEVX_R2_BUCKET:-openjevx-train}"
PROJECT=openjevx-destroy
API="https://api.cloudflare.com/client/v4/accounts/$ACCOUNT"
CLOUDFLARE_API_TOKEN="$(zsh -ic 'printf %s "$CLOUDFLARE_ALLPURPOSE_TOKEN"' 2>/dev/null)"
[ -n "$CLOUDFLARE_API_TOKEN" ] || { echo "CLOUDFLARE_ALLPURPOSE_TOKEN is not set in your login shell"; exit 1; }
export CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID="$ACCOUNT"
cf() { curl -fsS -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json" "$@"; }
ok() { python3 -c "import sys,json; d=json.load(sys.stdin); print(d['success'], d.get('errors') or '')"; }

echo "== R2 bucket $BUCKET"
if cf "$API/r2/buckets/$BUCKET" >/dev/null 2>&1; then echo "exists"; else cf -X POST "$API/r2/buckets" -d "{\"name\":\"$BUCKET\"}" | ok; fi

echo "== kill token"
sec ls --json | grep -q '"OPENJEVX_KILL_TOKEN"' && echo "exists in sec" || sec gen --save OPENJEVX_KILL_TOKEN --len 40

echo "== Pages project $PROJECT"
if cf "$API/pages/projects/$PROJECT" >/dev/null 2>&1; then echo "exists"; else
  cf -X POST "$API/pages/projects" -d "{\"name\":\"$PROJECT\",\"production_branch\":\"main\"}" | ok; fi

echo "== secrets (values go from sec straight into wrangler)"
cd destroy-worker
sec run OPENJEVX_KILL_TOKEN,VAST_API_KEY -- sh -c '
  printf %s "$OPENJEVX_KILL_TOKEN" | wrangler pages secret put KILL_TOKEN --project-name '"$PROJECT"' >/dev/null 2>&1 && echo KILL_TOKEN set
  printf %s "$VAST_API_KEY" | wrangler pages secret put VAST_API_KEY --project-name '"$PROJECT"' >/dev/null 2>&1 && echo VAST_API_KEY set'

echo "== deploy (after the secrets, so this deployment has them)"
wrangler pages deploy public --project-name "$PROJECT" --branch main --commit-dirty=true 2>&1 | grep -E "complete|✘" || true
sleep 8

echo "== check"
URL="https://$PROJECT.pages.dev/destroy"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL" -d '{"instance_id":1}')
[ "$code" = 403 ] && echo "no token -> 403 ok" || { echo "no token -> $code (expected 403)"; exit 1; }
sec exec -- curl -fsS -X POST "$URL" -H "X-Kill-Token: {{OPENJEVX_KILL_TOKEN}}" -d '{"instance_id":1}' | grep -q '"destroyed":false' \
  && echo "with token -> reaches Vast ok" || { echo "with token -> unexpected answer"; exit 1; }
echo "ready: bucket $BUCKET, endpoint $URL"

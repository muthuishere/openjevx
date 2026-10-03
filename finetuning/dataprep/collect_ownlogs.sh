#!/usr/bin/env bash
# Read-only collection of OUR OWN logs for the EVAL-ONLY ownlogs gate (gen_ownlogs_gate.py).
# Every line goes through `sec sh -c cat` (redacts every sec vault value) and then the regex scrub
# (gen_ownlogs_gate.py --scrub KIND) before anything is written; raw text never lands on disk.
# Writes <data>/raw/ownlogs/<kind>.txt. Never commit the output; never use it for training.
#   openjevx-server    */srv*/server.log under ~/openjevx/data/work and /private/tmp/claude-$UID
#   openjevx-pipeline  ~/openjevx/data/work/*.log
#   kamal-proxy        `docker logs --tail $KAMAL_TAIL kamal-proxy` on our deemwar servers, reduced ON THE
#                      SERVER to level/msg/service/method/status/duration; client services (words in $EXCLUDE_FILE), their hosts
#                      and non-access reqsume lines are dropped there too
#   ci-openjevx        `gh run view --log` for muthuishere/openjevx
#   ci-pgbx            `gh run view --log` for the pgbx test workflows (deemwar-products/pgbx)
# Needs: sec, gh (account muthuishere), ssh aliases deemwar-app1/app2/dev, python3.
set -euo pipefail
# Client / user-content markers live outside the repo (client names never reach git). Fail closed without them.
EXCLUDE_FILE="${OWNLOGS_EXCLUDE:-$HOME/.config/openjevx/ownlogs-exclude.txt}"
[ -s "$EXCLUDE_FILE" ] || { echo "collect_ownlogs: $EXCLUDE_FILE is missing or empty" >&2; exit 1; }
EXCL=$(grep -v '^#' "$EXCLUDE_FILE" | grep -v '^$' | paste -sd'|' -)
cd "$(dirname "$0")"
DATA_RAW="$(cd .. && python3 -c 'import paths; print(paths.RAW)')"
OUT="$DATA_RAW/ownlogs"
mkdir -p "$OUT"
KAMAL_HOSTS="${KAMAL_HOSTS:-deemwar-app1 deemwar-app2 deemwar-dev}"
KAMAL_TAIL="${KAMAL_TAIL:-10000}"
RUNS="${RUNS:-25}"
scrub() { sec sh -c cat | python3 gen_ownlogs_gate.py --scrub "$1"; }

echo "== openjevx-server"
{ find "$HOME/openjevx/data/work" "/private/tmp/claude-$(id -u)" -path '*srv*' -name server.log -size +0 2>/dev/null \
    | while read -r f; do cat "$f"; done; } | scrub openjevx-server > "$OUT/openjevx-server.txt"

echo "== openjevx-pipeline"
{ for f in "$HOME"/openjevx/data/work/*.log; do [ -s "$f" ] && cat "$f"; done; } \
  | scrub openjevx-pipeline > "$OUT/openjevx-pipeline.txt"

echo "== kamal-proxy"
REMOTE='EX=$(docker exec kamal-proxy kamal-proxy list 2>/dev/null | sed "s/\x1b\[[0-9;]*m//g" \
  | awk "NR>1 && tolower(\$0) ~ /'"$EXCL"'/ {print \$2}" | tr "," "\n" | jq -R . | jq -s -c .)
docker logs --tail '"$KAMAL_TAIL"' kamal-proxy 2>&1 | jq -R -c --argjson ex "${EX:-[]}" '"'"'
  fromjson? | select((tostring | test("'"$EXCL"'"; "i")) | not)
  | select(([.host, (.hosts // [])[]?] | map(select(. != null)) | any(. as $h | $ex | index($h))) | not)
  | if .msg == "Request" then {level, msg, service, method, path: "<path>", status, duration}
    else select((tostring | test("reqsume"; "i")) | not) | del(.time) end'"'"
{ for h in $KAMAL_HOSTS; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "$REMOTE" || echo "kamal-proxy: $h unreachable" >&2
  done; } | scrub kamal-proxy > "$OUT/kamal-proxy.txt"

echo "== CI logs"
GH_TOKEN="$(gh auth token --user muthuishere)"
export GH_TOKEN
ghlogs() {  # repo, kind, extra gh run list filters...
  local repo=$1 kind=$2; shift 2
  { gh run list -R "$repo" -L "$RUNS" --json databaseId -q '.[].databaseId' "$@"
    gh run list -R "$repo" -L "$RUNS" --status failure --json databaseId -q '.[].databaseId' "$@"; } \
    | sort -u | while read -r id; do
        gh run view "$id" -R "$repo" --log 2>/dev/null | sed -E $'s/^[^\t]*\t[^\t]*\t(\xef\xbb\xbf)?[0-9T:.-]+Z ?//' || true
      done
}
{ ghlogs muthuishere/openjevx ci-openjevx; } | scrub ci-openjevx > "$OUT/ci-openjevx.txt"
{ for wf in "CLI tests" "Install test" "Windows check"; do ghlogs deemwar-products/pgbx ci-pgbx -w "$wf"; done; } \
  | scrub ci-pgbx > "$OUT/ci-pgbx.txt"

wc -l "$OUT"/*.txt

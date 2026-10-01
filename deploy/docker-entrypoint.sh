#!/bin/sh
# Start the server with a real dashboard password: a mounted /app/openjevx.json, or OPENJEVX_PASSWORD.
# There is no default password; without one the container refuses to start.
set -eu
cfg=/app/openjevx.json
if [ ! -f "$cfg" ]; then
  pw="${OPENJEVX_PASSWORD:-}"
  [ -n "$pw" ] || { echo "openjevx: set OPENJEVX_PASSWORD or mount your own $cfg; refusing to start without a dashboard password" >&2; exit 1; }
  [ "${#pw}" -ge 12 ] || { echo "openjevx: OPENJEVX_PASSWORD must be at least 12 characters" >&2; exit 1; }
  case "$pw" in *[!A-Za-z0-9._~-]*) echo "openjevx: OPENJEVX_PASSWORD may use only letters, digits and . _ ~ -" >&2; exit 1 ;; esac
  printf '{\n  "listen": "0.0.0.0:21118",\n  "device": "cpu",\n  "model": "/app/model",\n  "password": "%s"\n}\n' "$pw" > "$cfg"
fi
exec /app/openjevx "$@"

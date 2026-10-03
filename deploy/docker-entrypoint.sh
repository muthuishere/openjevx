#!/bin/sh
# Start the server with real credentials: OPENJEVX_PASSWORD (dashboard) and OPENJEVX_API_KEY (POST /v1/systemone,
# "Authorization: Bearer <key>"), or your own mounted openjevx.json (/app or /data) that sets them.
# There are no defaults; without them the container refuses to start. OPENJEVX_ALLOW_NO_API_KEY=1 opens the
# decision API on purpose (e.g. behind a proxy that checks the key). The server reads both from the environment.
set -eu
if [ ! -f /app/openjevx.json ] && [ ! -f /data/openjevx.json ]; then
  pw="${OPENJEVX_PASSWORD:-}"
  [ -n "$pw" ] || { echo "openjevx: set OPENJEVX_PASSWORD or mount your own openjevx.json; refusing to start without a dashboard password" >&2; exit 1; }
  [ "${#pw}" -ge 12 ] || { echo "openjevx: OPENJEVX_PASSWORD must be at least 12 characters" >&2; exit 1; }
  if [ "${OPENJEVX_ALLOW_NO_API_KEY:-}" != 1 ]; then
    key="${OPENJEVX_API_KEY:-}"
    [ -n "$key" ] || { echo "openjevx: set OPENJEVX_API_KEY (16+ characters, e.g. openssl rand -hex 24); refusing to start with an open decision API" >&2; exit 1; }
    [ "${#key}" -ge 16 ] || { echo "openjevx: OPENJEVX_API_KEY must be at least 16 characters" >&2; exit 1; }
  fi
  printf '{\n  "listen": "0.0.0.0:21118",\n  "device": "cpu",\n  "model": "/app/model"\n}\n' > /data/openjevx.json
fi
exec /app/openjevx "$@"

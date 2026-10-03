#!/bin/bash
# Install the OpenJevX server and the current 8-bit model on a Linux box as a systemd service.
# Used by the DigitalOcean 1-Click image, the AWS AMI, and cloud-init on any VPS.
#   sudo bash install.sh     (server release from deploy/VERSION or OPENJEVX_VERSION,
#                             model from deploy/MODEL_VERSION or OPENJEVX_MODEL_VERSION)
# OPENJEVX_BASE is where the two release files are downloaded from (the release folder URL).
set -euo pipefail

# The server release: $OPENJEVX_VERSION, else deploy/VERSION next to this script. The model has its own version
# ($OPENJEVX_MODEL_VERSION, else deploy/MODEL_VERSION); its archive is attached to the server release too.
here="$(cd "$(dirname "$0")" && pwd)"
VERSION="${OPENJEVX_VERSION:-$(cat "$here/VERSION" 2>/dev/null || true)}"
[ -n "$VERSION" ] || { echo "install.sh: set OPENJEVX_VERSION (no VERSION file next to this script)" >&2; exit 1; }
MODEL_VERSION="${OPENJEVX_MODEL_VERSION:-$(cat "$here/MODEL_VERSION" 2>/dev/null || true)}"
[ -n "$MODEL_VERSION" ] || { echo "install.sh: set OPENJEVX_MODEL_VERSION (no MODEL_VERSION file next to this script)" >&2; exit 1; }
BASE="${OPENJEVX_BASE:-https://github.com/muthuishere/openjevx/releases/download/v${VERSION}}"
DIR=/opt/openjevx
PORT=21118

[ "$(id -u)" = 0 ] || { echo "install.sh: run as root" >&2; exit 1; }
case "$(uname -m)" in
  x86_64) arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "install.sh: OpenJevX is released for linux amd64 and arm64, not $(uname -m)" >&2; exit 1 ;;
esac
command -v curl >/dev/null || { apt-get update -q && apt-get install -y -q curl ca-certificates; }

# Download, then check every file against the release's SHA256SUMS-server before unpacking anything.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
server=openjevx-linux-$arch.tar
model="openjevx-model-${MODEL_VERSION}.tar.gz"
for f in SHA256SUMS-server "$server" "$model"; do curl -fsSL "$BASE/$f" -o "$tmp/$f"; done
for f in "$server" "$model"; do
  grep -q "  $f\$" "$tmp/SHA256SUMS-server" || { echo "install.sh: $f is not listed in SHA256SUMS-server" >&2; exit 1; }
done
(cd "$tmp" && grep -E "  ($server|$model)\$" SHA256SUMS-server | sha256sum -c --quiet -) \
  || { echo "install.sh: checksum mismatch; nothing installed" >&2; exit 1; }

# Unpack into a fresh folder and swap it in, so a re-run leaves no stale files. The config and
# its first-boot password survive a re-run.
rm -rf "$DIR.new" && mkdir -p "$DIR.new"
tar -x -C "$DIR.new" -f "$tmp/$server"
tar -xz -C "$DIR.new" -f "$tmp/$model"
[ -x "$DIR.new/openjevx" ] && [ -f "$DIR.new/model/config.json" ] || { echo "install.sh: download incomplete in $DIR.new" >&2; exit 1; }
rm -f "$DIR.new/openjevx.json"
for f in openjevx.json openjevx.json.done openjevx.api-key openjevx.password; do if [ -f "$DIR/$f" ]; then cp -p "$DIR/$f" "$DIR.new/$f"; fi; done
rm -rf "$DIR.old"; if [ -d "$DIR" ]; then mv "$DIR" "$DIR.old"; fi
mv "$DIR.new" "$DIR" && rm -rf "$DIR.old"

id openjevx >/dev/null 2>&1 || useradd --system --home "$DIR" --shell /usr/sbin/nologin openjevx
chown -R openjevx:openjevx "$DIR"

# The config (its dashboard password and API key) is written on first boot, so every server gets its own.
cat > /usr/local/sbin/openjevx-firstboot <<FIRSTBOOT
#!/bin/bash
set -euo pipefail
[ -f $DIR/openjevx.json.done ] && exit 0
pw=\$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 20)
key=\$(head -c 36 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32)
printf '{\n  "listen": "0.0.0.0:$PORT",\n  "device": "cpu",\n  "model": "$DIR/model",\n  "password": "%s",\n  "api_key": "%s"\n}\n' "\$pw" "\$key" > $DIR/openjevx.json
chown openjevx:openjevx $DIR/openjevx.json && chmod 600 $DIR/openjevx.json
printf 'OpenJevX dashboard password: %s\n' "\$pw" > /root/openjevx-password && chmod 600 /root/openjevx-password
printf '%s\n' "\$key" > /root/openjevx-api-key && chmod 600 /root/openjevx-api-key
touch $DIR/openjevx.json.done
FIRSTBOOT
chmod 755 /usr/local/sbin/openjevx-firstboot

cat > /etc/systemd/system/openjevx.service <<UNIT
[Unit]
Description=OpenJevX decision server
After=network-online.target
Wants=network-online.target

[Service]
User=openjevx
WorkingDirectory=$DIR
ExecStartPre=+/usr/local/sbin/openjevx-firstboot
ExecStart=$DIR/openjevx
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/update-motd.d/99-openjevx <<'MOTD'
#!/bin/sh
cat <<TXT

OpenJevX is running as the service 'openjevx' (systemctl status openjevx).
  Health:    curl http://127.0.0.1:21118/health
  Decide:    POST http://127.0.0.1:21118/v1/systemone
             with  -H "Authorization: Bearer \$(cat /root/openjevx-api-key)"
  Dashboard: http://<this-ip>:21118/  (password: cat /root/openjevx-password)
Servers installed before 0.5.7 got their API key on the first start of 0.5.7: /opt/openjevx/openjevx.api-key.
Port 21118 is closed by the firewall; open it only to your own network, e.g.
  ufw allow from 10.0.0.0/8 to any port 21118
Or use an SSH tunnel:  ssh -L 21118:127.0.0.1:21118 root@<this-ip>
TXT
MOTD
chmod 755 /etc/update-motd.d/99-openjevx

# Firewall: SSH only. The customer opens 21118 to their own network (the API key is a second lock, not the only one).
# Allow every port sshd really listens on, so a non-standard SSH port does not lock anyone out.
if command -v ufw >/dev/null; then
  ssh_ports="$( (sshd -T 2>/dev/null || true) | awk '$1=="port"{print $2}')"
  if [ -z "$ssh_ports" ]; then
    echo "install.sh: WARNING could not read the sshd port; allowing 22. Check 'ufw status' before logging out." >&2
    ssh_ports=22
  fi
  for p in $ssh_ports; do ufw allow "$p/tcp" >/dev/null; done
  ufw --force enable >/dev/null
  echo "install.sh: firewall on; SSH allowed on port(s) $(echo "$ssh_ports" | tr "\n" " "); 21118 closed until you open it." >&2
fi

systemctl daemon-reload
systemctl enable openjevx >/dev/null
if [ "${OPENJEVX_NO_START:-}" != 1 ]; then
  systemctl restart openjevx
  for _ in $(seq 60); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
  curl -fsS "http://127.0.0.1:$PORT/health" || { echo "install.sh: server not healthy; see journalctl -u openjevx" >&2; exit 1; }
  echo
fi
echo "OpenJevX $VERSION (model $MODEL_VERSION) installed in $DIR"

#!/bin/bash
# Install the OpenJevX server and the current 8-bit model on a Linux box as a systemd service.
# Used by the DigitalOcean 1-Click image, the AWS AMI, and cloud-init on any VPS.
#   sudo OPENJEVX_VERSION=0.5.0 bash install.sh
# OPENJEVX_BASE is where the two release files are downloaded from (the release folder URL).
set -euo pipefail

VERSION="${OPENJEVX_VERSION:-0.5.0}"
BASE="${OPENJEVX_BASE:-https://github.com/muthuishere/openjevx/releases/download/v${VERSION}}"
DIR=/opt/openjevx
PORT=21118

[ "$(id -u)" = 0 ] || { echo "install.sh: run as root" >&2; exit 1; }
[ "$(uname -m)" = x86_64 ] || { echo "install.sh: only linux amd64 is released today, not $(uname -m)" >&2; exit 1; }
command -v curl >/dev/null || { apt-get update -q && apt-get install -y -q curl ca-certificates; }

mkdir -p "$DIR"
curl -fsSL "$BASE/openjevx-linux-amd64.tar" | tar -x -C "$DIR"
curl -fsSL "$BASE/openjevx-model-${VERSION}.tar.gz" | tar -xz -C "$DIR"
[ -x "$DIR/openjevx" ] && [ -f "$DIR/model/config.json" ] || { echo "install.sh: download incomplete in $DIR" >&2; exit 1; }

id openjevx >/dev/null 2>&1 || useradd --system --home "$DIR" --shell /usr/sbin/nologin openjevx
chown -R openjevx:openjevx "$DIR"

# The config (and its dashboard password) is written on first boot, so every server gets its own password.
cat > /usr/local/sbin/openjevx-firstboot <<FIRSTBOOT
#!/bin/bash
set -euo pipefail
[ -f $DIR/openjevx.json.done ] && exit 0
pw=\$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 20)
printf '{\n  "listen": "0.0.0.0:$PORT",\n  "device": "cpu",\n  "model": "$DIR/model",\n  "password": "%s"\n}\n' "\$pw" > $DIR/openjevx.json
chown openjevx:openjevx $DIR/openjevx.json && chmod 600 $DIR/openjevx.json
printf 'OpenJevX dashboard password: %s\n' "\$pw" > /root/openjevx-password && chmod 600 /root/openjevx-password
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
  Dashboard: http://<this-ip>:21118/  (password: cat /root/openjevx-password)
The decision API has no password. Port 21118 is closed by the firewall; open it only to your own
network, e.g.  ufw allow from 10.0.0.0/8 to any port 21118
Or use an SSH tunnel:  ssh -L 21118:127.0.0.1:21118 root@<this-ip>
TXT
MOTD
chmod 755 /etc/update-motd.d/99-openjevx

# Firewall: SSH only. The decision API has no password, so the customer opens 21118 to their own network.
if command -v ufw >/dev/null; then
  ufw allow OpenSSH >/dev/null && ufw --force enable >/dev/null
fi

systemctl daemon-reload
systemctl enable openjevx >/dev/null
if [ "${OPENJEVX_NO_START:-}" != 1 ]; then
  systemctl restart openjevx
  for _ in $(seq 60); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
  curl -fsS "http://127.0.0.1:$PORT/health" || { echo "install.sh: server not healthy; see journalctl -u openjevx" >&2; exit 1; }
  echo
fi
echo "OpenJevX $VERSION installed in $DIR"

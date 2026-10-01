#!/bin/bash
# Leave no build secrets, keys, logs or history in the snapshot (DigitalOcean Marketplace rules).
set -euo pipefail
rm -f /opt/openjevx/openjevx.json /opt/openjevx/openjevx.json.done /root/openjevx-password
apt-get -y autoremove && apt-get -y autoclean
rm -rf /tmp/* /var/tmp/* /root/.ssh/authorized_keys
history -c; : > /root/.bash_history
find /var/log -type f -exec truncate -s 0 {} +
rm -f /var/lib/cloud/instances/*/sem/* || true
cloud-init clean --logs

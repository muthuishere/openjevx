#!/bin/bash
# DigitalOcean's own image validator; the build fails if the snapshot breaks a Marketplace rule.
# Pinned to one commit of digitalocean/marketplace-partners and its sha256; bump both together.
set -euo pipefail
ref=b70878804ca27c01d5f5e882d26485defbaba210
sum=91ff2b1880439c97ccdc49554c5ed8901b89ef250c8ac5fffe52811c68abac49
curl -fsSL "https://raw.githubusercontent.com/digitalocean/marketplace-partners/$ref/scripts/99-img-check.sh" -o /tmp/img-check.sh
echo "$sum  /tmp/img-check.sh" | sha256sum -c --quiet - || { echo "99-img-check.sh: checksum mismatch" >&2; exit 1; }
bash /tmp/img-check.sh
rm -f /tmp/img-check.sh

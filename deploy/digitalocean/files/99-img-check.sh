#!/bin/bash
# DigitalOcean's own image validator; the build fails if the snapshot breaks a Marketplace rule.
set -euo pipefail
curl -fsSL https://raw.githubusercontent.com/digitalocean/marketplace-partners/master/scripts/99-img-check.sh -o /tmp/img-check.sh
bash /tmp/img-check.sh
rm -f /tmp/img-check.sh

# Deploying OpenJevX in the customer's own cloud

The customer always runs the model in their own account; we never host inference. Every path below
installs the same thing: the openjevx server plus the current 8-bit model folder, started on boot,
with `GET /health` as the health check. `deploy/install.sh` does the work for the images and cloud-init.

| Path | Files | Who does the last step |
|---|---|---|
| DigitalOcean 1-Click (Marketplace) | `deploy/digitalocean/` | owner submits in the Vendor Portal |
| Deploy to DigitalOcean button (App Platform) | `.do/deploy.template.yaml` | the customer clicks |
| Any VPS: AWS, Azure, GCP, Hetzner, ... | `deploy/cloud-init.yaml`, `deploy/docker-compose.yml` | the customer pastes |
| AWS Marketplace AMI | `deploy/aws/` | owner registers and submits |

**Security default.** Every path below listens beyond loopback, so `POST /v1/systemone` needs
`Authorization: Bearer <API key>` and the dashboard needs its password (README, "Credentials"); there are no
default credentials anywhere.
- **Images and cloud-init** (`install.sh`): each server writes its own random dashboard password and API key on
  first boot, into `openjevx.json` and `/root/openjevx-password` / `/root/openjevx-api-key` (mode 0600). Servers
  installed before 0.5.7 generate their key on the first start of 0.5.7, in `/opt/openjevx/openjevx.api-key`. The
  firewall (ufw) still opens only SSH; the customer opens port 21118 to their own network or uses an SSH tunnel.
- **Docker image** (repo `Dockerfile`, Debian 12 slim, runs as uid 10001): refuses to start without
  `OPENJEVX_PASSWORD` (12+ characters) and `OPENJEVX_API_KEY` (16+), or your own `openjevx.json` mounted at
  `/data/openjevx.json` or read-only at `/app/openjevx.json`. Credentials it has to generate go to `/data`
  (`OPENJEVX_DATA`), and the logs show only their file and a fingerprint, never the value. App Platform asks for both
  as secrets.

`install.sh` checks both downloads against the release's `SHA256SUMS-server` and swaps in a clean folder, so a re-run
leaves no stale files and keeps the config and credentials. It allows every port sshd listens on before turning the
firewall on. Minimum size: linux amd64 or arm64, 2 vCPU, 4 GB RAM (the 8-bit model is 598 MB).

**Version.** `deploy/VERSION` is the server release and `deploy/MODEL_VERSION` the model; the server release
carries that model's archive too. `install.sh` and both Packer templates read both; the Dockerfile builds the
server from source and reads only `MODEL_VERSION`. `deploy/cloud-init.yaml` and `deploy/docker-compose.yml` are paste-able, so they are generated:
after changing `VERSION`, `MODEL_VERSION`, `install.sh` or the Dockerfile, commit, run `deploy/render.sh`, and commit again.
`deploy/render.sh --check` fails on a stale file. They pin the commit that last changed what they fetch, and
install.sh's sha256, never the moving `main`.

## Public text

Listings, buttons and the site say **deemwar** (the OSS partner). Never "muthuishere", and no
github.com links on deemwar surfaces. The build files may download from the GitHub release;
set `OPENJEVX_BASE` / `-var openjevx_base=` to a public R2 folder instead once the release files are copied there.

## DigitalOcean Marketplace 1-Click

1. Build the snapshot (needs Packer and a DigitalOcean token; costs a few cents of droplet time):
   ```
   cd deploy/digitalocean && packer init . && sec run DIGITALOCEAN_TOKEN -- packer build .
   ```
   The build runs DigitalOcean's `99-img-check.sh` (pinned to a commit and sha256); it must pass.
2. Test: create a droplet from the snapshot, SSH in, read the login message, `curl http://127.0.0.1:21118/health`.
3. Owner submits in the [Vendor Portal](https://cloud.digitalocean.com/vendorportal) with:
   - Name: OpenJevX · Vendor: deemwar · Category: Machine Learning / Developer Tools
   - Summary: "A small decision model you run on your own CPU. Ask yes/no, pick-one and rating questions over an HTTP API; answers come with a confidence."
   - Software included: OpenJevX `deploy/VERSION` with model `deploy/MODEL_VERSION` (today 0.5.7 with model 0.5.2;
     Apache-2.0), ONNX Runtime 1.29.0 (MIT), Ubuntu 24.04
   - Recommended size: s-2vcpu-4gb
   - Getting started: the text of the login message (`/etc/update-motd.d/99-openjevx`)
   - Support URL and email: deemwar's
   - The snapshot from step 1 (shared with the Marketplace team)

## Deploy to DigitalOcean button

```
[![Deploy to DO](https://www.deploytodo.com/do-btn-blue.svg)](https://cloud.digitalocean.com/apps/new?repo=<public repo URL>/tree/main)
```
App Platform reads `.do/deploy.template.yaml`, builds the Dockerfile and checks `/health`. Its URL is public, so the
template requires `OPENJEVX_API_KEY` and `OPENJEVX_PASSWORD` as secrets and the decision API answers 401 without the
key. Even so, this path suits trials; production should use the 1-Click droplet or a VPS inside a private network.

## Any VPS

- **cloud-init:** paste `deploy/cloud-init.yaml` as the instance's user data (EC2 "User data",
  Azure "Custom data", GCP `user-data` metadata, Hetzner/DO "User data").
- **Docker:** copy `deploy/docker-compose.yml` to the box and run
  `OPENJEVX_PASSWORD=<12+ characters> OPENJEVX_API_KEY=<16+ characters> docker compose up -d --build`.

## AWS Marketplace (AMI first, container later)

Build: `cd deploy/aws && packer init . && sec run AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY -- packer build .`
Then in the AWS Marketplace Management Portal: add the AMI, run the self-service scan, create the product.
The container product (EKS/ECS) is built in the jev-cloud repo (`image/Dockerfile`), not from this repo's Dockerfile:
it pins an openjevx release tag, runs on distroless as uid 10001, and puts its gate in front of the server (the gate
checks `Authorization: Bearer <JEV_API_KEY>` and runs openjevx on `127.0.0.1:21119`, where openjevx's own key is off).

### Serving the customer's own model from their S3

The marketplace flow trains in the customer's account and serves from their bucket; nothing leaves it:

1. The jev client uploads the customer's CSV to `s3://<bucket>/training/<job>/input/`; a SageMaker training job
   writes `training/<job>/output/model.tar.gz`.
2. Promote = copy the three files of a verified version into `s3://<bucket>/models/current/` (rollback = promote the
   previous version again).
3. The server runs with `OPENJEVX_MODEL=s3://<bucket>/models/current/`, `OPENJEVX_MODEL_RELOAD=5m` and
   `OPENJEVX_MODEL_CACHE=/tmp/jev-cache` (the marketplace container is uid 10001 on distroless). It picks up the promote within
   one interval, keeps the previous folder, and reports both on `/health`. Until the first promote it serves the base
   model shipped beside the binary (`/app/model`).

Its role is read-only: `s3:GetObject` on `arn:aws:s3:::<bucket>/models/current/*` and `s3:ListBucket` on the bucket,
conditioned on `s3:prefix` `models/current/` and `models/current/*`. Settings: README, "The model from S3".

The marketplace image (jev-cloud) is distroless (no `/bin/sh`) and runs as uid 10001. The server works there on amd64
and arm64 from 0.5.3: CI starts it in `gcr.io/distroless/cc-debian12` on both and answers one decision
(`docker-smoke`). CI also builds this repo's own image and checks it runs as uid 10001 (`image`).

Outside AWS (MinIO, Ceph, R2), add `AWS_ENDPOINT_URL_S3=<endpoint>` and `AWS_S3_USE_PATH_STYLE=true`
(or `"model_s3_path_style": true` in `openjevx.json`), so requests go to `<endpoint>/<bucket>/<key>` and need no
bucket DNS.

### What the owner must register (only the owner can do these)

- [ ] AWS account for selling (a dedicated one is best), with MFA on root
- [ ] Register as a seller in the AWS Marketplace Management Portal; accept the seller terms
- [ ] Public seller profile: company name deemwar, description, logo, support email and URL
- [ ] Tax interview (W-8BEN-E for a non-US company) in the portal
- [ ] Bank account for disbursements (a US bank account or a supported non-US one) and KYC verification
- [ ] Pricing decision: free (BYOL) or paid hourly/annual (paid needs the tax + bank steps done first)
- [ ] EULA: standard contract for AWS Marketplace, or our own
- [ ] Share the AMI with the Marketplace account and submit the product request

### DigitalOcean, for comparison

- [ ] DigitalOcean account with the Vendor Portal enabled (apply as a Marketplace partner)
- [ ] Listing text and logo above; 1-Click apps are free listings, so no tax or bank step

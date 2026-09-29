# ADR 0008 — Self-hosted OpenJevX appliance on cloud marketplaces, with a training UI

Status: proposed
Date: 2026-09-29

## Context

ADR 0007 runs fine-tuning for customers. Many teams (regulated, security-sensitive, or just
cautious) will not send their documents, tickets and logs to anyone. They will run software inside
their own cloud account. Cloud marketplaces are how they buy that: one click, billed on their
existing cloud invoice, running in their VPC.

Everything the appliance needs already exists as code: the openjevx server (Go, ONNX Runtime,
dashboard), the fine-tuning pipeline (`finetuning/`), the gate, and the 8-bit ONNX format.
What is missing is packaging, a UI that walks a non-ML engineer through training, and a way to get a
GPU inside the customer's account.

## Decision (proposed)

Ship **OpenJevX Appliance**: one image the customer launches in their own cloud, containing

1. **The decision server** (today's openjevx server) serving the current model.
2. **A training UI** (web, same dashboard) that does ADR 0007's steps in their account:
   - *Sources*: upload files, or connect a bucket / git repo / ticket export / log folder.
   - *Rules*: extracted rule candidates shown as editable checks; the user confirms each one.
   - *Decisions*: define the questions the model must answer (yes/no, pick one, rating) and see
     generated examples with near-miss twins; spot-check and fix.
   - *Gate*: the user's own test questions (kept out of training) plus our basics gate.
   - *Train*: one button; shows cost estimate, progress, logs.
   - *Results*: per decision right & confident / confidently wrong / abstain, compare with the
     current model, **promote** only if the gate passes and nothing regresses; one-click rollback.
3. **The pipeline** (`finetuning/`) as the engine behind the UI, with a provider interface for the
   GPU step (already `gpu/<provider>.py`).

Data never leaves the customer's account. Base model weights ship in the image (or pulled once).

## Options

**A. Marketplaces, in order**
- A1 (pick first): **DigitalOcean Marketplace 1-Click Droplet** — simplest listing (a Packer-built
  snapshot + a setup script), fast approval, good for small teams; GPU Droplets exist for training.
- A2 (pick second): **AWS Marketplace** — AMI product (EC2) first, then a container product for
  EKS/ECS. Where enterprises buy. Metered or annual pricing through AWS billing; more listing
  work (security scan, usage metering, seller registration).
- A3: Azure Marketplace, GCP Marketplace, Hetzner/Contabo images, and a plain Docker Compose /
  Helm chart for anyone else. The Docker image already exists; this is the free path.

**B. Where training runs**
- B1 (pick): the appliance itself is CPU-only for serving; for training it starts a **GPU
  instance in the customer's own account** (AWS g5/g6, DO GPU Droplet), runs the same one-job box
  script, writes results to the customer's own bucket, and the box destroys itself (same design
  as our Vast flow, with the customer's cloud as the provider).
- B2: appliance launched directly on a GPU instance — simplest, but pays GPU prices to serve.
- B3: send training to our managed service (ADR 0007) — only for customers who allow it.

**C. Pricing**
- C1 (pick to test): free community image (serve + train, basic UI) + paid listing with the full
  training UI, rule extraction, multi-model and support; hourly or annual via the marketplace.
- C2: BYOL (license key) for enterprises buying outside the marketplace.

**D. Packaging**
- One container image (server + UI + pipeline + ONNX Runtime) built in CI; Packer builds the
  DO snapshot and the AWS AMI from it; Helm/Compose for self-managed.
- Config in `~/.config/openjevx/config.json` inside the appliance; data in `~/openjevx/data`
  (same layout as today, so the pipeline code is unchanged).
- Updates: new image versions; models and data stay on the customer's volume/bucket.

**E. Training UI tech**
- E1 (pick): extend the existing server dashboard (Go-served) with a small web app; the UI calls
  the same `ft.py` steps as jobs. No second backend.
- E2: separate web app. More flexible, twice the surface.

## Consequences

- One engine (`finetuning/`), three front doors: open-source DIY (ADR 0006), managed (ADR 0007),
  self-hosted appliance (this ADR).
- The GPU provider interface becomes a real contract: `vast`, `aws`, `digitalocean`, `local-gpu`.
- Needs: image build pipeline, marketplace seller accounts and listings, metering for AWS,
  security hardening (auth on the UI, TLS, secrets in the customer's secret manager), docs.
- The training UI is the biggest piece of new work; the rule-confirm and gate screens are
  shared with ADR 0007.

## Open questions

- DigitalOcean GPU Droplet availability/regions and quotas for customer accounts.
- AWS: AMI first or container first; which instance family is the default for training.
- Whether the base model licence allows redistribution inside a paid marketplace image
  (check ADR 0001's base model licence before listing).

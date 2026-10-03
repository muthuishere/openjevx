# ADR 0009 — One GPU job: everything through R2, the box destroys itself

Status: accepted
Date: 2026-09-29
Supersedes: the launcher, hardware-image and watcher parts of ADR 0002

## Context

ADR 0002 ran training as one payload on a Vast 4090, but the data was pushed over SSH from the
laptop, a local watcher had to stay alive to pull results and destroy the box, and every box
installed its Python packages from PyPI and downloaded the base model from Hugging Face. On
2026-09-29 a box sat 22 minutes "loading" (host pulling the 9 GB devel image) while the local wait
had no limit, and the owner asked for the whole run to be one job that does not depend on the
laptop, pulls what it needs from our own bucket, and cleans up after itself.

## Decision

One command, `task all` in `finetuning/` (settings in `~/.config/openjevx/config.json`, data in
`~/openjevx/data`): dataprep → leakage check → smoke run → full run → gate. The GPU step:

1. **Inputs from R2.** The laptop uploads the shard to the private bucket `openjevx-train`
   (`shards/<version>/`) and a `job.env` (`runs/<run>/job.env`) holding only signed links and the
   kill token (uploaded from memory via `sec`, never written to disk). The box gets one signed link
   to `job.env` and nothing else.
2. **Runtime: stock PyTorch image + pip on the box.** The box uses `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
   (PyTorch + CUDA included) and installs the few packages in `finetuning/train/requirements-box.txt` from
   PyPI, with retries; the base model comes from Hugging Face. Both download on the data-center network.
   Tried and dropped the same day: a prebuilt runtime in R2 (built on a box: slow and hit the smoke time
   limit; built on the owner's Mac: the home connection is too slow to upload several GB). Only the shard
   (tens of MB) goes up from the owner's machine.
3. **The box does the whole job.** Clone the pushed commit, train, calibrate, export ONNX on CUDA,
   quantize to 8-bit, **fail if the 8-bit ONNX is over 750 MB**, upload `openjevx.w8.onnx`,
   `checkpoint.tar.gz` (trainable), `eval_report.json`, `job.log`, `status.json` to
   `runs/<run>/` through signed PUT links.
4. **The box destroys itself** by calling `https://openjevx-destroy.pages.dev/destroy` (a Cloudflare
   Pages Function in `finetuning/destroy-worker/`; it holds the Vast key and destroys only
   `openjevx-*-DESTROY-AFTER` boxes when given the kill token). A timer on the box destroys it at
   `DEADLINE_HOURS` whatever happens. No long-lived key ever goes on the box.
5. **The laptop only waits on R2.** `gpu/vast.py` polls `runs/<run>/status.json`, downloads the
   results to `~/openjevx/data/work/runs/<run>/out/`, and destroys the box itself if it is somehow
   still alive. A box that is not `running` within 15 minutes (`VAST_START_MINUTES`) is destroyed
   and the next machine is tried, up to 3.
6. **Everything Cloudflare is in code**: `finetuning/cloud_setup.sh` (`task cloud-setup`) creates
   or refreshes the bucket, the endpoint and its secrets from `sec`, and checks the endpoint.

GPU providers stay pluggable (`gpu/<name>.py`, prints `RUN_DIR=`); Kaggle is not used.

## Options considered

- **Vast persistent volume** for packages and model: cheap, but it lives on one physical machine,
  so every run would be tied to that machine being free; it also cannot hold the Docker image.
  Rejected for now; worth revisiting only if we train daily on one reliable host.
- **Own image in GHCR** with packages and model baked in: fast and reproducible, but a registry
  pull per box and an image build per change. Kept as a later option if the R2 bundle is not enough.
- **R2-backed Docker registry** (Cloudflare serverless-registry): needs a Worker; blocked by token scope.
- **SSH push + local watcher** (ADR 0002): rejected; ties the run to the laptop being awake.

## Consequences

- A run survives the laptop sleeping or the agent session ending; results wait in R2.
- ~~Only the stock PyTorch image comes from outside; data, packages and the model come from our bucket.~~
- ~~The first run after changing `requirements-box.txt`, the image or the base model is slower
  (it rebuilds the bundle); every later run starts from R2.~~
  Corrected 2026-10-03: these two lines described the prebuilt R2 runtime that decision 2 dropped. Only the data
  comes from our bucket; packages come from PyPI and the base model from Hugging Face on every run
  (`finetuning/train/run_job.sh`).
- `openjevx-train` holds shards, runs, checkpoints, bundles and gate reports and must stay private;
  only released models go to the public `openjevx` bucket.

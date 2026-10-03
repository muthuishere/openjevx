#!/usr/bin/env python3
"""Provider for ft.py train (config.providers.vast): one rented GPU box that does the whole job itself.

usage: vast.py SHARD_DIR SHARD [--max-price-per-hour 0.6] [--timeout-hours 7]
  SHARD is smoke or full; SHARD_DIR holds openjevx-<SHARD>-{train.jsonl.gz,eval.jsonl.gz,run.json}.
  Needs $OPENJEVX_KILL_TOKEN (ft.py runs this under `sec run OPENJEVX_KILL_TOKEN`).

1. HEAD must be on origin (the box clones that exact commit).
   The box uses the stock PyTorch image and installs the few extra packages from PyPI itself.
2. Upload the shard to R2 (private bucket) and a job.env of signed links + the kill token (from memory).
3. Rent the cheapest verified GPU; onstart clones the commit and runs finetuning/train/run_job.sh detached.
   The box trains, uploads its results to R2 and destroys itself through the destroy endpoint.
4. Wait for runs/<run>/status.json in R2, download the results, make sure the box is gone.
   Backstop: if the box is still alive past the deadline, destroy it from here.
Prints RUN_DIR=<dir>; results land in <dir>/out/ (model/ = the model folder the server runs,
openjevx.w8.onnx, checkpoint.tar.gz, eval_report.json, job.log).
"""
import argparse, hashlib, json, os, shlex, subprocess, sys, tarfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "finetuning"))
import paths  # noqa: E402
import r2  # noqa: E402

KILL_URL = os.environ.get("OPENJEVX_KILL_URL", "https://openjevx-destroy.pages.dev/destroy")
IMAGE = os.environ.get("VAST_IMAGE", "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")  # has PyTorch + CUDA
SSH_PUB = Path.home() / ".ssh/id_ed25519_muthuishere.pub"
MAX_W8_MB = os.environ.get("MAX_W8_MB", "750")
START_MIN = int(os.environ.get("VAST_START_MINUTES", "15"))
# A box can report "running" while its launcher never runs our onstart (seen 2026-10-03: the host's /.launch looped on
# "ssh: command not found"). run_job.sh uploads its log every 2 minutes, so no log this long after "running" = dead box.
LOG_MIN = int(os.environ.get("VAST_FIRST_LOG_MINUTES", "12"))
RESULTS = {"W8": "openjevx.w8.onnx", "MODEL": "model.tar", "CKPT": "checkpoint.tar.gz", "REPORT": "eval_report.json",
           "LOG": "job.log", "STATUS": "status.json"}


def out(*args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, cwd=ROOT).stdout.strip()


def pick_offer(max_price, skip=()):
    gpu = os.environ.get("GPU_NAME") or "RTX_4090"
    # Bandwidth is billed per GB on top of dph and varies ~10x by host (2026-10-04: $0.11 vs $0.011 for the same
    # ~2.8 GB of wheels + base model): skip hosts above VAST_MAX_GB_COST $/GB either way.
    gb = os.environ.get("VAST_MAX_GB_COST", "0.01")
    query = (f"gpu_name={gpu} num_gpus=1 verified=true reliability>0.99 disk_space>=80 inet_down>=200 dph<={max_price} "
             f"inet_down_cost<={gb} inet_up_cost<={gb}")
    offers = json.loads(out("vastai", "search", "offers", query, "-o", "dph", "--raw"))
    offers = [o for o in offers if o.get("machine_id") not in skip]
    if not offers:
        raise SystemExit(f"no verified {gpu} offer at <= ${max_price}/h")
    o = offers[0]
    print(f"offer {o['id']}: {o.get('geolocation')} ${o['dph_total']:.3f}/h reliability {o.get('reliability2', 0):.3f}", flush=True)
    return o["id"], o.get("machine_id")


def state(iid):
    rows = json.loads(out("vastai", "show", "instances", "--raw") or "[]")
    return next((r.get("actual_status") or "unknown" for r in rows if r["id"] == iid), None)


def alive(iid):
    rows = json.loads(out("vastai", "show", "instances", "--raw") or "[]")
    return any(r["id"] == iid for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("shard", choices=["smoke", "full"])
    ap.add_argument("--max-price-per-hour", type=float, default=0.6)
    ap.add_argument("--timeout-hours", type=float, default=7)
    a = ap.parse_args()
    if a.shard == "smoke":  # a smoke run takes minutes; never let a stuck one bill for hours
        a.timeout_hours = min(a.timeout_hours, 1.5)
    kill_token = os.environ.get("OPENJEVX_KILL_TOKEN") or sys.exit("OPENJEVX_KILL_TOKEN missing; run under sec run")
    shard_dir = Path(a.shard_dir).resolve()
    files = {s: shard_dir / f"openjevx-{a.shard}-{s}" for s in ("train.jsonl.gz", "eval.jsonl.gz", "run.json")}
    for f in files.values():
        if not f.exists():
            raise SystemExit(f"missing {f}; run the package step first")

    if out("git", "status", "--porcelain", "--untracked-files=no", "--", "finetuning"):
        raise SystemExit("finetuning/ has uncommitted changes; commit them so the box runs what you tested")
    ref = out("git", "rev-parse", "HEAD")
    if ref not in out("git", "ls-remote", "origin"):
        print("pushing HEAD so the box can clone it", flush=True)
        out("git", "push", "origin", "HEAD")
    repo = out("git", "remote", "get-url", "origin")
    if repo.startswith("git@"):  # the box has no GitHub SSH key: clone the public repo over HTTPS
        repo = "https://" + repo[4:].replace(":", "/", 1)

    name = f"{shard_dir.name}-{time.strftime('%Y%m%d-%H%M')}"
    run_dir = paths.WORK / "runs" / name
    (run_dir / "out").mkdir(parents=True)
    print(f"RUN_DIR={run_dir}", flush=True)
    link_s = int((a.timeout_hours + 3) * 3600)

    shard_key = f"shards/{shard_dir.name}"
    for s, f in files.items():
        print(f"uploading {f.name} to r2://{r2.PRIVATE}/{shard_key}/", flush=True)
        r2.put(f, f"{shard_key}/{s}")
    if (shard_dir / "licence-excluded.json").exists():  # what the licence filter left out, kept with the run
        r2.put(shard_dir / "licence-excluded.json", f"runs/{name}/licence-excluded.json")
    env = {"SHARD": a.shard, "MAX_W8_MB": MAX_W8_MB,
           "MODEL_VERSION": shard_dir.name.removeprefix("v").removesuffix("-smoke"), "DEADLINE_HOURS": str(a.timeout_hours),
           "KILL_URL": KILL_URL, "KILL_TOKEN": kill_token,
           "TRAIN_URL": r2.link_get(f"{shard_key}/train.jsonl.gz", link_s),
           "EVAL_URL": r2.link_get(f"{shard_key}/eval.jsonl.gz", link_s),
           "RUN_URL": r2.link_get(f"{shard_key}/run.json", link_s)}
    env.update({f"PUT_{k}_URL": r2.link_put(f"runs/{name}/{v}", link_s) for k, v in RESULTS.items()})
    body = "".join(f"{k}={shlex.quote(v)}\n" for k, v in env.items())
    r2.client().put_object(Bucket=r2.PRIVATE, Key=f"runs/{name}/job.env", Body=body.encode())
    job_env_url = r2.link_get(f"runs/{name}/job.env", link_s)

    # If anything before run_job.sh fails (clone, checkout), report it and destroy the box anyway:
    # run_job.sh's own cleanup only exists once it runs.
    job = ("command -v git >/dev/null && command -v curl >/dev/null || { apt-get update -qq && "
           "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git curl ca-certificates; }; "
           f"export GIT_TERMINAL_PROMPT=0 JOB_ENV_URL={shlex.quote(job_env_url)}; "
           "curl -fsSL --retry 5 -o /root/job.env \"$JOB_ENV_URL\" && . /root/job.env; "
           f"{{ git clone --filter=blob:none {shlex.quote(repo)} /root/job && "
           f"git -C /root/job checkout --detach {ref} && "
           "bash /root/job/finetuning/train/run_job.sh; } || { "
           "[ -f /root/JOB_FAILED ] || [ -f /root/JOB_COMPLETE ] || { "
           "echo 'box setup failed before run_job.sh'; "
           "printf '{\"status\":\"failed\",\"stage\":\"setup\"}' > /root/status.json; "
           "curl -fsS -X PUT -T /root/job.log \"$PUT_LOG_URL\"; curl -fsS -X PUT -T /root/status.json \"$PUT_STATUS_URL\"; "
           "curl -fsS -X POST \"$KILL_URL\" -H \"X-Kill-Token: $KILL_TOKEN\" "
           "-d \"{\\\"instance_id\\\": ${CONTAINER_ID:-${VAST_CONTAINERLABEL#C.}}}\"; }; }")
    onstart = ("mkdir -p /root/.ssh; chmod 700 /root/.ssh; "
               f"printf '%s\\n' {shlex.quote(SSH_PUB.read_text().strip())} >> /root/.ssh/authorized_keys; "
               "chmod 600 /root/.ssh/authorized_keys; "
               f"python3 -c 'import os,sys; os.setsid(); os.execvp(\"bash\", [\"bash\", \"-c\", sys.argv[1]])' "
               f"{shlex.quote('{ ' + job + '; } > /root/job.log 2>&1')} &")  # the whole job, not just its last command
    # Some hosts never finish pulling the image. Give each box START_MIN minutes to reach "running",
    # otherwise destroy it and try the next machine (up to 3).
    # VAST_SKIP_MACHINES=id,id: hosts already seen failing in an earlier run.
    bad, iid = {int(m) for m in os.environ.get("VAST_SKIP_MACHINES", "").split(",") if m.strip()}, None
    for attempt in range(3):
        offer, machine = pick_offer(a.max_price_per_hour, bad)
        created = json.loads(out("vastai", "create", "instance", str(offer), "--image", IMAGE,
                                 "--disk", "80", "--ssh", "--direct", "--label", f"openjevx-{env['MODEL_VERSION']}-DESTROY-AFTER",
                                 "--onstart-cmd", onstart, "--raw"))
        iid = created["new_contract"]
        print(f"instance {iid} starting (attempt {attempt + 1})", flush=True)
        start_by = time.time() + START_MIN * 60
        while time.time() < start_by and state(iid) not in ("running", None):
            time.sleep(30)
        if state(iid) == "running":
            log_by = time.time() + LOG_MIN * 60
            while time.time() < log_by and f"runs/{name}/job.log" not in r2.ls(f"runs/{name}/"):
                time.sleep(30)
            if f"runs/{name}/job.log" in r2.ls(f"runs/{name}/"):
                break
            print(f"instance {iid} running but no job log after {LOG_MIN} min; destroying it and trying another machine",
                  flush=True)
        else:
            print(f"instance {iid} not running after {START_MIN} min; destroying it and trying another machine", flush=True)
        out("vastai", "destroy", "instance", str(iid), "-y")
        bad.add(machine)
        iid = None
    if iid is None:
        raise SystemExit("no box reached running after 3 tries")
    (run_dir / "instance.json").write_text(json.dumps({"instance_id": iid, "commit": ref, "run": name}, indent=2))
    print(f"instance {iid} running commit {ref[:10]}; results go to r2://{r2.PRIVATE}/runs/{name}/", flush=True)

    deadline = time.time() + a.timeout_hours * 3600 + 1800
    status = None
    while time.time() < deadline:
        if f"runs/{name}/status.json" in r2.ls(f"runs/{name}/"):
            r2.get(f"runs/{name}/status.json", run_dir / "out/status.json")
            status = json.loads((run_dir / "out/status.json").read_text())
            break
        if state(iid) == "exited":  # stopped by the host (seen 2026-10-02): it still "exists" and would be waited on
            print(f"instance {iid} exited before reporting a status; destroying it", flush=True)
            out("vastai", "destroy", "instance", str(iid), "-y")
            break
        if not alive(iid):
            time.sleep(60)  # the status upload may land just after the box goes
            if f"runs/{name}/status.json" not in r2.ls(f"runs/{name}/"):
                break
            continue
        time.sleep(120)
    for _ in range(20):  # the box destroys itself right after the status file; make sure it did
        if not alive(iid):
            break
        time.sleep(30)
    else:
        print(f"instance {iid} still alive; destroying it from here", flush=True)
        out("vastai", "destroy", "instance", str(iid), "-y")

    have = set(r2.ls(f"runs/{name}/"))
    for v in RESULTS.values():
        if f"runs/{name}/{v}" in have:
            r2.get(f"runs/{name}/{v}", run_dir / "out" / v)
    print(f"status: {status or 'none (box died or timed out)'}", flush=True)
    if not status or status.get("status") != "complete":
        raise SystemExit(f"run did not complete; log: {run_dir / 'out/job.log'}")
    model_tar = run_dir / "out/model.tar"
    if not model_tar.exists():
        raise SystemExit(f"run completed but model.tar is missing; log: {run_dir / 'out/job.log'}")
    with tarfile.open(model_tar) as tar:  # -> <run>/out/model/
        tar.extractall(run_dir / "out", filter="data")
    model_tar.unlink()
    print(f"model folder: {run_dir / 'out/model'}", flush=True)
    w8 = run_dir / "out/model/openjevx.w8.onnx"
    mb = w8.stat().st_size // 2**20
    print(f"8-bit ONNX: {w8} ({mb} MB, limit {MAX_W8_MB})", flush=True)
    if mb > int(MAX_W8_MB):
        raise SystemExit("8-bit ONNX is over the size limit")


if __name__ == "__main__":
    main()

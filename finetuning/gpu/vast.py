#!/usr/bin/env python3
"""Run one packaged shard on a rented Vast.ai GPU, end to end, then wait for the result.

Provider for ft.py train (config.providers.vast).
usage: vast.py SHARD_DIR SHARD [--max-price-per-hour 0.6] [--timeout-hours 7]
  SHARD is smoke or full; SHARD_DIR holds openjevx-<SHARD>-{train.jsonl.gz,eval.jsonl.gz,run.json}.

Steps: HEAD must be on origin (the box clones that exact commit) -> cheapest verified 4090 ->
vast-one-shot launch.py (onstart job + detached watcher that pulls /root/out/openjevx-model and
ALWAYS destroys the box) -> detached push_data.py -> wait until the box is gone, then check the
8-bit ONNX came back and is within MAX_W8_MB (750). Exit 0 only when it did.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ONE_SHOT = Path(os.environ.get("VAST_ONE_SHOT", Path.home() / ".claudedefault/skills/vast-one-shot/scripts"))
KEY = Path.home() / ".ssh/id_ed25519_muthuishere"
MAX_W8_MB = int(os.environ.get("MAX_W8_MB", "750"))


def out(*args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, cwd=ROOT).stdout.strip()


def detached(cmd, log):
    """Start cmd in its own session so it survives this shell (and the agent session)."""
    with open(log, "a") as f:
        subprocess.Popen([sys.executable, "-c", "import os,sys;os.setsid();os.execvp(sys.argv[1],sys.argv[1:])", *cmd],
                         stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=ROOT)


def pick_offer(max_price):
    query = f"gpu_name={os.environ.get('GPU_NAME') or 'RTX_4090'} num_gpus=1 verified=true reliability>0.99 disk_space>=80 inet_down>=200 dph<={max_price}"
    offers = json.loads(out("vastai", "search", "offers", query, "-o", "dph", "--raw"))
    if not offers:
        raise SystemExit(f"no verified GPU offer at <= ${max_price}/h")
    o = offers[0]
    print(f"offer {o['id']}: {o.get('geolocation')} ${o['dph_total']:.3f}/h reliability {o.get('reliability2', 0):.3f}")
    return o["id"]


def instance_alive(iid):
    rows = json.loads(out("vastai", "show", "instances", "--raw") or "[]")
    return any(r["id"] == iid for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("shard", choices=["smoke", "full"])
    ap.add_argument("--max-price-per-hour", type=float, default=0.6)
    ap.add_argument("--timeout-hours", type=float, default=7)
    a = ap.parse_args()
    shard_dir = Path(a.shard_dir).resolve()
    for s in ("train.jsonl.gz", "eval.jsonl.gz", "run.json"):
        if not (shard_dir / f"openjevx-{a.shard}-{s}").exists():
            raise SystemExit(f"missing {shard_dir}/openjevx-{a.shard}-{s}; run the package step first")

    if out("git", "status", "--porcelain", "--untracked-files=no", "--", "finetuning"):
        raise SystemExit("finetuning/ has uncommitted changes; commit them so the box runs what you tested")
    ref = out("git", "rev-parse", "HEAD")
    if ref not in out("git", "ls-remote", "origin"):
        print("pushing HEAD so the box can clone it")
        out("git", "push", "origin", "HEAD")
    repo = out("git", "remote", "get-url", "origin")

    run_dir = shard_dir.parent.parent / "runs" / f"{shard_dir.name}-{time.strftime('%Y%m%d-%H%M')}"
    run_dir.mkdir(parents=True)
    print(f"RUN_DIR={run_dir}", flush=True)
    artifacts = run_dir / "out"
    launch = json.loads(out(
        sys.executable, str(ONE_SHOT / "launch.py"), "--offer", str(pick_offer(a.max_price_per_hour)),
        "--repo", repo, "--ref", ref, "--run", f"SHARD={a.shard} MAX_W8_MB={MAX_W8_MB} bash finetuning/train/run_job.sh",
        "--label", f"openjevx-{a.shard}-DESTROY-AFTER", "--artifact-remote", "/root/out/openjevx-model",
        "--artifact-local", str(artifacts), "--disk", "80", "--timeout", str(int(a.timeout_hours * 3600)),
        "--ssh-private-key", str(KEY), "--ssh-public-key", str(KEY) + ".pub",
        "--watch-log", str(run_dir / "watch.log")))
    iid = launch["instance_id"]
    (run_dir / "instance.json").write_text(json.dumps(launch, indent=2))
    print(f"instance {iid} running commit {ref[:10]}; watcher log {run_dir / 'watch.log'}")
    detached([sys.executable, str(ROOT / "finetuning/train/push_data.py"), str(iid), str(shard_dir), a.shard],
             run_dir / "push.log")

    # The watcher owns pull + destroy; we only wait for the box to disappear.
    deadline = time.time() + a.timeout_hours * 3600 + 1800
    while instance_alive(iid):
        if time.time() > deadline:
            raise SystemExit(f"instance {iid} still alive past the timeout; check {run_dir / 'watch.log'}")
        time.sleep(120)
    w8 = artifacts / "openjevx.w8.onnx"
    if not w8.exists():
        raise SystemExit(f"box is gone but no 8-bit ONNX came back; see {run_dir / 'watch.log'}")
    mb = w8.stat().st_size // 2**20
    print(f"8-bit ONNX: {w8} ({mb} MB, limit {MAX_W8_MB})")
    if mb > MAX_W8_MB:
        raise SystemExit("8-bit ONNX is over the size limit")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Copy a private training shard onto a running vast.ai box over SSH, then write /root/in/READY.

Pairs with finetuning/train/run_job.sh (no-URL mode), which waits for READY.
Usage: push_data.py INSTANCE_ID SHARD_DIR SHARD_NAME [--key ~/.ssh/id_ed25519_muthuishere]
SHARD_DIR must hold openjevx-<SHARD_NAME>-{train.jsonl.gz,eval.jsonl.gz,run.json}.
Run it detached; it gives up after 40 minutes (the job then fails and the watcher destroys the box).
"""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_id")
    parser.add_argument("shard_dir")
    parser.add_argument("shard")
    parser.add_argument("--key", default=str(Path.home() / ".ssh/id_ed25519_muthuishere"))
    args = parser.parse_args()
    files = [Path(args.shard_dir) / f"openjevx-{args.shard}-{suffix}"
             for suffix in ("train.jsonl.gz", "eval.jsonl.gz", "run.json")]
    for path in files:
        if not path.exists():
            raise SystemExit(f"missing {path}")
    opts = ["-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-i", args.key]
    deadline = time.time() + 40 * 60
    while time.time() < deadline:
        shown = subprocess.run(["vastai", "show", "instance", args.instance_id, "--raw"],
                               text=True, capture_output=True)
        if shown.returncode == 0:
            info = json.loads(shown.stdout)
            if info.get("ssh_host") and info.get("ssh_port"):
                direct = (info.get("ports") or {}).get("22/tcp") or []
                host = info.get("public_ipaddr") if direct else info["ssh_host"]
                port = str(direct[0].get("HostPort") if direct else info["ssh_port"])
                target = f"root@{host}"
                ready = subprocess.run(["ssh", *opts, "-p", port, target, "mkdir -p /root/in"],
                                       capture_output=True)
                if ready.returncode == 0:
                    copied = subprocess.run(["scp", *opts, "-P", port, *map(str, files), f"{target}:/root/in/"])
                    if copied.returncode == 0:
                        subprocess.run(["ssh", *opts, "-p", port, target, "touch /root/in/READY"], check=True)
                        print(f"pushed {len(files)} files to {args.instance_id}", flush=True)
                        return
        time.sleep(20)
    raise SystemExit("gave up: box never became reachable")


if __name__ == "__main__":
    main()

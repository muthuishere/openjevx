#!/usr/bin/env python3
"""Detached Vast finisher: pull, publish, and always destroy the paid instance."""

import argparse
import json
import subprocess
import time
from pathlib import Path


def run(*args, check=True, capture=False):
    return subprocess.run(args, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


def instance(instance_id):
    result = run("vastai", "show", "instance", str(instance_id), "--raw", capture=True)
    return json.loads(result.stdout)


def ssh_args(info):
    return [
        "ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-i", str(Path.home() / ".ssh/id_ed25519_muthuishere"),
        "-p", str(info["ssh_port"]), f"root@{info['ssh_host']}",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_id", type=int)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--output", default=str(Path.home() / "muthu/gitworkspace/openjevx/artifacts"))
    args = parser.parse_args()
    destination = Path(args.output)
    deadline = time.time() + args.timeout
    published = False
    try:
        while time.time() < deadline:
            info = instance(args.instance_id)
            if info.get("actual_status") == "running" and info.get("ssh_host"):
                probe = run(*ssh_args(info),
                            "test -f /root/OPENJEVX_COMPLETE && echo complete || "
                            "(test -f /root/OPENJEVX_FAILED && echo failed || echo running)",
                            check=False, capture=True)
                state = probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else "unreachable"
                print(time.strftime("%F %T"), state, flush=True)
                if state == "complete":
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    run("scp", "-r", "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null", "-i",
                        str(Path.home() / ".ssh/id_ed25519_muthuishere"), "-P", str(info["ssh_port"]),
                        f"root@{info['ssh_host']}:/root/openjevx-model", str(destination))
                    run("uv", "run", "--with", "huggingface_hub", "python",
                        str(Path(__file__).with_name("publish_hf.py")), str(destination))
                    published = True
                    break
                if state == "failed":
                    run(*ssh_args(info), "tail -200 /root/openjevx-job.log", check=False)
                    raise RuntimeError("Vast job failed")
            time.sleep(60)
        if not published:
            raise TimeoutError("Vast job did not complete before watchdog timeout")
    finally:
        run("vastai", "destroy", "instance", str(args.instance_id), "-y", check=False)


if __name__ == "__main__":
    main()

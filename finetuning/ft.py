#!/usr/bin/env python3
"""OpenJevX fine-tuning pipeline. Every step fails loudly.

Settings: ~/.config/openjevx/config.json (or $OPENJEVX_FT_CONFIG). On first run it is created from
finetuning/config.example.json; edit it there. Relative data paths are relative to the repo root.

  ft.py dataprep        generate the rule-labelled sets into data/
  ft.py validate        adapter dry-parse + leakage check (writes .local/ft/leaked_keys.json)
  ft.py package [--smoke]   build the shard in .local/ft/<version>[-smoke]/
  ft.py train [--smoke]     run the shard on config.provider (gpu/<provider>.py), get the 8-bit ONNX back
  ft.py gate MODEL.onnx     serve MODEL locally and score it; exit 1 if it misses the config thresholds
  ft.py all             every step in order, with a smoke run before the full run

A provider is finetuning/gpu/<name>.py taking `SHARD_DIR SHARD` plus its own options from
config.providers.<name>; it must leave SHARD_DIR/../<dir>_<shard>_run/out/openjevx.w8.onnx.
"""
import json, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

FT = Path(__file__).resolve().parent
ROOT = FT.parent
CFG_PATH = Path(os.environ.get("OPENJEVX_FT_CONFIG", Path.home() / ".config/openjevx/config.json"))
if not CFG_PATH.exists():
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FT / "config.example.json", CFG_PATH)
    print(f"created {CFG_PATH} from config.example.json", flush=True)
CFG = json.loads(CFG_PATH.read_text())
WORK = ROOT / ".local" / "ft"
PY = str(ROOT / ".local/eval/venv/bin/python") if (ROOT / ".local/eval/venv/bin/python").exists() else sys.executable


def sh(*args, env=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=ROOT, env={**os.environ, **(env or {})})


def shard_dir(smoke):
    return WORK / (CFG["version"] + ("-smoke" if smoke else ""))


def dataprep():
    for gen in CFG["dataprep"]["generators"]:
        sh(PY, FT / "dataprep" / gen)


def validate():
    WORK.mkdir(parents=True, exist_ok=True)
    sh(PY, FT / "datavalidate/check_adapter.py")
    v = CFG["datavalidate"]
    args = [PY, FT / "datavalidate/leak_check.py", "--out", WORK / "leaked_keys.json"]
    for f in v["leak_train"]:
        args += ["--train", ROOT / f]
    for f in v["leak_eval"]:
        if (ROOT / f).exists():
            args += ["--eval", ROOT / f]
        else:
            print(f"note: test file {f} not present, skipped in the leakage check")
    sh(*args)


def package(smoke):
    d = CFG["dataprep"]
    args = [PY, FT / "dataprep/package_shards.py", "--out", shard_dir(smoke),
            "--budget-gb", d["budget_gb"], "--exclude-keys", WORK / "leaked_keys.json"]
    if smoke:
        args.append("--smoke")
    for x in d["extra_train"]:
        args += ["--extra-train", ROOT / x["file"]] * x.get("repeat", 1)
    for f in d["extra_eval"]:
        args += ["--extra-eval", ROOT / f]
    sh(*args, env=CFG["train"])


def train(smoke):
    name = CFG["provider"]
    opts = CFG["providers"].get(name, {})
    flags = [f"--{k.replace('_', '-')}={v}" for k, v in opts.items() if k != "gpu"]
    sh(PY, FT / "gpu" / f"{name}.py", shard_dir(smoke), "smoke" if smoke else "full", *flags,
       env={"MAX_W8_MB": str(CFG["model"]["max_w8_mb"]), "GPU_NAME": opts.get("gpu", "")})
    d = shard_dir(smoke)
    return d.parent / f"{d.name}_{'smoke' if smoke else 'full'}_run" / "out" / "openjevx.w8.onnx"


def gate(model):
    g = CFG["gate"]
    model = Path(model).resolve()
    mb = model.stat().st_size // 2**20
    if mb > CFG["model"]["max_w8_mb"]:
        sys.exit(f"GATE FAIL: {model.name} is {mb} MB, limit {CFG['model']['max_w8_mb']}")
    binary, runtime = ROOT / ".local/openjevx", next(ROOT.glob(".local/libonnxruntime.*"), None)
    if not binary.exists() or runtime is None:
        sh("task", "build")
        runtime = next(ROOT.glob(".local/libonnxruntime.*"))
    srv = WORK / "gate-server"
    srv.mkdir(parents=True, exist_ok=True)
    (srv / "openjevx.json").write_text(json.dumps(
        {"listen": f"127.0.0.1:{g['port']}", "device": "cpu", "model": str(model), "runtime": str(runtime)}))
    proc = subprocess.Popen([str(binary)], cwd=srv, stdout=open(srv / "server.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{g['port']}/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    sys.exit(f"gate server exited; see {srv / 'server.log'}")
                time.sleep(1)
        sys.path.insert(0, str(FT / "gate"))
        import gate_eval
        url = f"http://127.0.0.1:{g['port']}/v1/systemone"
        report, failures = {"model": str(model), "size_mb": mb, "files": {}}, []
        for f in g["files"]:
            m = gate_eval.score(url, ROOT / f)
            report["files"][f] = m
            print(f"{f:40s} n={m['n']:6d} accuracy {m['accuracy']:6.1%} right&confident {m['confident_right']:6.1%} "
                  f"confidently WRONG {m['confident_wrong']:5.1%}", flush=True)
            if f in g["basics_files"]:
                if m["confident_right"] < g["min_basics_confident_right"]:
                    failures.append(f"{f}: right&confident {m['confident_right']:.1%} < {g['min_basics_confident_right']:.0%}")
                if m["confident_wrong"] > g["max_basics_confident_wrong"]:
                    failures.append(f"{f}: confidently wrong {m['confident_wrong']:.1%} > {g['max_basics_confident_wrong']:.0%}")
        j = subprocess.run(["bash", FT / "gate/jevx13.sh", g["jevx_profile"]], text=True, capture_output=True, cwd=ROOT)
        print(j.stdout, flush=True)
        correct = int(j.stdout.split("correct ")[-1].split("/")[0]) if "correct " in j.stdout else 0
        report["jevx13_correct"] = correct
        if correct < g["min_jevx13_correct"]:
            failures.append(f"jevx 13 fundamentals: {correct}/13 < {g['min_jevx13_correct']}")
        report["pass"], report["failures"] = not failures, failures
        (WORK / f"gate-{model.stem}.json").write_text(json.dumps(report, indent=2))
        print("GATE PASS" if not failures else "GATE FAIL:\n  " + "\n  ".join(failures))
        if failures:
            sys.exit(1)
    finally:
        proc.terminate()


def main():
    step, rest = (sys.argv[1] if len(sys.argv) > 1 else "help"), sys.argv[2:]
    smoke = "--smoke" in rest
    if step == "dataprep": dataprep()
    elif step == "validate": validate()
    elif step == "package": package(smoke)
    elif step == "train": print(train(smoke))
    elif step == "gate": gate(rest[0])
    elif step == "all":
        dataprep(); validate()
        package(True); train(True)
        package(False); gate(train(False))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()

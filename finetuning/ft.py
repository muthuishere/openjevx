#!/usr/bin/env python3
"""OpenJevX fine-tuning pipeline. Every step fails loudly.

Settings: ~/.config/openjevx/config.json (or $OPENJEVX_FT_CONFIG). On first run it is created from
finetuning/config.example.json; edit it there. Data paths are relative to the data folder
(~/openjevx/data or $OPENJEVX_DATA, see finetuning/paths.py).

  ft.py dataprep        generate the rule-labelled sets into <data>/{train,eval,gate}
  ft.py validate        adapter dry-parse + leakage check (writes <data>/work/leak/leaked_keys.json;
                        fails if a gate question asks about a state that is in training)
  ft.py package [--smoke]   build the shard in <data>/work/shards/<version>[-smoke]/ and run every
                            row through the trainer's build_item on CPU (needs uv)
  ft.py train [--smoke]     run the shard on config.provider (gpu/<provider>.py), get the 8-bit ONNX back
  ft.py gate MODEL          serve MODEL (a model folder, or a .onnx for old models) locally and score it;
                           exit 1 if it misses the config thresholds
  ft.py all             every step in order, with a smoke run before the full run

A provider is finetuning/gpu/<name>.py taking `SHARD_DIR SHARD` plus its own options from
config.providers.<name>; it must print RUN_DIR=<dir> and leave the model folder <dir>/out/model/
(runs go under <data>/work/runs/).
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
sys.path.insert(0, str(FT))
import paths  # noqa: E402
WORK = paths.WORK
PY = str(ROOT / ".local/eval/venv/bin/python") if (ROOT / ".local/eval/venv/bin/python").exists() else sys.executable


def sh(*args, env=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=ROOT, env={**os.environ, **(env or {})})


BUILD_DEPS = ["torch", "transformers==4.57.6", "datasets", "safetensors", "huggingface_hub", "sentencepiece",
              "laya @ git+https://github.com/NandhaKishorM/laya.git@9d955671415fc19f069b9cc998928075c1f255ec"]


def shard_dir(smoke):
    return WORK / "shards" / (CFG["version"] + ("-smoke" if smoke else ""))


def dataprep():
    for gen in CFG["dataprep"]["generators"]:
        sh(PY, FT / "dataprep" / gen)


def validate():
    (WORK / "leak").mkdir(parents=True, exist_ok=True)
    sh(PY, FT / "datavalidate/check_adapter.py")
    v = CFG["datavalidate"]
    # Exact matches are removed from training at package time; a gate question about a state we train
    # on (reworded or not) fails validate; the same for an eval file only warns.
    args = [PY, FT / "datavalidate/leak_check.py", "--out", WORK / "leak/leaked_keys.json", "--fail-on-gate"]
    extra = [x["file"] for x in CFG["dataprep"].get("extra_train", [])]
    for f in dict.fromkeys(v["leak_train"] + extra):
        args += ["--train", paths.DATA / f]
    for f in v["leak_eval"]:
        if (paths.DATA / f).exists():
            args += ["--gate" if f.startswith("gate/") else "--eval", paths.DATA / f]
        else:
            print(f"note: test file {f} not present, skipped in the leakage check")
    try:
        sh(*args)
    except subprocess.CalledProcessError:
        sys.exit("validate: a gate file shares question states with training (see the table above); "
                 "regenerate the gate so it only asks about unseen states")


def package(smoke):
    d = CFG["dataprep"]
    args = [PY, FT / "dataprep/package_shards.py", "--out", shard_dir(smoke),
            "--budget-gb", d["budget_gb"], "--exclude-keys", WORK / "leak/leaked_keys.json"]
    if smoke:
        args.append("--smoke")
    for x in d["extra_train"]:
        args += ["--extra-train", paths.DATA / x["file"]] * x.get("repeat", 1)
    for f in d["extra_eval"]:
        args += ["--extra-eval", paths.DATA / f]
    sh(*args, env=CFG["train"])
    # Prove the trainer itself accepts every row (all of them for smoke, 500 per source for full) on CPU,
    # so a data bug fails here instead of on a rented GPU.
    d = shard_dir(smoke)
    shards = [d / "train_smoke.jsonl.gz", d / "eval_smoke.jsonl.gz"] if smoke else [d / "train.jsonl.gz", d / "eval.jsonl.gz"]
    sh("uv", "run", "-q", *[x for dep in BUILD_DEPS for x in ("--with", dep)], "python",
       FT / "datavalidate/check_build.py", *shards, *([] if smoke else ["--sample", "500"]))


def train(smoke):
    name = CFG["provider"]
    opts = CFG["providers"].get(name, {})
    flags = [f"--{k.replace('_', '-')}={v}" for k, v in opts.items() if k != "gpu"]
    # The kill token reaches the provider only through sec; the provider hands it to the box in job.env.
    cmd = ["sec", "run", "OPENJEVX_KILL_TOKEN", "--", PY, FT / "gpu" / f"{name}.py", shard_dir(smoke),
           "smoke" if smoke else "full", *flags]
    print("+", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            env={**os.environ, "MAX_W8_MB": str(CFG["model"]["max_w8_mb"]), "GPU_NAME": opts.get("gpu", ""),
                                 "OPENJEVX_R2_BUCKET": CFG["storage"]["private_bucket"],
                                 "OPENJEVX_KILL_URL": CFG["storage"]["destroy_url"]})
    run_dir = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("RUN_DIR="):
            run_dir = Path(line.strip().split("=", 1)[1])
    if proc.wait() != 0 or run_dir is None:
        sys.exit(f"{name} provider failed")
    model = run_dir / "out" / "model"
    if not (model / "config.json").exists():
        sys.exit(f"{name} provider left no model folder at {model}")
    return model


def free_port():
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def gate(model):
    g = dict(CFG["gate"])
    # Never trust a fixed port: an old server already listening there answers /health and gets scored
    # instead of this model (that happened once). Use a free port and a temporary jevx profile.
    g["port"] = free_port()
    model = Path(model).resolve()
    graph = model / "openjevx.w8.onnx" if model.is_dir() else model
    if not graph.is_file() or (model.is_dir() and not (model / "config.json").is_file()):
        sys.exit(f"GATE FAIL: {model} is not a model folder (openjevx.w8.onnx + config.json) or a .onnx file")
    mb = graph.stat().st_size // 2**20
    # A run's folder is <run>/out/model: name its gate report after the run, not "model".
    label = model.stem if not model.is_dir() else model.parent.parent.name if model.name == "model" else model.name
    if mb > CFG["model"]["max_w8_mb"]:
        sys.exit(f"GATE FAIL: {model.name} is {mb} MB, limit {CFG['model']['max_w8_mb']}")
    binary, runtime = ROOT / ".local/openjevx", next(ROOT.glob(".local/libonnxruntime.*"), None)
    if not binary.exists() or runtime is None:
        sh("task", "build")
        runtime = next(ROOT.glob(".local/libonnxruntime.*"))
    srv = WORK / "gate" / "server"
    srv.mkdir(parents=True, exist_ok=True)
    (srv / "openjevx.json").write_text(json.dumps(
        {"listen": f"127.0.0.1:{g['port']}", "device": "cpu", "model": str(model), "runtime": str(runtime)}))
    proc = subprocess.Popen([str(binary)], cwd=srv, stdout=open(srv / "server.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if proc.poll() is not None:
                sys.exit(f"gate server exited; see {srv / 'server.log'}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{g['port']}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        time.sleep(2)
        if proc.poll() is not None:  # e.g. bind failed but something else answered /health
            sys.exit(f"gate server exited; see {srv / 'server.log'}")
        sys.path.insert(0, str(FT / "gate"))
        import gate_eval
        url = f"http://127.0.0.1:{g['port']}/v1/systemone"
        report, failures = {"model": str(model), "size_mb": mb, "files": {}}, []
        for f in g["files"]:
            m = gate_eval.score(url, paths.DATA / f)
            report["files"][f] = m
            print(f"{f:40s} n={m['n']:6d} accuracy {m['accuracy']:6.1%} right&confident {m['confident_right']:6.1%} "
                  f"confidently WRONG {m['confident_wrong']:5.1%}", flush=True)
            if f in g["basics_files"]:
                if m["confident_right"] < g["min_basics_confident_right"]:
                    failures.append(f"{f}: right&confident {m['confident_right']:.1%} < {g['min_basics_confident_right']:.0%}")
                if m["confident_wrong"] > g["max_basics_confident_wrong"]:
                    failures.append(f"{f}: confidently wrong {m['confident_wrong']:.1%} > {g['max_basics_confident_wrong']:.0%}")
        profile = f"openjevx-gate-{g['port']}"
        subprocess.run(["jevx", "profile", "add", profile, url, "--model", f"openjevx-gate-{label}"],
                       capture_output=True, check=True)
        try:
            j = subprocess.run(["bash", FT / "gate/jevx13.sh", profile], text=True, capture_output=True, cwd=ROOT)
        finally:
            subprocess.run(["jevx", "profile", "remove", profile], capture_output=True)
        print(j.stdout, flush=True)
        correct = int(j.stdout.split("correct ")[-1].split("/")[0]) if "correct " in j.stdout else 0
        report["jevx13_correct"] = correct
        if correct < g["min_jevx13_correct"]:
            failures.append(f"jevx 13 fundamentals: {correct}/13 < {g['min_jevx13_correct']}")
        report["pass"], report["failures"] = not failures, failures
        (WORK / "gate" / f"{label}.json").write_text(json.dumps(report, indent=2))
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

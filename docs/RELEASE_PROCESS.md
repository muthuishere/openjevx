# Release process

How an OpenJevX model goes from data to a public release. Every step is a command in this repo; the
decisions behind them are in `docs/adr/` (0003 export and release, 0009 the GPU job, 0006 your own data).

Where things live: code in the repo, settings in `~/.config/openjevx/config.json` (from
`finetuning/config.example.json`), data in `~/openjevx/data` (`finetuning/paths.py`), runs and model folders
in `~/openjevx/data/work/`, private run outputs in the R2 bucket `openjevx-train`.

## 0. Before you start

- Set `"version"` in `~/.config/openjevx/config.json` (and `finetuning/config.example.json`) to the new version.
- `task cloud-setup` (in `finetuning/`) once per machine: R2 bucket, self-destroy endpoint, secrets from `sec`.
- The owner's home connection is slow: never upload large files from this machine. The GPU box downloads its
  own packages; models move between buckets server-side.

## 1. Data

```bash
cd finetuning
python3 ft.py dataprep                   # generators -> <data>/{train,eval,gate}
python3 dataprep/import_csv.py yours.csv --name mydata --add-to-config   # optional: your own decisions
```

Labels come from evaluating a rule or from what really happened, never from a model's guess.

## 2. Validate

```bash
python3 ft.py validate
```

- Adapter dry-parse, and the leakage check: no test or gate question may appear in training.
- Gate files must not share a **state** with training (a reworded question over the same facts is still a
  leak). A gate overlap fails validate.

## 3. Package

```bash
python3 ft.py package --smoke && python3 ft.py package
```

Builds the shard, removes leaked questions, and runs every smoke row (500 per source for full) through the
trainer's own `build_item` on CPU, so a data bug fails here, not on a rented GPU.

## 4. Smoke run, then full run (one GPU job each)

```bash
python3 ft.py train --smoke     # ~10-20 min, ~$0.10
python3 ft.py train             # full run; v0.5.2: 791k decisions, 2.5 h, ~$1.25
```

The box (stock PyTorch image on Vast.ai) downloads the shard from R2, installs `train/requirements-box.txt`,
trains, calibrates, exports, quantizes to 8-bit (fails over 750 MB), builds the **model folder**, uploads it and
the trainable checkpoint to R2, and destroys itself. A box that doesn't start in 15 minutes is replaced; the
log reaches R2 every 2 minutes. Results land in `~/openjevx/data/work/runs/<version>-<time>/out/`.

## 5. The model folder

```
openjevx-<version>/
  openjevx.w8.onnx   8-bit weight-only graph
  config.json        version, calibration temperatures (choice, score, noul), max_len, special ids, sha256
  tokenizer.json
```

The server loads a folder (`"model"` in `openjevx.json`, `-model`, or `OPENJEVX_MODEL`) and uses that model's
own temperatures. Nothing is compiled into the binary. `finetuning/export/make_model_folder.py` builds one by
hand.

## 6. Gate (must pass to ship)

```bash
python3 ft.py gate ~/openjevx/data/work/runs/<run>/out/model
```

Serves the folder on a free port and scores: the basics gates (right & confident >= 90%, confidently wrong
<= 2%), the logs gate, the held-out test sets, and the 13 fundamentals through jevx (>= 12/13). Compare
with the previous release; nothing may regress. The owner can accept a named miss; write it down in the release
notes.

## 7. Benchmarks

Run base / previous / new on the public Jev benchmarks on Hugging Face (LocalLLaMA/typed-decisions,
Luni/laya-jev-benchmark, open-system-one-bench sources, laya card extras) and ours; record in
`llmresults/NN-<version>-benchmarks.md` with which sets are external vs in-distribution. Measure CPU latency
and size.

## 8. Publish (after the gate, in this order)

1. **Hugging Face** `muthuishere/openjevx`: upload the model folder files and the checkpoint; update the card
   (container first, Python second, gate + benchmark tables, known misses); tag the previous version.
2. **R2 public bucket** `openjevx`: copy the model server-side from `openjevx-train` (no re-upload).
3. **GitHub**: bump versions (README, Taskfile, launcher, Docker), `MODEL_DIR=<folder> task package` (small binary
   per platform + `openjevx-model-<version>.tar.gz`), tag `v<version>`, create the release with the gate and
   benchmark numbers, via a PR to `main`.
4. **Site** (GitHub Pages, `site/`):
   - results page: the new gate and benchmark numbers;
   - **fine-tune page** (https://muthuishere.github.io/openjevx/finetune/) must let someone download and do
     everything for this version: the model folder archive (`openjevx-model-<version>.tar.gz`, GitHub release),
     the trainable checkpoint / fine-tune kit (public R2 link + sha256), the Hugging Face repo, the sample CSV
     (`/openjevx/examples/decisions.csv`), and the step-by-step commands with this version's numbers (cost, time,
     decisions). Check every link on the deployed page returns 200 after the Pages deploy.
5. **Users**: jevx caches answers by model name for 7 days, so tell upgraders to run `jevx cache clear`, or use
   a versioned model name in the profile (`openjevx-<version>`).

## 9. After

- `vastai show instances` shows no `openjevx-*` box.
- Write the run report in `llmresults/` (cost, time, gate, benchmarks, what went wrong).
- Restart local servers on the new folder and check `/health` shows the new version and sha256.

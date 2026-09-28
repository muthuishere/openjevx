# Basic-reasoning eval harness

Independent, self-checkable accuracy probe for the OpenJevX **server**
(the shipped release binary: Go + the int8 ONNX graph), not the PyTorch
training checkpoint.

It exists because the project's own accuracy claim (`docs/adr/0001-base-model.md`,
0.774 overall) is measured once, in Python, on the fp16 GPU checkpoint
(`scripts/train_openjevx.py:benchmark`), against `LocalLLaMA/typed-decisions`.
That number is never re-checked against the artifact users actually run: the
Go tokenizer, the Go request encoder (`internal/decide`), and the int8
CPU ONNX export. This harness closes that gap without needing GPU or
Hugging Face access (see `docs/eval/2026-09-28-basic-reasoning-eval.md` for
why Hugging Face wasn't reachable when this was written).

Every case has a gold answer derived mechanically from the `state` text
(a numeric comparison, an explicit stated fact, a count, ...), not from an
external label file, so grading needs no reference dataset and is exactly
reproducible.

## Run it

```bash
# 1. start the server (release binary or `task run`), e.g.:
./openjevx   # listens on 127.0.0.1:21118, password "adminadmin" by default

# 2. from this directory:
python3 gen_cases.py   # writes cases.jsonl (regenerate to change the suite)
python3 run_eval.py    # posts every case to /v1/systemone, writes
                        # results.jsonl, errors.jsonl, summary.json,
                        # and prints the accuracy/calibration report
```

Edit the `URL`/`AUTH` constants at the top of `run_eval.py` if the server
runs on a different address or password.

## Extending it

`gen_cases.py` is plain Python — add a new category function that appends
`add(category, state, questions, gold)` rows. Keep every gold label
mechanically derivable from the `state` string so a case can never be
ambiguous.

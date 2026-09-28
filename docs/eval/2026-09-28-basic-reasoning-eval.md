# OpenJevX basic-reasoning eval — 2026-09-28

Status: informational (not an ADR — no decision is recorded here, only
findings and recommendations for the next training run).

## Why this exists

The report was requested after testing showed the model getting "very
basic questions" wrong. This is a 1,260-decision, mechanically-graded
accuracy run against the actual shipped server (the `v0.2.0` GitHub
release binary — Go server + the embedded int8 ONNX graph, the exact
artifact a `npx git+https://github.com/muthuishere/openjevx.git` install
runs), plus a diagnosis of what's missing that let this ship unnoticed.

**Headline: the complaint is correct, and it's reproducible.** Overall
accuracy on unambiguous, mechanically-checkable questions was **69.8%**
(879/1,260), and it fails outright on two categories any human would call
"basic": explicit negation ("the account is **not** verified" →
model says verified, 80% confident) and finding the minimum of three
explicit dollar amounts (28.75% accuracy — **worse than the 33% you'd get
guessing**). Full breakdown below.

## Important context first: this is not a chatbot

Before the results — the single most important finding for whoever is
testing this. **OpenJevX has no free-text chat interface.** Per
`docs/adr/0001-base-model.md`, it's a "Jev-class decision model," not a
chat model: `answerdotai/ModernBERT-large` (an encoder, not a
decoder/generator) plus a fixed-shape decision head, trained with RLCD to
answer exactly three runtime-defined question shapes — `choice`, `score`,
`noul` (yes/no) — over an explicit `state`, and return calibrated
**probabilities**, never generated text. There's no autoregressive
generation anywhere in the graph (see `output_names=["logits",
"act_logits"]` in `scripts/export_onnx_gpu.py`).

The only endpoint is `POST /v1/systemone` and it requires this exact
shape:

```json
{
  "state": "Invoice total is $1200. Approved budget is $1000.",
  "questions": {
    "over_budget": {
      "type": "noul",
      "instructions": "Does the invoice total exceed the approved budget?"
    }
  }
}
```

If "basic questions" means something like `{"message": "What is the
capital of France?"}` or plain chat turns, the server correctly rejects
it — `400 questions missing` — because that's not what this model does.
That's not a training bug, but **it is a real product gap**: nothing in
`README.md` documents the `/v1/systemone` request shape, the three
question types, or gives a single example payload. A new user has no way
to discover the contract above except by reading Go source
(`cmd/openjevx/main.go`, `internal/decide/decide.go`). See recommendation
R1 below.

Everything past this point tests the model on the job it's actually
built for — typed decisions over an explicit state — and it still fails
on cases a human finds trivial.

## Methodology

- **Server under test**: `v0.2.0` GitHub release (`openjevx-linux-amd64.tar`),
  run unmodified, CPU device (the release's default/auto path — same as
  what most users get; no GPU in this environment). This is the exact
  artifact end users download.
- **Dataset**: the project's own accuracy claims (ADR 0001: 0.774 overall,
  trained/measured against `LocalLLaMA/typed-decisions`) come from Hugging
  Face. **`huggingface.co` is blocked by this sandbox's outbound network
  policy** (`403`, confirmed via the proxy status endpoint — policy
  denial, not a transient failure), so that benchmark could not be
  fetched or reproduced here. Instead, `scripts/eval/gen_cases.py`
  generates 1,260 decisions across 9 categories where the correct answer
  is mechanically derivable from the `state` text itself (a number
  comparison, a stated fact, a count, ...), so grading needs no external
  reference file and no human judgment call. This is a *different and
  arguably easier* benchmark than the paper one — these are the kinds of
  questions a human would call "basic," which is precisely what was
  reported as failing.
- **Scoring**: `scripts/eval/run_eval.py` posts every case to the live
  `/v1/systemone` endpoint and compares the argmax answer to the gold
  label. Full methodology, and how to regenerate or extend the suite, is
  in `scripts/eval/README.md`.
- **Reproduce**: `cd scripts/eval && python3 gen_cases.py && python3 run_eval.py`
  against a running server. Raw results are not committed (regenerate
  them — the harness is deterministic, seed `20260928`); `summary-2026-09-28.json`
  in this folder has the exact numbers below.

## Results

### Overall

| | |
|---|---|
| Decisions | 1,260 |
| Correct | 879 |
| **Accuracy** | **0.698** |
| Server errors | 0 |
| Latency (CPU) | p50 53.5 ms · p95 71.5 ms per case |

### By category

| Category | Accuracy | n | What it tests |
|---|---|---|---|
| C — explicit choice (value literally restated) | **1.000** | 180 | `state` says `'closed'`, question: what is the status? |
| D — count → ordinal bucket | 0.867 | 180 | "There are 8 tickets" → score bucket "some (4-10)" |
| F — two-fact conjunction | 0.828 | 180 | "hot AND humid" from two stated numbers |
| B — explicit stated fact (incl. negation) | 0.750 | 180 | "The X is/is not Y" → noul |
| A — numeric compare, clear gap (>15%) | 0.740 | 146 | "1200 vs 1000" → exceeds? |
| A — numeric compare, close gap (≤5%) | 0.554 | 74 | near-coinflip baseline is 0.50 |
| G — no-context trivia, noul | 0.500 | 64 | general knowledge, empty `state` |
| G — no-context trivia, choice | 0.333 | 96 | = random guessing among 4 options |
| **E — cheapest of three (explicit $ values)** | **0.288** | 160 | **worse than the 0.333 random baseline** |

### By question type

| Type | Accuracy | n |
|---|---|---|
| score | 0.867 | 180 |
| noul | 0.722 | 644 |
| choice | 0.592 | 436 |

### Calibration — is the confidence number trustworthy?

The product's core pitch (RLCD = "Reinforcement Learning for **Calibrated**
Decisions") is that the returned probability is meaningful, not just the
argmax. It's not holding up well:

| Model's stated confidence | n | mean confidence | actual accuracy | gap |
|---|---|---|---|---|
| 0.4–0.5 | 198 | 0.449 | 0.571 | −0.122 |
| 0.5–0.6 | 255 | 0.549 | 0.620 | −0.070 |
| 0.6–0.7 | 216 | 0.641 | **0.782** | −0.141 |
| 0.7–0.8 | 251 | 0.757 | **0.693** | +0.064 |
| 0.8–0.9 | 149 | 0.845 | 0.919 | −0.075 |
| 0.9–1.0 | 101 | 0.941 | 1.000 | −0.059 |

Accuracy is **not monotonically increasing** with stated confidence (it
drops from 0.782 at the 0.6–0.7 bucket to 0.693 at 0.7–0.8) — a
well-calibrated model shouldn't do that. Brier score for `noul` questions
is **0.182** (0 = perfect, 0.25 = coin-flip baseline), i.e. modestly
better than chance but not the tight calibration the RLCD training method
is supposed to buy.

### Concrete failures (the "basic questions" the report was about)

Explicit negation reversed with high confidence:

```
state: "The account is not verified."
Q: "Is the account verified?"
gold: false   model: true (77.5% confident)
```

```
state: "The ticket is not resolved."
Q: "Is the ticket resolved?"
gold: false   model: true (52.0% confident)
```

Numeric comparison wrong even with a huge, explicit gap:

```
state: "The current temperature is 1379.18C. The safety threshold temperature is 825C."
Q: "Does the current temperature exceed the safety threshold temperature?"
gold: true (1379 > 825, +67%)   model: false (63.9% confident)
```

"Cheapest of three" — simple minimum over three stated numbers, wrong
more often than not:

```
state: "Vendor A costs $1572. Vendor B costs $89. Vendor C costs $1852."
Q: "Which vendor is cheapest?"
gold: B ($89)   model: C (59.6% confident) — the actual most expensive vendor
```

For contrast, category C (the value is restated verbatim in `state`, no
comparison or arithmetic needed) is **100% accurate** — so the tokenizer,
request encoding, and decode path are all working correctly; the model
genuinely cannot compare or negate reliably, this isn't a plumbing bug.

## What's missing / diagnosis

Ranked by how much of the reported "very bad" experience each one
explains:

1. **No accuracy check for the artifact that actually ships.** The only
   test in the repo is `internal/bpe/bpe_test.go`, a tokenizer unit test.
   The 0.774 figure in ADR 0001 is produced once, in Python, on a GPU,
   against the **fp16/fp32 PyTorch checkpoint** — never against the
   **int8-quantized ONNX graph** that the release binary actually embeds
   and that every user runs on CPU. Nothing in `Taskfile.yml`, CI, or
   `scripts/publish_hf.py`'s gate re-verifies accuracy after the ONNX
   export + int8 quantization step in `docs/adr/0003-export-and-release.md`.
   Quantization-induced accuracy loss is a well-known failure mode and is
   the most likely single explanation for "much worse in practice than
   the published number." **This harness (`scripts/eval/`) is exactly
   that missing check** — it should be wired into `Taskfile.yml`
   (`task eval`) and re-run every time a new `openjevx.int8.onnx` is
   published, gated the same way `scripts/train_openjevx.py` already
   gates training at 0.70.

2. **The hardcoded calibration temperatures may not match the quantized
   graph.** `internal/decide/decide.go:22` hardcodes
   `temperature = [1.0255, 1.0340, 1.1479]`, fit in
   `scripts/train_openjevx.py:fit_temperature` against the fp16 model's
   *logits*. Int8 dynamic quantization changes the logit distribution
   (that's what the non-monotonic calibration table above looks like).
   Those three numbers were never refit against the int8 ONNX graph's
   actual output logits — worth checking directly against `LocalLLaMA/typed-decisions`
   once Hugging Face is reachable, or the next benchmark run should refit
   temperature on the exported graph, not just the training checkpoint.

3. **Negation handling.** "The X is not Y" → "Is the X Y?" reversed with
   high confidence, repeatably (category B, and a chunk of category F's
   "not both hot and humid" misses). This points at the RLCD training
   data (`LocalLLaMA/typed-decisions`) underrepresenting negated
   statements relative to affirmative ones — worth an explicit
   negation-focused data slice in the next training run per ADR 0002.

4. **Numeric/arithmetic comparison is weak**, including on wide, explicit
   gaps (category A "clear" is still only 74%) and the near-random
   "cheapest of three" case (28.75%, category E). A 395M-parameter
   masked-language-model encoder has no native arithmetic circuit; this
   is a known limitation class for BERT-style models, not unique to this
   fine-tune, but it's directly in-scope for a *decision* model whose
   whole job is comparing numbers in policy/business contexts (budgets,
   thresholds, SLAs — exactly the ADR 0001 example). Recommend
   augmenting the next training run's domain data with explicit
   comparison/min/max tasks over numeric spans, since this is unlikely to
   improve from more general RLCD data alone.

5. **No world-knowledge grounding, undocumented.** Category G (empty
   `state`, general trivia) performs at or below random-guessing
   baseline, which is expected and correct for this architecture — it's
   a classifier over provided context, not a knowledge base — but this
   is not stated anywhere a user would see it before testing. Anyone who
   throws a "basic" general-knowledge question at it with no `state` will
   get a confident-looking wrong answer and reasonably conclude "the
   model is bad," when the actual issue is a documentation gap about
   what `state` is for.

6. **No documented request/response contract.** `README.md` shows the
   base URL and the `jevx` CLI wiring, never the `/v1/systemone` JSON
   shape, the three question types, or a runnable `curl` example — the
   ones in this report are the first ones written down anywhere in the
   repo. This is the most likely reason "basic questions" were being
   sent in a shape the server was never built to accept.

## Recommendations (priority order)

- **R1** — Add a "Request format" section to `README.md` with the
  `choice`/`score`/`noul` shapes and 1–2 runnable `curl` examples (the
  budget example above is ready to paste in).
- **R2** — Add `task eval`: run `scripts/eval/` (or a HF-backed
  equivalent once reachable) against the built release artifact, and
  gate publishing on it the same way `scripts/publish_hf.py` already
  gates on the 0.70 PyTorch number — but measured on the **shipped int8
  ONNX graph**, not the training checkpoint.
- **R3** — Before the next training run (ADR 0002), refit
  `temperature_by_options`/`temperature` directly against the exported
  int8 ONNX logits, not just the fp16 checkpoint's.
- **R4** — Add a negation-focused and a numeric-comparison-focused data
  slice to the next domain dataset (still `state + questions +
  gold.probabilities` shape per ADR 0001), specifically targeting the two
  failure modes reproduced above.
- **R5** — Document the "no state, no answer" behavior explicitly (a
  line in `README.md` or the dashboard) so users don't mistake missing
  grounding for a broken model.

## Appendix

- Eval harness: `scripts/eval/gen_cases.py`, `scripts/eval/run_eval.py`,
  `scripts/eval/README.md`.
- Raw summary for this run: `docs/eval/summary-2026-09-28.json`.
- Server tested: GitHub release `v0.2.0`, `openjevx-linux-amd64.tar`,
  CPU device, password-protected default config, unmodified.
- Network note: `huggingface.co` (model weights host, and the official
  `LocalLLaMA/typed-decisions` benchmark) returned `403` from this
  sandbox's egress proxy — an organization policy denial, not a
  transient error. `github.com` and its release-asset CDN were reachable
  and are how the tested binary was obtained. Re-running against the
  official benchmark from an environment with Hugging Face access is the
  natural follow-up to confirm/refute the quantization-regression
  hypothesis in finding 1.

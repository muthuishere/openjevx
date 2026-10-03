# What you get

This page says only what the shipped OpenJevX server does today, and every number on it was measured.

## One server, one model, on your machine

- **One executable** (Go) with an HTTP API, a live dashboard on `/`, Prometheus metrics on `/metrics`, a health
  check on `/health`, and these recipes on `/recipes`. The executable holds no model: the model is the folder
  `model/` next to it (from `openjevx-model-<version>.tar.gz`); the Docker image copies it to `/app/model`, and
  `task build` puts it in `.local/model/`.
- **One model folder, version 0.5.2: `openjevx.w8.onnx`, 8-bit, 598 MB**, plus `config.json` (its calibration) and
  `tokenizer.json`.
- **It runs on your side.** Inference happens in your process, on your CPU, or on a GPU provider that loads and
  matches the CPU on a probe (CUDA, CoreML on macOS, DirectML on Windows). We never run your inference and never
  see your requests.

## Measured latency on CPU

Machine: Apple M5 Pro (Apple Silicon, 18 cores), macOS, CPU only, ONNX Runtime 1.29.0, measured in process with
`cmd/openjevx/bench_test.go` (20 rounds after 2 warm-up rounds) on the v0.5.0 model folder, which has the same
architecture and the same 598 MB graph as 0.5.2 (llmresults/12):

| request | tokens | p50 | p95 |
|---|---|---|---|
| one question, short state | 36 | 22 ms | 23 ms |
| one question, typical state | 164 | 85 ms | 86 ms |
| eight questions about one typical state | 1,319 | 666 ms | 690 ms |
| one question, long state | 912 | 540 ms | 577 ms |

Over HTTP the typical request is 85 ms p50. On x86, hosts with only AVX2 are about 2x slower than AVX-512 VNNI ones
(llmresults/14). Requests are answered one model run at a time; all the questions in one request go through the
model together, so asking several questions in one call is cheaper than several calls.

## Three question types

All three go to `POST /v1/systemone` as `{"state": ..., "questions": {NAME: {type, instructions, criteria}}}`.
Every answer below is real output from model 0.5.2 on server 0.5.7 (2026-10-03). Send
`-H "Authorization: Bearer <key>"` too when the server has an API key (README, "Credentials").

### Yes/no: `noul`

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": "INVOICE 2026-117  Issued: 2026-09-01\nConsulting services, September .......... 1,250.00 EUR\nTotal due: 1,250.00 EUR\nPayment terms: due within 45 days of the issue date.", "questions": {"claim": {"type": "noul", "instructions": "Does the invoice say the total is 1,500 EUR?"}}}'
```

```json
{"claim":{"action":{"act_probability":1},"answer_confidence":0.8829,"confidence":0.8829,"noul":0.1171,"probabilities":{"false":0.8829,"true":0.1171},"type":"noul"}}
```

`noul` is the probability of yes (here: no). `confidence` is how sure it is of whichever side it leans to.

### Pick one: `choice`

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": "I want my money back for order 88", "questions": {"tool": {"type": "choice", "instructions": "Which function should handle this request?", "criteria": {"refund_payment": "refund a charge", "create_invoice": "create a new invoice", "update_email": "change the billing email"}}}}'
```

```json
{"tool":{"action":{"act_probability":1},"answer_confidence":0.971,"choice":"refund_payment","confidence":0.971,"probabilities":{"create_invoice":0.0149,"refund_payment":0.971,"update_email":0.014},"type":"choice"}}
```

### Rating: `score`

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": "Checkout is down, customers are being charged twice", "questions": {"sev": {"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high"]}}}'
```

```json
{"sev":{"action":{"act_probability":1},"answer_confidence":0.9337,"confidence":0.9337,"probabilities":{"0":0.0293,"1":0.037,"2":0.9337},"score":1.9044,"type":"score"}}
```

`score` is the expected level (0 = the first criterion); `probabilities` are per level. The same question inside a
three-question request gives nearly the same answer: the runtime quantizes activations per call, so a probability
can move by a few thousandths with what else is in the request ([Several judgements in one call](17-several-judgements.md)).

Each block above is the `answers` object; the full response also carries `"model": "openjevx"` and `"usage": {"input_tokens": N, "output_tokens": 0, "server_ms": T}`.

## How good is it

It is a small, general model. The model 0.5.2 gate (`llmresults/13-v0.5.2-gate-misses.md`, questions kept out of
training): everyday basics (463) 98.7% right and confident, 0.9% confidently wrong; rule-checking basics (300) 100% /
0.0%; log alerts (900) 91.9% / 7.8%; jevx's 13 fundamentals 13/13. On the 18 recipe pages (re-run on 0.5.2) it is
right on most everyday calls and misses some that hosted Jev gets: a due date, a prompt injection, a password in a
URL, a force-push. Each page says how it did. Test it on your own questions before you rely on it.

## Getting a model fitted to your data

These are ways to get a model trained on your rules and records. They are **not** features of this binary;
the binary serves whichever 8-bit model you point `"model"` in `openjevx.json` at.

1. **Do it yourself** (ADR 0006, accepted): a guide and a step-by-step toolkit from your documents and records
   to labelled questions, a leakage check, training, the gate, and an 8-bit model you serve with this server.
2. **We train it for you** (ADR 0007, proposed): you bring documents and records, we do the data work and train
   on our GPUs, you confirm every rule, and you get the 8-bit model, its gate report and the trainable checkpoint.
   You run it yourself; we never serve your traffic.
3. **Inside your own cloud** (ADR 0008, proposed): an appliance from a cloud marketplace with the server and a
   training UI, so your data never leaves your account.

The ADRs are in `docs/adr/` in the repository.

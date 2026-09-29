# What you get

This page says only what the shipped OpenJevX server does today, and every number on it was measured.

## One server, one model, on your machine

- **One executable** (Go) with an HTTP API, a live dashboard on `/`, Prometheus metrics on `/metrics`, a health
  check on `/health`, and these recipes on `/recipes`. The Docker image and the release binaries carry the model
  inside; `task build` puts it next to the binary.
- **One model file: `openjevx.w8.onnx`, 8-bit, 598,046,561 bytes (598 MB)**, from the v0.4.0 release.
- **It runs on your side.** Inference happens in your process, on your CPU (or your GPU with a CUDA ONNX Runtime
  and `"device": "gpu"`). We never run your inference and never see your requests.

## Measured latency on CPU

Machine: Apple M5 Pro (Apple Silicon, 18 cores), macOS, CPU only (`"device": "cpu"`), ONNX Runtime 1.22.0,
the server built with `task build`. Other work was running on the machine at the time (load average about 7),
so treat these as a busy laptop, not a benchmark rig. 50 sequential requests after 5 warm-up requests, timed
with `curl -w '%{time_total}'` (so the numbers include HTTP):

| request | p50 | p95 | max |
|---|---|---|---|
| one yes/no question, one-sentence state | 178 ms | 192 ms | 211 ms |
| three questions (yes/no + pick-one + rating) in one request | 268 ms | 284 ms | 361 ms |

The loaded server used about 1.6 GB of resident memory. Requests are answered one model run at a time; all the
questions in one request go through the model together, so asking three questions in one call is cheaper than
three calls.

```bash
# the command used, per request body
for i in $(seq 50); do curl -s -o /dev/null -w '%{time_total}\n' localhost:21118/v1/systemone -d @body.json; done | sort -n
```

## Three question types

All three go to `POST /v1/systemone` as `{"state": ..., "questions": {NAME: {type, instructions, criteria}}}`.
Every answer below is real output from the server described above.

### Yes/no: `noul`

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "INVOICE 2026-117  Issued: 2026-09-01\nConsulting services, September .......... 1,250.00 EUR\nTotal due: 1,250.00 EUR\nPayment terms: due within 45 days of the issue date.", "questions": {"claim": {"type": "noul", "instructions": "Does the invoice say the total is 1,500 EUR?"}}}'
```

```json
{"claim":{"action":{"act_probability":1},"answer_confidence":0.8777,"confidence":0.8777,"noul":0.1223,"probabilities":{"false":0.8777,"true":0.1223},"type":"noul"}}
```

`noul` is the probability of yes (here: no). `confidence` is how sure it is of whichever side it leans to.

### Pick one: `choice`

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "I want my money back for order 88", "questions": {"tool": {"type": "choice", "instructions": "Which function should handle this request?", "criteria": {"refund_payment": "refund a charge", "create_invoice": "create a new invoice", "update_email": "change the billing email"}}}}'
```

```json
{"tool":{"action":{"act_probability":1},"answer_confidence":0.9484,"choice":"refund_payment","confidence":0.9484,"probabilities":{"create_invoice":0.0276,"refund_payment":0.9484,"update_email":0.024},"type":"choice"}}
```

### Rating: `score`

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "Checkout is down, customers are being charged twice", "questions": {"sev": {"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high"]}}}'
```

```json
{"sev":{"action":{"act_probability":1},"answer_confidence":0.7616,"confidence":0.7616,"probabilities":{"0":0.1161,"1":0.1223,"2":0.7616},"score":1.6455,"type":"score"}}
```

`score` is the expected level (0 = the first criterion); `probabilities` are per level. The same question inside a
three-question request gives the same answer ([Several judgements in one call](17-several-judgements.md)).

Each block above is the `answers` object; the full response also carries `"model": "openjevx"` and `"usage": {"input_tokens": N, "output_tokens": 0}`.

## How good is it

It is a small, general model: on the 18 recipes here it gets most everyday calls right, and it misses some that
hosted Jev gets (a due date, a prompt injection, a password in a URL). Each recipe page says how it did.
The committed gate numbers are in `llmresults/10-v041-baseline.md`: on the everyday-basics gate (469 questions
worded differently from training) v0.4 scores 66.3% accurate, 46.9% usable (right and past jevx's default
thresholds) and 17.9% confidently wrong. Test it on your own questions before you rely on it.

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

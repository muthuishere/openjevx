# Which file to open first

Yes/no per file path, sorted by the probability of yes.

## Input

`files.txt`:

```
internal/payments/gateway.go
internal/payments/retry.go
cmd/server/main.go
web/src/components/Button.tsx
internal/users/store.go
docs/README.md
```

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "internal/payments/gateway.go", "questions": {"hit": {"type": "noul", "instructions": "Is this file likely where a payment gateway timeout is handled?"}}}'
```

Real answer for the first input (openjevx v0.4.0 8-bit model, CPU):

```json
{"answers":{"hit":{"action":{"act_probability":1},"answer_confidence":0.6187,"confidence":0.6187,"noul":0.6187,"probabilities":{"false":0.3813,"true":0.6187},"type":"noul"}},"model":"openjevx","usage":{"input_tokens":47,"output_tokens":0}}
```

## Same thing with jevx

```bash
jevx rank --profile openjevx --no-context "Is this file likely where a payment gateway timeout is handled?" --top 3 < files.txt
```

Real output:

```
0.62  internal/payments/gateway.go
0.51  internal/payments/retry.go
0.38  cmd/server/main.go
```

## What to do with the answer

Read the top files first.

## How the local model did

Right: the two payments files come first.

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

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": "internal/payments/gateway.go", "questions": {"hit": {"type": "noul", "instructions": "Is this file likely where a payment gateway timeout is handled?"}}}'
```

Real answer for the first input (model 0.5.2, server 0.5.7, CPU):

```json
{"answers":{"hit":{"action":{"act_probability":1},"answer_confidence":0.6172,"confidence":0.6172,"noul":0.6172,"probabilities":{"false":0.3828,"true":0.6172},"type":"noul"}},"model":"openjevx","usage":{"input_tokens":47,"output_tokens":0,"server_ms":284.88}}
```

## Same thing with jevx

```bash
jevx rank --profile openjevx --no-context "Is this file likely where a payment gateway timeout is handled?" --top 3 < files.txt
```

Real output:

```
0.62  internal/payments/gateway.go
0.42  internal/payments/retry.go
0.15  internal/users/store.go
```

## What to do with the answer

Read the top files first.

## How the local model did

Right: gateway.go comes first (0.62), then retry.go (0.42); the third pick (store.go, 0.15) is not a candidate. Open the first two.

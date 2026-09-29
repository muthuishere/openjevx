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
while IFS= read -r l; do
  jq -nc --arg l "$l" '{state: $l, questions: {"hit": {"type": "noul", "instructions": "Is this file likely where a payment gateway timeout is handled?"}}}' \
  | curl -s localhost:21118/v1/systemone -d @- | jq -c --arg l "$l" '{p: .answers.hit.noul, file: $l}'
done < files.txt | sort -r
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"p":0.8046,"file":"internal/payments/retry.go"}
{"p":0.7514,"file":"internal/payments/gateway.go"}
{"p":0.4792,"file":"cmd/server/main.go"}
{"p":0.4288,"file":"internal/users/store.go"}
{"p":0.2612,"file":"web/src/components/Button.tsx"}
{"p":0.2107,"file":"docs/README.md"}
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

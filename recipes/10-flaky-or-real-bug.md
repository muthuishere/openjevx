# Flaky test or real bug?

Pick-one per CI failure.

## Input

`tests.txt`:

```
TestInvoiceTotal: expected 1250.00, got 1249.99 (rounding)
TestFetchRates: context deadline exceeded after 5s calling rates.example.com
TestUserLoad: panic: runtime error: invalid memory address or nil pointer dereference
TestWebsocketReconnect: connection reset by peer, passed on retry
```

## Call the local server

```bash
while IFS= read -r l; do
  jq -nc --arg l "$l" '{state: $l, questions: {"kind": {"type": "choice", "instructions": "Is this test failure a flaky test or a real bug?", "criteria": {"flaky": "timing, network or environment, passes on retry", "bug": "wrong result or crash in the code"}}}}' \
  | curl -s localhost:21118/v1/systemone -d @- | jq -c --arg l "$l" '{kind: .answers.kind.choice, p: .answers.kind.confidence}'
done < tests.txt
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"kind":"flaky","p":0.7032}
{"kind":"bug","p":0.8214}
{"kind":"bug","p":0.913}
{"kind":"bug","p":0.6976}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --lines tests.txt --choice kind="Is this test failure a flaky test or a real bug?|flaky=timing, network or environment, passes on retry;bug=wrong result or crash in the code" | jq -c "{line, kind: .answers.kind.verdict, p: .answers.kind.p}"
```

Real output:

```
{"line":1,"kind":"unsure","p":0.54}
{"line":2,"kind":"bug","p":0.6}
{"line":3,"kind":"bug","p":0.76}
{"line":4,"kind":"unsure","p":0.58}
```

## What to do with the answer

Fix the bugs first; re-run the flaky ones.

## How the local model did

Mixed. From curl, the network timeout and the websocket reset (flaky) come back as bugs and the rounding failure as flaky; only the nil pointer is clearly right. Through jevx two lines are unsure. Hosted Jev gets all four right.

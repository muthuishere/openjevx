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
curl -s localhost:21118/v1/systemone -d '{"state": "TestInvoiceTotal: expected 1250.00, got 1249.99 (rounding)", "questions": {"kind": {"type": "choice", "instructions": "Is this test failure a flaky test or a real bug?", "criteria": {"flaky": "timing, network or environment, passes on retry", "bug": "wrong result or crash in the code"}}}}'
```

Real answer for the first input (openjevx v0.4.0 8-bit model, CPU):

```json
{"answers":{"kind":{"action":{"act_probability":1},"answer_confidence":0.531,"choice":"bug","confidence":0.531,"probabilities":{"bug":0.531,"flaky":0.469},"type":"choice"}},"model":"openjevx","usage":{"input_tokens":63,"output_tokens":0}}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --lines tests.txt --choice kind="Is this test failure a flaky test or a real bug?|flaky=timing, network or environment, passes on retry;bug=wrong result or crash in the code"
```

Real output:

```
VERDICT  P     INPUT
unsure   0.54  TestInvoiceTotal: expected 1250.00, got 1249.99 (rounding)
bug      0.60  TestFetchRates: context deadline exceeded after 5s calling rates.example.com
bug      0.76  TestUserLoad: panic: runtime error: invalid memory address or nil pointer dereference
unsure   0.58  TestWebsocketReconnect: connection reset by peer, passed on retry
```

## What to do with the answer

Fix the bugs first; re-run the flaky ones.

## How the local model did

Mixed. Only the nil pointer (bug, 0.76) is clearly right; the network timeout comes back as a bug, and the rounding failure and the websocket reset are unsure. Hosted Jev gets all four right.

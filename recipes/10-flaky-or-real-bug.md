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
{"answers":{"kind":{"action":{"act_probability":1},"answer_confidence":0.7032,"choice":"flaky","confidence":0.7032,"probabilities":{"bug":0.2968,"flaky":0.7032},"type":"choice"}},"model":"openjevx","usage":{"input_tokens":63,"output_tokens":0}}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --lines tests.txt --choice kind="Is this test failure a flaky test or a real bug?|flaky=timing, network or environment, passes on retry;bug=wrong result or crash in the code"
```

Real output:

```
VERDICT  P     INPUT
flaky    0.71  TestInvoiceTotal: expected 1250.00, got 1249.99 (rounding)
bug      0.79  TestFetchRates: context deadline exceeded after 5s calling rates.example.com
bug      0.91  TestUserLoad: panic: runtime error: invalid memory address or nil pointer dereference
bug      0.63  TestWebsocketReconnect: connection reset by peer, passed on retry
```

The table output needs jevx v0.11.0 or newer.

## What to do with the answer

Fix the bugs first; re-run the flaky ones.

## How the local model did

Mixed: two of four right. The nil pointer is bug (0.91), but the rounding failure comes back as flaky (0.71) and the network timeout (0.79) and the websocket reset that passed on retry (0.63) as bugs. Hosted Jev gets all four right.

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

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": "TestInvoiceTotal: expected 1250.00, got 1249.99 (rounding)", "questions": {"kind": {"type": "choice", "instructions": "Is this test failure a flaky test or a real bug?", "criteria": {"flaky": "timing, network or environment, passes on retry", "bug": "wrong result or crash in the code"}}}}'
```

Real answer for the first input (model 0.5.2, server 0.5.7, CPU):

```json
{"answers":{"kind":{"action":{"act_probability":1},"answer_confidence":0.7101,"choice":"bug","confidence":0.7101,"probabilities":{"bug":0.7101,"flaky":0.2899},"type":"choice"}},"model":"openjevx","usage":{"input_tokens":63,"output_tokens":0,"server_ms":344.68}}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --lines tests.txt --choice kind="Is this test failure a flaky test or a real bug?|flaky=timing, network or environment, passes on retry;bug=wrong result or crash in the code"
```

Real output:

```
VERDICT  P     INPUT
bug      0.63  TestInvoiceTotal: expected 1250.00, got 1249.99 (rounding)
flaky    0.78  TestFetchRates: context deadline exceeded after 5s calling rates.example.com
bug      0.90  TestUserLoad: panic: runtime error: invalid memory address or nil pointer dereference
unsure   0.55  TestWebsocketReconnect: connection reset by peer, passed on retry
```

The table output needs jevx v0.11.0 or newer.

## What to do with the answer

Fix the bugs first; re-run the flaky ones.

## How the local model did

Three of four right: the rounding failure is a bug (0.63), the network timeout is flaky (0.78) and the nil pointer is a bug (0.90). The websocket reset that passed on retry comes back unsure (0.55) instead of flaky. Hosted Jev gets all four right.

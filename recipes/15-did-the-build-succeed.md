# Did the build succeed?

Yes/no over build output before reporting "done".

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "ok  core 2.1s\nok  web 0.8s\nFAIL payments 0.3s\nFAIL", "questions": {"ok": {"type": "noul", "instructions": "Did the build and tests succeed?"}}}' | jq -c .answers
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"ok":{"action":{"act_probability":1},"answer_confidence":0.7592,"confidence":0.7592,"noul":0.2408,"probabilities":{"false":0.7592,"true":0.2408},"type":"noul"}}
```

## Same thing with jevx

```bash
jevx is --profile openjevx --no-context "Did the build and tests succeed?" --in "$(printf 'ok  core 2.1s\nok  web 0.8s\nFAIL payments 0.3s\nFAIL')"
```

Real output:

```
unsure 0.24
```

## What to do with the answer

Never claim success unless this is a confident yes; for a pass/fail line, a plain `grep FAIL` is cheaper and exact.

## How the local model did

Leans no (P(yes) 0.24), but not below jevx's 0.2 no line, so jevx says unsure. Use `grep FAIL`.

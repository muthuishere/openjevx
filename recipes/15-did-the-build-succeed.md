# Did the build succeed?

Yes/no over build output before reporting "done".

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "ok  core 2.1s\nok  web 0.8s\nFAIL payments 0.3s\nFAIL", "questions": {"ok": {"type": "noul", "instructions": "Did the build and tests succeed?"}}}' | jq -c .answers
```

Real answer (model 0.5.2, server 0.5.7, CPU):

```json
{"ok":{"action":{"act_probability":1},"answer_confidence":0.8689,"confidence":0.8689,"noul":0.1311,"probabilities":{"false":0.8689,"true":0.1311},"type":"noul"}}
```

## Same thing with jevx

```bash
jevx is --profile openjevx --no-context "Did the build and tests succeed?" --in "$(printf 'ok  core 2.1s\nok  web 0.8s\nFAIL payments 0.3s\nFAIL')"
```

Real output:

```
no 0.13
```

## What to do with the answer

Never claim success unless this is a confident yes; for a pass/fail line, a plain `grep FAIL` is cheaper and exact.

## How the local model did

Right: no (P(yes) 0.13 from curl, `no 0.13` from jevx). A `grep FAIL` is still cheaper.

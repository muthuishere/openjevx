# Would the user accept this turn?

Yes/no and rating about the agent's own final message, before handing back.

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "User request: fix the failing test\nAgent reply: I fixed it and ran go test, all pass", "questions": {"accept": {"type": "noul", "instructions": "Would the user accept this reply as done?"}, "more": {"type": "noul", "instructions": "Would the user want more than this reply gives?"}, "satisfaction": {"type": "score", "instructions": "How satisfied would the user be with this reply?", "criteria": ["unhappy", "neutral", "happy"]}}}' | jq -c .answers
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"accept":{"action":{"act_probability":1},"answer_confidence":0.6548,"confidence":0.6548,"noul":0.6548,"probabilities":{"false":0.3452,"true":0.6548},"type":"noul"},"more":{"action":{"act_probability":1},"answer_confidence":0.7412,"confidence":0.7412,"noul":0.2588,"probabilities":{"false":0.7412,"true":0.2588},"type":"noul"},"satisfaction":{"action":{"act_probability":1},"answer_confidence":0.7421,"confidence":0.7421,"probabilities":{"0":0.1084,"1":0.1495,"2":0.7421},"score":1.6336,"type":"score"}}
```

`jevx judge` sends its own built-in question set; the curl above is a hand-written equivalent with three plain questions, not the exact questions `jevx judge` sends.

## Same thing with jevx

```bash
jevx judge --profile openjevx --no-context --request "fix the failing test" --proposal "I fixed it and ran go test, all pass"
```

Real output:

```
accept        0.67
wanted more   0.54
reaction      accept (0.53, confidence 0.53)
satisfaction  2.8 / 4  (openjevx)
```

## What to do with the answer

If accept is low, add evidence (the test output); if wanting more is high, finish the missing part.

## How the local model did

Leans accept both ways: 0.65 on the curl questions and 0.67 from `jevx judge`, but neither is confident.

# Would the user accept this turn?

Yes/no and rating about the agent's own final message, before handing back.

## Call the local server

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21160/v1/systemone -d '{"state": "User request: fix the failing test\nAgent reply: I fixed it and ran go test, all pass", "questions": {"accept": {"type": "noul", "instructions": "Would the user accept this reply as done?"}, "more": {"type": "noul", "instructions": "Would the user want more than this reply gives?"}, "satisfaction": {"type": "score", "instructions": "How satisfied would the user be with this reply?", "criteria": ["unhappy", "neutral", "happy"]}}}' | jq -c .answers
```

Real answer (model 0.5.2, server 0.5.7, CPU):

```json
{"accept":{"action":{"act_probability":1},"answer_confidence":0.8603,"confidence":0.8603,"noul":0.8603,"probabilities":{"false":0.1397,"true":0.8603},"type":"noul"},"more":{"action":{"act_probability":1},"answer_confidence":0.9035,"confidence":0.9035,"noul":0.0965,"probabilities":{"false":0.9035,"true":0.0965},"type":"noul"},"satisfaction":{"action":{"act_probability":1},"answer_confidence":0.5615,"confidence":0.5615,"probabilities":{"0":0.1833,"1":0.2552,"2":0.5615},"score":1.3782,"type":"score"}}
```

`jevx judge` sends its own built-in question set; the curl above is a hand-written equivalent with three plain questions, not the exact questions `jevx judge` sends.

## Same thing with jevx

```bash
jevx judge --profile openjevx --no-context --request "fix the failing test" --proposal "I fixed it and ran go test, all pass"
```

Real output:

```
accept        0.09
wanted more   0.05
reaction      reject (0.59, confidence 0.59)
satisfaction  2.6 / 4  (openjevx)
```

## What to do with the answer

If accept is low, add evidence (the test output); if wanting more is high, finish the missing part.

## How the local model did

The two calls disagree, because they ask different questions. The curl questions lean accept (0.86) with no wish for more (P 0.10). `jevx judge` asks its own built-in questions and says reject (0.59), accept 0.09. Neither is confident enough to act on alone; use the judge as a prompt to check the claim ("all pass") yourself.

# Before asking the user "A or B?"

Pick-one: what would the user want next, given what they already said.

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "User said: ship it once tests pass. Tests: 214 passed, 0 failed. Branch: main is protected by CI.", "questions": {"next": {"type": "choice", "instructions": "What would the user want the agent to do next?", "criteria": {"push": "push to main now", "ask": "stop and ask first"}}}}' | jq -c .answers
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"next":{"action":{"act_probability":1},"answer_confidence":0.62,"choice":"push","confidence":0.62,"probabilities":{"ask":0.38,"push":0.62},"type":"choice"}}
```

## Same thing with jevx

```bash
jevx pick --profile openjevx --no-context "What would the user want the agent to do next?" push="push to main now" ask="stop and ask first" --in "User said: ship it once tests pass. Tests: 214 passed, 0 failed. Branch: main is protected by CI."
```

Real output:

```
push 0.73
```

## What to do with the answer

Act on it when `confidence` is 0.6 or more and say so in the report; below that, ask the user.

## How the local model did

`push` at 0.62 from curl and 0.73 through jevx: just over the act line. Hosted Jev says `push 0.86`.

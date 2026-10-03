# Before asking the user "A or B?"

Pick-one: what would the user want next, given what they already said.

## Call the local server

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": "User said: ship it once tests pass. Tests: 214 passed, 0 failed. Branch: main is protected by CI.", "questions": {"next": {"type": "choice", "instructions": "What would the user want the agent to do next?", "criteria": {"push": "push to main now", "ask": "stop and ask first"}}}}' | jq -c .answers
```

Real answer (model 0.5.2, server 0.5.7, CPU):

```json
{"next":{"action":{"act_probability":1},"answer_confidence":0.6632,"choice":"push","confidence":0.6632,"probabilities":{"ask":0.3368,"push":0.6632},"type":"choice"}}
```

## Same thing with jevx

```bash
jevx pick --profile openjevx --no-context "What would the user want the agent to do next?" push="push to main now" ask="stop and ask first" --in "User said: ship it once tests pass. Tests: 214 passed, 0 failed. Branch: main is protected by CI."
```

Real output:

```
push 0.67
```

## What to do with the answer

Act on it when `confidence` is 0.6 or more and say so in the report; below that, ask the user.

## How the local model did

`push` at 0.66 from curl and 0.67 through jevx: just over jevx's 0.6 confidence line. Hosted Jev says `push 0.86`.

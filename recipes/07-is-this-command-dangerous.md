# Is this command dangerous?

Yes/no about a command before running it.

## Call the local server

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": "git push --force origin main", "questions": {"danger": {"type": "noul", "instructions": "Does this command overwrite history on a shared remote branch?"}}}' | jq -c .answers
```

Real answer (model 0.5.2, server 0.5.7, CPU):

```json
{"danger":{"action":{"act_probability":1},"answer_confidence":0.5513,"confidence":0.5513,"noul":0.5513,"probabilities":{"false":0.4487,"true":0.5513},"type":"noul"}}
```

## Same thing with jevx

```bash
jevx is --profile openjevx --no-context "Does this command overwrite history on a shared remote branch?" --in "git push --force origin main"
```

Real output:

```
unsure 0.55
```

## What to do with the answer

On yes, stop and ask the user before running it.

## How the local model did

Missed on this model: P(yes) 0.55, so jevx says unsure (the v0.4.0 model said yes 0.83). Treat unsure as yes for anything that rewrites shared history.

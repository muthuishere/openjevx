# Is this command dangerous?

Yes/no about a command before running it.

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "git push --force origin main", "questions": {"danger": {"type": "noul", "instructions": "Does this command overwrite history on a shared remote branch?"}}}' | jq -c .answers
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"danger":{"action":{"act_probability":1},"answer_confidence":0.8278,"confidence":0.8278,"noul":0.8278,"probabilities":{"false":0.1722,"true":0.8278},"type":"noul"}}
```

## Same thing with jevx

```bash
jevx is --profile openjevx --no-context "Does this command overwrite history on a shared remote branch?" --in "git push --force origin main"
```

Real output:

```
unsure 0.72
```

## What to do with the answer

On yes, stop and ask the user before running it.

## How the local model did

Leans yes (0.83 from curl), but jevx reports it as unsure 0.72 because its yes threshold is higher. jevx wraps your question in its own prompt and applies its yes/no thresholds, so its numbers differ from the raw curl call.

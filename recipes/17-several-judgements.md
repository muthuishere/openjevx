# Several judgements in one call

One request, three questions of the three types: yes/no, pick-one and rating.

## Call the local server

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21160/v1/systemone -d '{"state": "Checkout is down, customers are being charged twice", "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}, "team": {"type": "choice", "instructions": "Which team?", "criteria": {"web": "frontend", "api": "backend", "billing": "payments"}}, "sev": {"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high"]}}}' | jq -c .answers
```

Real answer (model 0.5.2, server 0.5.7, CPU):

```json
{"sev":{"action":{"act_probability":1},"answer_confidence":0.9339,"confidence":0.9339,"probabilities":{"0":0.0294,"1":0.0367,"2":0.9339},"score":1.9045,"type":"score"},"team":{"action":{"act_probability":1},"answer_confidence":0.5491,"choice":"billing","confidence":0.5491,"probabilities":{"api":0.3245,"billing":0.5491,"web":0.1264},"type":"choice"},"urgent":{"action":{"act_probability":1},"answer_confidence":0.9119,"confidence":0.9119,"noul":0.9119,"probabilities":{"false":0.0881,"true":0.9119},"type":"noul"}}
```

## Same thing with jevx

```bash
echo "Checkout is down, customers are being charged twice" | jevx ask --profile openjevx --no-context --noul urgent="Is this urgent?" --choice team="Which team?|web=frontend;api=backend;billing=payments" --score sev="How severe?|low;medium;high"
```

Real output:

```
sev              high       0.93
team             api        0.79
urgent           yes        0.91
```

## What to do with the answer

Page the chosen team when urgent is yes and severity is high.

## How the local model did

Urgent and severity are right both ways: urgent yes (0.91), severity high (0.93). The team is not settled on this model: the curl request gives billing at only 0.55 (below jevx's 0.6 confidence line), and jevx, which sends the same state from stdin, gives api 0.79. Ask the team question with more detail in the state before routing on it.

# Several judgements in one call

One request, three questions of the three types: yes/no, pick-one and rating.

## Call the local server

```bash
curl -s localhost:21118/v1/systemone -d '{"state": "Checkout is down, customers are being charged twice", "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}, "team": {"type": "choice", "instructions": "Which team?", "criteria": {"web": "frontend", "api": "backend", "billing": "payments"}}, "sev": {"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high"]}}}' | jq -c .answers
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"sev":{"action":{"act_probability":1},"answer_confidence":0.7616,"confidence":0.7616,"probabilities":{"0":0.1161,"1":0.1223,"2":0.7616},"score":1.6455,"type":"score"},"team":{"action":{"act_probability":1},"answer_confidence":0.8944,"choice":"billing","confidence":0.8944,"probabilities":{"api":0.054,"billing":0.8944,"web":0.0516},"type":"choice"},"urgent":{"action":{"act_probability":1},"answer_confidence":0.8499,"confidence":0.8499,"noul":0.8499,"probabilities":{"false":0.1501,"true":0.8499},"type":"noul"}}
```

## Same thing with jevx

```bash
echo "Checkout is down, customers are being charged twice" | jevx ask --profile openjevx --no-context --noul urgent="Is this urgent?" --choice team="Which team?|web=frontend;api=backend;billing=payments" --score sev="How severe?|low;medium;high"
```

Real output:

```
sev              high       0.71
team             billing    0.70
urgent           unsure     0.79
```

## What to do with the answer

Page the chosen team when urgent is yes and severity is high.

## How the local model did

Right on all three: urgent yes 0.85, team billing 0.89, severity high 0.76. jevx calls urgent unsure at 0.79 because of its stricter yes threshold.

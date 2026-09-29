# Route a ticket queue

Pick-one per ticket: which team owns it.

## Input

`tickets.jsonl`:

```
{"ticket": "The save button on the settings page is misaligned on mobile"}
{"ticket": "I was charged twice for my March invoice"}
{"ticket": "POST /v2/orders returns 500 when the cart has 0 items"}
{"ticket": "Dark mode makes the chart labels unreadable"}
{"ticket": "How do I export my data to CSV?"}
```

## Call the local server

```bash
while IFS= read -r l; do
  jq -nc --arg l "$l" '{state: ($l | fromjson), questions: {"team": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"web": "frontend or UI", "api": "backend or API", "billing": "payments and invoices", "docs": "how-to question"}}}}' \
  | curl -s localhost:21118/v1/systemone -d @- | jq -c --arg l "$l" '{team: .answers.team.choice, p: .answers.team.confidence}'
done < tickets.jsonl
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"team":"web","p":0.8999}
{"team":"billing","p":0.9089}
{"team":"api","p":0.8998}
{"team":"web","p":0.616}
{"team":"docs","p":0.8924}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --states tickets.jsonl --choice team="Which team should handle this ticket?|web=frontend or UI;api=backend or API;billing=payments and invoices;docs=how-to question" | jq -c "{line, team: .answers.team.verdict, p: .answers.team.p}"
```

Real output:

```
{"line":1,"team":"web","p":0.84}
{"line":2,"team":"billing","p":0.97}
{"line":3,"team":"api","p":0.71}
{"line":4,"team":"web","p":0.68}
{"line":5,"team":"docs","p":0.72}
```

## What to do with the answer

Assign each ticket to `choice`. Below 0.6 `confidence`, leave it for a human instead of guessing.

## How the local model did

All five routed as expected; the dark-mode ticket is the least sure (0.62).

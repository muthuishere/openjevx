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

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": {"ticket": "The save button on the settings page is misaligned on mobile"}, "questions": {"team": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"web": "frontend or UI", "api": "backend or API", "billing": "payments and invoices", "docs": "how-to question"}}}}'
```

Real answer for the first input (model 0.5.2, server 0.5.7, CPU):

```json
{"answers":{"team":{"action":{"act_probability":1},"answer_confidence":0.9788,"choice":"web","confidence":0.9788,"probabilities":{"api":0.0073,"billing":0.0066,"docs":0.0073,"web":0.9788},"type":"choice"}},"model":"openjevx","usage":{"input_tokens":58,"output_tokens":0,"server_ms":429.5}}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --states tickets.jsonl --choice team="Which team should handle this ticket?|web=frontend or UI;api=backend or API;billing=payments and invoices;docs=how-to question"
```

Real output:

```
VERDICT  P     INPUT
web      0.98  {"ticket": "The save button on the settings page is misaligned on mobile"}
billing  0.97  {"ticket": "I was charged twice for my March invoice"}
api      0.97  {"ticket": "POST /v2/orders returns 500 when the cart has 0 items"}
web      0.71  {"ticket": "Dark mode makes the chart labels unreadable"}
docs     0.96  {"ticket": "How do I export my data to CSV?"}
```

The table output needs jevx v0.11.0 or newer.

## What to do with the answer

Assign each ticket to `choice`. Below 0.6 `confidence`, leave it for a human instead of guessing.

## How the local model did

All five routed as expected; the dark-mode ticket is the least sure (0.71), the others 0.96-0.98.

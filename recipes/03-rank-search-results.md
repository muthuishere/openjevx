# Rank search results

Yes/no per result, sorted by the probability of yes.

## Input

`results.txt`:

```
Pricing of the enterprise plan
Refund policy for annual plans
Cancelling a subscription and getting money back
Changing your avatar
```

## Call the local server

```bash
while IFS= read -r l; do
  jq -nc --arg l "$l" '{state: $l, questions: {"hit": {"type": "noul", "instructions": "Does this page answer: how do I get my money back?"}}}' \
  | curl -s localhost:21118/v1/systemone -d @- | jq -c --arg l "$l" '{p: .answers.hit.noul, title: $l}'
done < results.txt | sort -r
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"p":0.779,"title":"Cancelling a subscription and getting money back"}
{"p":0.2938,"title":"Refund policy for annual plans"}
{"p":0.1588,"title":"Pricing of the enterprise plan"}
{"p":0.1291,"title":"Changing your avatar"}
```

## Same thing with jevx

```bash
jevx rank --profile openjevx --no-context "Does this page answer: how do I get my money back?" --top 3 < results.txt
```

Real output:

```
0.51  Cancelling a subscription and getting money back
0.31  Refund policy for annual plans
0.10  Pricing of the enterprise plan
```

## What to do with the answer

Open the top results only. The order matters more than the absolute numbers.

## How the local model did

The best hit ranks first, but "Refund policy for annual plans" scores only 0.29, so use the order, not a fixed cut-off.

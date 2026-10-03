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
curl -s localhost:21118/v1/systemone -d '{"state": "Pricing of the enterprise plan", "questions": {"hit": {"type": "noul", "instructions": "Does this page answer: how do I get my money back?"}}}'
```

Real answer for the first input (model 0.5.2, server 0.5.7, CPU):

```json
{"answers":{"hit":{"action":{"act_probability":1},"answer_confidence":0.8435,"confidence":0.8435,"noul":0.1565,"probabilities":{"false":0.8435,"true":0.1565},"type":"noul"}},"model":"openjevx","usage":{"input_tokens":46,"output_tokens":0,"server_ms":504.21}}
```

## Same thing with jevx

```bash
jevx rank --profile openjevx --no-context "Does this page answer: how do I get my money back?" --top 3 < results.txt
```

Real output:

```
0.64  Cancelling a subscription and getting money back
0.36  Refund policy for annual plans
0.16  Pricing of the enterprise plan
```

## What to do with the answer

Open the top results only. The order matters more than the absolute numbers.

## How the local model did

The best hit ranks first (0.64), but "Refund policy for annual plans" scores only 0.36, so use the order, not a fixed cut-off.

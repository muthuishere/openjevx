# Sort PR review comments

Pick-one per review comment, four ways.

## Input

`reviews.txt`:

```
nit: rename x to count
This loop never terminates when the list is empty
LGTM, nice work
Could we reuse the retry helper here instead of a new one?
Looks good to me
This builds the SQL with string concatenation from user input: SQL injection
```

## Call the local server

```bash
while IFS= read -r l; do
  jq -nc --arg l "$l" '{state: $l, questions: {"kind": {"type": "choice", "instructions": "What kind of review comment is this?", "criteria": {"must": "a real bug or risk that must be fixed", "should": "a reasonable change request", "nit": "style or naming only", "none": "praise or approval"}}}}' \
  | curl -s localhost:21118/v1/systemone -d @- | jq -c --arg l "$l" '{kind: .answers.kind.choice, p: .answers.kind.confidence}'
done < reviews.txt
```

Real answer (openjevx v0.4.0 8-bit model, CPU):

```json
{"kind":"nit","p":0.5296}
{"kind":"must","p":0.7984}
{"kind":"none","p":0.8707}
{"kind":"should","p":0.6667}
{"kind":"none","p":0.6574}
{"kind":"must","p":0.8373}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --lines reviews.txt --choice kind="What kind of review comment is this?|must=a real bug or risk that must be fixed;should=a reasonable change request;nit=style or naming only;none=praise or approval" | jq -c "{line, kind: .answers.kind.verdict}"
```

Real output:

```
{"line":1,"kind":"unsure"}
{"line":2,"kind":"unsure"}
{"line":3,"kind":"none"}
{"line":4,"kind":"unsure"}
{"line":5,"kind":"none"}
{"line":6,"kind":"must"}
```

## What to do with the answer

Handle `must` first, reply to `should`, batch the nits.

## How the local model did

Curl gets all six kinds as expected (nit 0.53 is the least sure); through jevx three lines come out unsure. jevx wraps your question in its own prompt and applies its yes/no thresholds, so its numbers differ from the raw curl call.

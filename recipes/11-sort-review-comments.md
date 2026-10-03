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

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": "nit: rename x to count", "questions": {"kind": {"type": "choice", "instructions": "What kind of review comment is this?", "criteria": {"must": "a real bug or risk that must be fixed", "should": "a reasonable change request", "nit": "style or naming only", "none": "praise or approval"}}}}'
```

Real answer for the first input (model 0.5.2, server 0.5.7, CPU):

```json
{"answers":{"kind":{"action":{"act_probability":1},"answer_confidence":0.748,"choice":"should","confidence":0.748,"probabilities":{"must":0.0181,"nit":0.1985,"none":0.0354,"should":0.748},"type":"choice"}},"model":"openjevx","usage":{"input_tokens":53,"output_tokens":0,"server_ms":364.58}}
```

## Same thing with jevx

```bash
jevx ask --profile openjevx --no-context --lines reviews.txt --choice kind="What kind of review comment is this?|must=a real bug or risk that must be fixed;should=a reasonable change request;nit=style or naming only;none=praise or approval"
```

Real output:

```
VERDICT  P     INPUT
should   0.66  nit: rename x to count
must     0.86  This loop never terminates when the list is empty
none     0.76  LGTM, nice work
should   0.60  Could we reuse the retry helper here instead of a new one?
none     0.66  Looks good to me
must     0.89  This builds the SQL with string concatenation from user input: SQL injection
```

The table output needs jevx v0.11.0 or newer.

## What to do with the answer

Handle `must` first, reply to `should`, batch the nits.

## How the local model did

Five of six right: the endless loop (`must`, 0.86), LGTM (`none`, 0.76), "Looks good to me" (`none`, 0.66), the SQL injection (`must`, 0.89) and the retry-helper request (`should`, 0.60). The one miss: "nit: rename x to count" comes back `should` (0.66) instead of `nit`, even with "nit:" in the text.

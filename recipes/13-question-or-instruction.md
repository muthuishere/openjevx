# Question or instruction?

Pick-one: answer the message, or change code?

## Call the local server

The calls send `OPENJEVX_API_KEY`, and the `jevx` lines use the `openjevx` profile that carries it: [the API key](README.md).

```bash
curl -s -H "Authorization: Bearer $OPENJEVX_API_KEY" localhost:21118/v1/systemone -d '{"state": "why does the retry helper sleep before the first attempt?", "questions": {"intent": {"type": "choice", "instructions": "What is the user asking for?", "criteria": {"answer": "a question to answer, no code change", "change": "an instruction to change code"}}}}' | jq -c .answers
```

Real answer (model 0.5.2, server 0.5.7, CPU):

```json
{"intent":{"action":{"act_probability":1},"answer_confidence":0.8948,"choice":"answer","confidence":0.8948,"probabilities":{"answer":0.8948,"change":0.1052},"type":"choice"}}
```

## Same thing with jevx

```bash
jevx pick --profile openjevx --no-context "What is the user asking for?" answer="a question to answer, no code change" change="an instruction to change code" --in "why does the retry helper sleep before the first attempt?"
```

Real output:

```
answer 0.89
```

## What to do with the answer

On `answer`, explain and do not touch the code.

## How the local model did

Right: answer (0.89).

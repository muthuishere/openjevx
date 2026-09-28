# OpenJevX

OpenJevX is an open-weight, Jev-compatible System One decision model built by specializing
[Laya](https://github.com/NandhaKishorM/laya). It answers runtime-defined `choice`, `score`,
and `noul` questions with calibrated probabilities in one non-autoregressive forward pass.

- Model: https://huggingface.co/muthuishere/openjevx
- ONNX release: https://github.com/muthuishere/openjevx/releases/tag/v0.1.0

## Run

Python 3.10+ and Node.js 18+ are the only prerequisites. The first run creates an isolated
runtime and downloads the model from Hugging Face; later runs reuse the cache.

```bash
npx github:muthuishere/openjevx --port 8000
```

Test the Jev-compatible endpoint:

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": {"body": "We were billed twice. Refund the duplicate."},
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "payments, invoices, and refunds",
          "technical": "bugs and outages"
        }
      }
    }
  }'
```

## Use With jevx

```bash
jevx profile add openjevx http://127.0.0.1:8000/v1/systemone --model openjevx
jevx profile use openjevx
jevx ask "We were billed twice" --noul refund:"Is a refund required?"
```

Use another device when available:

```bash
npx github:muthuishere/openjevx --device mps
npx github:muthuishere/openjevx --device cuda
```

## Provenance

- Decision architecture and runtime: Laya by ConvAI Innovations
- Encoder: `answerdotai/ModernBERT-large`
- Training method: RLCD, Reinforcement Learning for Calibrated Decisions
- License: Apache-2.0

This project is not affiliated with TypeSafe. Jev is a trademark of its respective owner.

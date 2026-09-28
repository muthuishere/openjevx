# OpenJevX

OpenJevX is a Jev-compatible decision model. No Python.

- Linux / macOS: `./README`
- Windows: `README.cmd`

The release executable contains the int8 ONNX model. `openjevx.json` selects `cpu` or `gpu`.
`gpu` does not fall back to CPU.

Model card: https://huggingface.co/muthuishere/openjevx

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

Device is `cpu` or `gpu` in `openjevx.json`. Do not use a Python or Node launcher.

## Provenance

- Decision architecture and runtime: Laya by ConvAI Innovations
- Encoder: `answerdotai/ModernBERT-large`
- Training method: RLCD, Reinforcement Learning for Calibrated Decisions
- License: Apache-2.0

This project is not affiliated with TypeSafe. Jev is a trademark of its respective owner.

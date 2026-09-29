"""MPS-backed NLI judge wrapper for quality_local.py.

Loads MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli from the local
HF cache (HF_HOME must already point at .local/quality/hf_home) and exposes
predict_batch(pairs) -> [{"label": "true"|"false", "conf": float, "probs": {...}}].

label "true" means the NLI model leans entailment (premise entails the
hypothesis statement); "false" means it leans contradiction. conf is the raw
softmax probability of whichever of entailment/contradiction is larger.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

REPO = Path(__file__).resolve().parents[2]


def flatten_state(state, prefix="", depth=0, out=None):
    if out is None:
        out = []
    if depth > 4 or len(out) > 40:
        return out
    if isinstance(state, dict):
        for k, v in state.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            flatten_state(v, key, depth + 1, out)
    elif isinstance(state, list):
        for i, v in enumerate(state[:10]):
            flatten_state(v, f"{prefix}[{i}]", depth + 1, out)
    else:
        out.append(f"{prefix}: {state}")
    return out


def state_text(state):
    parts = flatten_state(state)
    return "; ".join(parts)[:2000] or "(empty state)"


class NLIJudge:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "mps" else torch.float32
        print(f"[nli] loading {MODEL_ID} on {self.device} ({self.dtype})", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, l in self.id2label.items() if "entail" in l)
        self.contra_idx = next(i for i, l in self.id2label.items() if "contra" in l)
        try:
            probe = self.predict_batch([("The sky is blue.", "The sky has a color.")])
            print(f"[nli] sanity check: {probe}", flush=True)
        except Exception as e:
            print(f"[nli] sanity check failed (continuing): {e}", flush=True)

    def predict_batch(self, pairs):
        """pairs: list of (state, instructions) or (premise_text, hypothesis_text)."""
        premises = []
        hyps = []
        for state, hyp in pairs:
            premises.append(state_text(state) if not isinstance(state, str) or state.strip().startswith("{") else state)
            hyps.append(hyp)
        enc = self.tok(premises, hyps, truncation=True, max_length=384, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        out = []
        for row in probs:
            p_entail = float(row[self.entail_idx])
            p_contra = float(row[self.contra_idx])
            label = "true" if p_entail >= p_contra else "false"
            conf = max(p_entail, p_contra)
            out.append({"label": label, "conf": conf, "probs": {"entailment": p_entail, "contradiction": p_contra}})
        return out

    def unload(self):
        del self.model
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    j = NLIJudge()
    print(j.predict_batch([("The build failed with exit code 1.", "The build succeeded.")]))

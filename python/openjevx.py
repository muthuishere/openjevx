"""OpenJevX in Python: the same answers as the OpenJevX server, without the server.

    pip install onnxruntime tokenizers huggingface_hub

    from openjevx import OpenJevX
    jev = OpenJevX()                       # downloads openjevx.w8.onnx + tokenizer from Hugging Face once
    jev.decide({"age": 17}, {
        "adult": {"type": "noul", "instructions": "Is this person an adult (18 or older)?"},
    })

A port of the server's internal/decide (prompt layout, option markers, temperature, softmax); the
request and answer shapes match POST /v1/systemone. The recommended way to run OpenJevX is still the
container (see the README); this is for notebooks and Python services that want it in-process.
"""
import json
import math

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = "muthuishere/openjevx"
CLS, SEP, MASK = 50281, 50282, 50284
MAX_LEN, HEAD_MAX = 1024, 256  # used only when config.json has no max_len / head_max (as the server does)


def _criterion(v):
    return v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))


def _parse(q):
    t, crit = q["type"], q.get("criteria")
    if t == "choice":
        pairs = [(_criterion(c), "") for c in crit] if isinstance(crit, list) else \
            [(k, "" if v is None else _criterion(v)) for k, v in crit.items()]
        return 0, [k for k, _ in pairs], [k if not v else f"{k}: {v}" for k, v in pairs]
    if t == "score":
        return 1, [str(i) for i in range(len(crit))], [f"level {i}: {_criterion(c)}" for i, c in enumerate(crit)]
    if t == "noul":
        f, tr = "no, the statement does not hold", "yes, the statement holds"
        if isinstance(crit, dict):
            f = crit.get("false") or f
            tr = crit.get("true") or tr
        return 2, ["false", "true"], [f"false: {f}", f"true: {tr}"]
    raise ValueError(f"unknown type {t!r}")


class OpenJevX:
    def __init__(self, model=None, tokenizer=None, config=None, providers=None):
        model = model or hf_hub_download(REPO, "openjevx.w8.onnx")
        tokenizer = tokenizer or hf_hub_download(REPO, "tokenizer.json")
        config = config or hf_hub_download(REPO, "config.json")  # this model's own calibration temperatures
        cfg = json.load(open(config))
        t = cfg["temperature"]
        self.max_len, self.head_max = cfg.get("max_len") or MAX_LEN, cfg.get("head_max") or HEAD_MAX
        self.temperature = {0: t["choice"], 1: t["score"], 2: t["noul"]}
        self.tok = Tokenizer.from_file(tokenizer)
        self.session = ort.InferenceSession(model, providers=providers or ["CPUExecutionProvider"])

    def _enc(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def _item(self, state_ids, q):
        qtype, keys, options = _parse(q)
        head = self._enc(q["type"] + " question: " + q["instructions"].replace("[MASK]", " "))
        opts = [[MASK] + self._enc(" " + o.replace("[MASK]", " "))[:48] for o in options]
        budget = self.head_max - sum(map(len, opts))
        if budget < 16:
            per = max(4, (self.head_max - 16) // max(1, len(opts)))
            opts = [o[:per] for o in opts]
            budget = self.head_max - sum(map(len, opts))
        head = head[:max(8, budget)]
        ids, markers = [CLS] + head + [SEP], []
        for o in opts:
            markers.append(len(ids))
            ids += o
        ids.append(SEP)
        ids += state_ids[:max(0, self.max_len - len(ids) - 1)] + [SEP]
        return ids[:self.max_len], [m for m in markers if m < self.max_len], qtype, keys

    def decide(self, state, questions):
        """state: dict/list (sent as JSON) or str; questions: {id: {type, instructions, criteria}}."""
        text = state if isinstance(state, str) else json.dumps(state, separators=(",", ":"))
        state_ids = self._enc(text.replace("[MASK]", " "))
        qids = list(questions)
        items = [self._item(state_ids, questions[i]) for i in qids]
        n, length, width = len(items), max(len(i[0]) for i in items), max(len(i[1]) for i in items)
        ids = np.zeros((n, length), np.int64); attn = np.zeros((n, length), np.int64)
        pos = np.zeros((n, width), np.int64); pmask = np.zeros((n, width), bool)
        qt = np.array([i[2] for i in items], np.int64)
        for r, (tokens, markers, _, _) in enumerate(items):
            ids[r, :len(tokens)] = tokens; attn[r, :len(tokens)] = 1
            pos[r, :len(markers)] = markers; pmask[r, :len(markers)] = True
        logits, act = self.session.run(["logits", "act_logits"], {
            "input_ids": ids, "attention_mask": attn, "marker_pos": pos, "marker_mask": pmask, "qtype": qt})
        out = {}
        for r, qid in enumerate(qids):
            _, markers, qtype, keys = items[r]
            z = [float(x) / self.temperature[qtype] for x in logits[r, :len(markers)]]
            m = max(z); p = [math.exp(x - m) for x in z]; s = sum(p); p = [x / s for x in p]
            best = max(range(len(p)), key=p.__getitem__)
            ans = {"type": questions[qid]["type"], "probabilities": {k: round(p[j], 4) for j, k in enumerate(keys)},
                   "confidence": round(p[best], 4), "answer_confidence": round(p[best], 4)}
            if qtype == 0:
                ans["choice"] = keys[best]
            elif qtype == 1:
                ans["score"] = round(sum(j * pj for j, pj in enumerate(p)), 4)
            else:
                ans["noul"] = round(p[1], 4)
                ans["confidence"] = round(max(p[1], 1 - p[1]), 4)
            a0, a1 = float(act[r][0]), float(act[r][1]); am = max(a0, a1)
            e0, e1 = math.exp(a0 - am), math.exp(a1 - am)
            ans["action"] = {"act_probability": round(e0 / (e0 + e1), 4)}
            out[qid] = ans
        return {"answers": out}


if __name__ == "__main__":
    jev = OpenJevX()
    print(json.dumps(jev.decide({"age": 17}, {
        "adult": {"type": "noul", "instructions": "Is this person an adult (18 or older)?"},
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "payments and charges", "engineering": "bugs and outages"}},
        "sev": {"type": "score", "instructions": "How severe is this?", "criteria": ["low", "medium", "high"]},
    }), indent=2))

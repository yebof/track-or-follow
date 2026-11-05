#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E7 worker: FinBERT tone of news texts, local CPU, $0, no outcome.

Runs in the isolated local-model environment (.venv-ng: torch + transformers, no pandas), called by
experiments/exploratory/e7_reference_readers.py:

    .venv-ng/bin/python -I experiments/exploratory/e7_finbert_worker.py IN.jsonl OUT.csv [--threads 4]

IN.jsonl: one {"eid": ..., "text": ...} per line (headline + " " + summary, raw). OUT.csv: eid, p_neutral, p_positive,
p_negative (no text). Resumable: eids already in OUT.csv are skipped. The checkpoint is pinned (MODEL, REVISION); its
model card (literature/meta/huang2023finbert.hf_model_card.md) names the paper to cite (huang2023finbert) and the label
order (LABEL_0 neutral, LABEL_1 positive, LABEL_2 negative), which the script checks against the config.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

MODEL = "yiyanghkust/finbert-tone"
REVISION = "4921590d3c0c3832c0efea24c8381ce0bda7844b"
MAX_LEN, BATCH = 128, 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    torch.set_num_threads(a.threads)
    tok = BertTokenizer.from_pretrained(MODEL, revision=REVISION)
    mdl = BertForSequenceClassification.from_pretrained(MODEL, revision=REVISION, num_labels=3).eval()
    labels = {int(k): v for k, v in (mdl.config.id2label or {}).items()}
    order = ("neutral", "positive", "negative")  # the model card's LABEL_0..2; the config must say the same
    if [labels.get(i, "").lower() for i in range(3)] != list(order):
        raise SystemExit(f"unexpected label order in the config: {labels}")
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            done = {r["eid"] for r in csv.DictReader(fh)}
    todo = []
    with open(a.inp) as fh:
        for line in fh:
            r = json.loads(line)
            if r["eid"] not in done:
                todo.append(r)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["eid", "p_neutral", "p_positive", "p_negative"])
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            enc = tok([r["text"] for r in chunk], padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt")
            with torch.no_grad():
                p = torch.softmax(mdl(**enc).logits, dim=-1).numpy()
            for r, q in zip(chunk, p):
                w.writerow([r["eid"], *(f"{float(x):.6f}" for x in q)])
            fh.flush()
            if (i // BATCH) % 50 == 0:
                print(f"  finbert: {i + len(chunk)}/{len(todo)}", flush=True)
    print(json.dumps({"model": MODEL, "revision": REVISION, "config_id2label": labels, "order_used": order,
                      "scored": len(todo), "skipped_done": len(done)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""B10: honesty of the compliance claim after an exact check is placed in the loop.

Reads only saved outputs.  For the default (base) cell and for the first and last
rounds of the retry and FACTER-style memory cells, reports the violation rate,
the rate of ``claim = true``, the same rate among violating outputs, and the
AUROC of the claim against compliance.  No model inference is performed.

Conventions (same as analyze.py):
  * a missing claim counts as ``true``;
  * ``violated`` is undefined for an invalid output, so claim metrics use valid
    outputs only;
  * for the violation rate an invalid output counts as a violation (it cannot
    be certified compliant), matching the feedback macros in make_assets.py;
  * AUROC is NaN unless both classes have at least MIN_CLASS slates.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"]
MIN_CLASS = 10
OUT = os.path.join(ROOT, "results", "feedback_claims.csv")


def claim_value(x):
    """analyze.py convention: a missing claim is treated as a claim of compliance."""
    return 1.0 if x in (True, None) else 0.0


def auroc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    if len(y) == 0 or min(y.sum(), len(y) - y.sum()) < MIN_CLASS:
        return np.nan
    return float(roc_auc_score(y, s))


def metrics(rows):
    """rows: list of dicts with valid, violated and claim."""
    n = len(rows)
    n_invalid = sum(not r["valid"] for r in rows)
    valid = [r for r in rows if r["valid"]]
    n_valid = len(valid)
    if n_valid == 0:
        return dict(n=n, n_valid=0, n_invalid=n_invalid, n_violated=0, n_satisfied=0,
                    violation=np.nan, claim_true=np.nan, claim_true_given_violated=np.nan,
                    n_claim_false=0, auroc_claim=np.nan, notes="no valid output")
    violated = np.array([1.0 if r["violated"] else 0.0 for r in valid])
    claim = np.array([claim_value(r.get("claim")) for r in valid])
    given = claim[violated == 1]
    n_satisfied = int((violated == 0).sum())
    notes = []
    if claim.min() == claim.max():
        notes.append(f"claim constant {'true' if claim[0] else 'false'}")
    if min(violated.sum(), n_satisfied) < MIN_CLASS:
        notes.append(f"AUROC untestable: {n_satisfied} compliant < {MIN_CLASS}")
    if n_invalid:
        notes.append(f"{n_invalid} invalid counted as violation")
    return dict(
        n=n,
        n_valid=n_valid,
        n_invalid=n_invalid,
        n_violated=int(violated.sum()),
        n_satisfied=n_satisfied,
        # invalid outputs cannot be certified compliant
        violation=float((violated.sum() + n_invalid) / n),
        claim_true=float(claim.mean()),
        claim_true_given_violated=float(given.mean()) if len(given) else np.nan,
        n_claim_false=int((claim == 0).sum()),
        auroc_claim=auroc(1 - violated, claim),
        notes="; ".join(notes),
    )


def main():
    checked = pd.read_json(os.path.join(ROOT, "results", "checked.jsonl"), lines=True)
    rows = []
    for model in MODELS:
        base = checked[(checked.model == model) & (checked.dataset == "ml1m")
                       & (checked.cell == "base")]
        rows.append(dict(model=model, dataset="ml1m", cell="base", **metrics([
            dict(valid=bool(r.valid), violated=bool(r.violated), claim=r.claim)
            for r in base.itertuples()
        ])))

        pattern = os.path.join(ROOT, "results", "raw", model.replace(":", "_"),
                               "ml1m", "feedback_*.jsonl")
        for path in sorted(glob.glob(pattern)):
            recs = [json.loads(line) for line in open(path) if line.strip()]
            if not recs:
                continue
            mode = recs[0]["cell"].split("_")[1]
            for label, pick in (("first", lambda r: r["rounds"][0]),
                                ("final", lambda r: r["rounds"][-1])):
                rows.append(dict(model=model, dataset="ml1m", cell=f"feedback_{mode}_{label}",
                                 **metrics([pick(r) for r in recs])))

    out = pd.DataFrame(rows, columns=[
        "model", "dataset", "cell", "n", "n_valid", "n_invalid", "n_violated",
        "n_satisfied", "violation", "claim_true", "claim_true_given_violated",
        "n_claim_false", "auroc_claim", "notes",
    ])
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

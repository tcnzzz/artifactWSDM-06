"""B9: split-half robustness and the Qwen3 partial duplicate run.

1. Split the 400 sampled users per dataset into the first and second 200 in the
   frozen slate order, and recompute the default-cell claim rate, violation,
   McNemar test and token AUROC for the three small models.  Core cells use the
   first 300 (ML-1M) or 200 (MIND) sampled users, so the second half is partly
   or wholly absent by design; that is reported, not imputed.
2. Compare Qwen3-8B's base cell with `ablation_duplicate_client_base`.  DATA.md
   records the latter as 54 replies from an accidental second client process on
   the same slates, with the first reply per slate used; this is therefore a
   partial duplicate, and we report violation, claim and per-slate agreement.
No model is run and no network call is made.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"]
DATASETS = ["ml1m", "mind"]
MIN_CLASS = 10


def ptrue(model, ds, cell):
    md = model.replace(":", "_")
    p = os.path.join(ROOT, "results", "ptrue", md, ds, cell + ".jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        x = json.loads(line)
        tot = x["p_true"] + x["p_false"]
        out[x["sid"]] = x["p_true"] / tot if tot > 0 else np.nan
    return out


def claim_value(r):
    # Same convention as analyze.py: a missing claim counts as "satisfied".
    return 0.0 if r.get("claim") is False else 1.0


def mcnemar(rows):
    nb = sum(1 for r in rows if r["binding"] and r["violated"])
    nf = sum(1 for r in rows if (not r["binding"]) and (not r["violated"]))
    p = binomtest(nb, nb + nf, 0.5).pvalue if nb + nf else np.nan
    return nb, nf, p


def token_auroc(rows, pt):
    pairs = [(1.0 if not r["violated"] else 0.0, pt[r["sid"]])
             for r in rows if r["sid"] in pt and not np.isnan(pt[r["sid"]])]
    if not pairs:
        return np.nan, 0, 0, 0
    y = np.array([a for a, _ in pairs])
    s = np.array([b for _, b in pairs])
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    if min(npos, nneg) < MIN_CLASS:
        return np.nan, len(y), npos, nneg
    return float(roc_auc_score(y, s)), len(y), npos, nneg


def split_halves(rows):
    out = []
    for ds in DATASETS:
        S = json.load(open(os.path.join(ROOT, "data", f"slates_{ds}.json")))
        halves = {"first": {s["sid"] for s in S[:200]},
                  "second": {s["sid"] for s in S[200:]}}
        for model in MODELS:
            d = [r for r in rows if r["model"] == model and r["dataset"] == ds
                 and r["cell"] == "base" and r.get("valid")]
            pt = ptrue(model, ds, "base")
            for half, sids in halves.items():
                dd = [r for r in d if r["sid"] in sids]
                nb, nf, p = mcnemar(dd)
                a, na, npos, nneg = token_auroc(dd, pt)
                claim = float(np.mean([claim_value(r) for r in dd])) if dd else np.nan
                viol = float(np.mean([1.0 if r["violated"] else 0.0 for r in dd])) if dd else np.nan
                bind = float(np.mean([1.0 if r["binding"] else 0.0 for r in dd])) if dd else np.nan
                # The paper's main conclusion has three parts: the claim is
                # constant, compliance is worse than the input order, and the
                # token probability is not a useful signal.  "Not useful" is
                # operationalised as AUROC <= 0.70, below the 0.8 reached by
                # the free input-order feature; this threshold is documented
                # here and in CHANGELOG_REVISION.md (B9).
                claim_ok = (claim >= 0.99) if dd else np.nan
                compliance_ok = (viol > bind and p < 0.05) if dd and not np.isnan(p) else np.nan
                token_ok = (a <= 0.70) if not np.isnan(a) else np.nan
                holds = (bool(claim_ok) and bool(compliance_ok) and bool(token_ok)) if (
                    not pd.isna(claim_ok) and not pd.isna(compliance_ok) and not pd.isna(token_ok)) else np.nan
                note = ""
                if not dd:
                    note = ("second half of the 400 sampled users was not run: "
                            "core MIND cells use the first 200")
                elif ds == "ml1m" and half == "second":
                    note = "second half of the 400 sampled users; 92 remain in the first-300 core slates after filtering"
                if dd and np.isnan(a):
                    note = (note + "; " if note else "") + \
                        "token AUROC not testable: fewer than 10 positives in this half"
                out.append({
                    "model": model, "dataset": ds, "half": half, "n": len(dd),
                    "claim_true": claim, "binding": bind, "violation": viol,
                    "n_broken": nb, "n_fixed": nf, "mcnemar_p": p,
                    "auroc_tok": a, "n_auroc": na, "n_pos": npos, "n_neg": nneg,
                    "claim_constant": claim_ok, "compliance_worse": compliance_ok,
                    "token_auroc_low": token_ok, "conclusion_holds": holds,
                    "note": note,
                })
    return pd.DataFrame(out)


def duplicate_run(rows):
    base = {r["sid"]: r for r in rows if r["model"] == "qwen3:8b" and r["dataset"] == "ml1m"
            and r["cell"] == "base"}
    dup = {r["sid"]: r for r in rows if r["model"] == "qwen3:8b" and r["dataset"] == "ml1m"
           and r["cell"] == "ablation_duplicate_client_base"}
    common = sorted(set(base) & set(dup))
    both = [s for s in common if base[s].get("valid") and dup[s].get("valid")]
    def stats(D, sids):
        d = [D[s] for s in sids]
        return (float(np.mean([1.0 if r["violated"] else 0.0 for r in d])),
                float(np.mean([claim_value(r) for r in d])))
    bviol, bclaim = stats(base, both)
    dviol, dclaim = stats(dup, both)
    return pd.DataFrame([{
        "check": "partial_duplicate", "model": "qwen3:8b", "dataset": "ml1m",
        "cell": "ablation_duplicate_client_base", "n_common": len(common),
        "n_both_valid": len(both),
        "base_violation": bviol, "dup_violation": dviol,
        "base_claim_true": bclaim, "dup_claim_true": dclaim,
        "violation_agreement": float(np.mean([base[s]["violated"] == dup[s]["violated"] for s in both])),
        "claim_agreement": float(np.mean([claim_value(base[s]) == claim_value(dup[s]) for s in both])),
        "ranking_agreement": float(np.mean([base[s].get("rank_groups") == dup[s].get("rank_groups") for s in both])),
        "note": "DATA.md: 54 replies from an accidental second client process on the same slates; first reply per slate used",
    }])


def main():
    rows = [json.loads(line) for line in open(os.path.join(ROOT, "results", "checked.jsonl"))]
    H = split_halves(rows)
    D = duplicate_run(rows)
    H.to_csv(os.path.join(ROOT, "results", "split_halves.csv"), index=False)
    D.to_csv(os.path.join(ROOT, "results", "duplicate_run.csv"), index=False)
    pd.set_option("display.width", 250)
    print(H.round(4).to_string(index=False))
    print()
    print(D.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

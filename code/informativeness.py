"""How much does the claim tell about compliance, compared with features the operator gets for free?

For single-pass explicit-floor cells: compliance y of each valid output, the token probability of the
claim p, and free features (input order compliant, number of group switches in the output). AUROC for
y per cell, and pooled per model as a stratified AUROC (pairs only within a cell), with a bootstrap
over slates within cells. Also the exposure shortfall max(0, -margin) of output and input.
Writes results/informativeness.csv and results/shortfall.csv."""
import glob
import json
import os

import numpy as np
import pandas as pd

from fairlib import margin, satisfied

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = ["base", "rho070", "rho080", "rho095", "k05", "k20", "G3", "G5", "order_random", "order_reverse",
         "paraphrase", "minbudget", "reason", "history"]
DATASETS = ["ml1m", "mind"]
B = 1000
rng = np.random.default_rng(20261005)


def pairs_stats(y, s):
    """(concordant + ties/2, number of pairs) for compliant (y=1) above violating (y=0)."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 0
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values
    rpos = ranks[:len(pos)].sum()
    u = rpos - len(pos) * (len(pos) + 1) / 2
    return float(u), len(pos) * len(neg)


def load():
    rows = []
    for p in glob.glob(os.path.join(ROOT, "results", "raw", "*", "*", "*.jsonl")):
        ds, cell = os.path.basename(os.path.dirname(p)), os.path.basename(p)[:-len(".jsonl")]
        if ds not in DATASETS or cell not in CELLS:
            continue
        md = os.path.basename(os.path.dirname(os.path.dirname(p)))
        pt = {}
        pp = os.path.join(ROOT, "results", "ptrue", md, ds, cell + ".jsonl")
        if os.path.exists(pp):
            for line in open(pp):
                x = json.loads(line)
                tot = x["p_true"] + x["p_false"]
                pt[x["sid"]] = x["p_true"] / tot if tot > 0 else np.nan
        for line in open(p):
            r = json.loads(line)
            if not r.get("valid"):
                continue
            by = {c["cid"]: c["group"] for c in r["cands"]}
            g = [by[i] for i in r["ranking"]]
            bg = [c["group"] for c in sorted(r["cands"], key=lambda c: c["base_rank"])]
            fl = r["floors"]
            # infeasible slates (G=5) are excluded from AUROC by construction (no compliant output exists)
            rows.append({"model": r["model"], "dataset": ds, "cell": cell, "sid": r["sid"],
                         "y": float(satisfied(g, fl)), "p": pt.get(r["sid"], np.nan),
                         "claim": 0.0 if r.get("claim") is False else 1.0,
                         "input_ok": float(satisfied(bg, fl)),
                         "switches": float(sum(a != b for a, b in zip(g, g[1:]))),
                         "short_out": max(0.0, -margin(g, fl)), "short_in": max(0.0, -margin(bg, fl))})
    return pd.DataFrame(rows)


def strat_auroc(d, score):
    num = den = 0.0
    for _, c in d.groupby(["dataset", "cell"]):
        u, n = pairs_stats(c.y.values, c[score].values)
        num += u
        den += n
    return num / den if den else np.nan


def boot(d, score):
    est = strat_auroc(d, score)
    groups = [c for _, c in d.groupby(["dataset", "cell"])]
    vals = []
    for _ in range(B):
        bs = pd.concat([c.iloc[rng.integers(0, len(c), len(c))] for c in groups])
        vals.append(strat_auroc(bs, score))
    vals = [v for v in vals if not np.isnan(v)]
    return est, np.percentile(vals, 2.5), np.percentile(vals, 97.5), int(sum(
        min(c.y.sum(), len(c) - c.y.sum()) >= 1 for c in groups))


if __name__ == "__main__":
    D = load()
    D = D[D.p.notna()]
    out = []
    for model, d in D.groupby("model"):
        scopes = [("pooled", d)] + [(f"{ds}-base", d[(d.dataset == ds) & (d.cell == "base")]) for ds in DATASETS]
        for scope, dd in scopes:
            if dd.empty:
                continue
            for score in ("p", "input_ok", "switches"):
                est, lo, hi, ncell = boot(dd, score)
                out.append({"model": model, "scope": scope, "score": score, "auroc": est, "lo": lo, "hi": hi,
                            "n": len(dd), "n_compliant": int(dd.y.sum()), "cells": ncell})
    R = pd.DataFrame(out)
    R.to_csv(os.path.join(ROOT, "results", "informativeness.csv"), index=False)
    S = D[D.cell == "base"].groupby(["model", "dataset"]).agg(
        short_out=("short_out", "mean"), short_in=("short_in", "mean"), n=("y", "size")).reset_index()
    S.to_csv(os.path.join(ROOT, "results", "shortfall.csv"), index=False)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string())
    print(S.round(4).to_string())

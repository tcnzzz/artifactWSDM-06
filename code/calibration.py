"""B8: calibration of the token probability P(True) against output compliance.

Reads saved per-slate outputs (results/raw/**.jsonl) and the teacher-forced
token probabilities (results/ptrue/**), and writes results/calibration.csv for
the default cell.  For each model x dataset (and pooled per model) it reports
the Brier score and the 10-bin expected calibration error (ECE) of P(True)
against whether the output actually satisfies the floor.

The verbal confidence is not calibrated here: it takes only one to three
distinct values, so the paper reports its discrimination by AUROC instead.
No model is run and no network call is made.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from fairlib import satisfied

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["ml1m", "mind"]
CELL = "base"
BINS = 10


def ece(y, p, bins=BINS):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.sum() / len(p) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(total)


def load():
    rows = []
    for path in glob.glob(os.path.join(ROOT, "results", "raw", "*", "*", "*.jsonl")):
        ds = os.path.basename(os.path.dirname(path))
        cell = os.path.basename(path)[:-len(".jsonl")]
        if ds not in DATASETS or cell != CELL:
            continue
        md = os.path.basename(os.path.dirname(os.path.dirname(path)))
        pt = {}
        pp = os.path.join(ROOT, "results", "ptrue", md, ds, cell + ".jsonl")
        if os.path.exists(pp):
            for line in open(pp):
                x = json.loads(line)
                tot = x["p_true"] + x["p_false"]
                pt[x["sid"]] = x["p_true"] / tot if tot > 0 else np.nan
        for line in open(path):
            r = json.loads(line)
            if not r.get("valid"):
                continue
            by = {c["cid"]: c["group"] for c in r["cands"]}
            g = [by[i] for i in r["ranking"]]
            rows.append({
                "model": r["model"], "dataset": ds, "cell": cell, "sid": r["sid"],
                "y": float(satisfied(g, r["floors"])),
                "p": pt.get(r["sid"], np.nan),
                "confidence": r.get("confidence"),
            })
    return pd.DataFrame(rows)


def main():
    D = load()
    out = []
    for model, d in D.groupby("model"):
        scopes = [("pooled", d)] + [(ds, d[d.dataset == ds]) for ds in DATASETS]
        for scope, dd in scopes:
            if dd.empty:
                continue
            with_p = dd[dd.p.notna()]
            row = {
                "model": model, "dataset": scope, "cell": CELL, "n": len(dd),
                "n_with_p": len(with_p), "n_compliant": int(dd.y.sum()),
                "conf_distinct": int(pd.Series(dd.confidence).nunique()),
                "bins": BINS,
            }
            if len(with_p):
                y = with_p.y.values
                p = with_p.p.values
                row.update({
                    "brier": float(np.mean((p - y) ** 2)),
                    "ece": ece(y, p),
                    "mean_p": float(p.mean()),
                    "base_rate": float(y.mean()),
                    "note": "",
                })
            else:
                row.update({"brier": np.nan, "ece": np.nan, "mean_p": np.nan,
                            "base_rate": float(dd.y.mean()),
                            "note": "no token probabilities saved"})
            out.append(row)
    R = pd.DataFrame(out)
    R.to_csv(os.path.join(ROOT, "results", "calibration.csv"), index=False)
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

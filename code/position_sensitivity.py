#!/usr/bin/env python3
"""Post-hoc sensitivity of the violation verdict to the position-bias model.

The rerankers only ever saw the logarithmic weights w_j = 1/log2(j+1).  This
script re-scores their saved default-cell permutations under four alternative
weight vectors, keeping the same rho, the same three-decimal floors and the same
slate length k:

    log2       1 / log2(j+1)          (the paper's weights, baseline)
    inverse    1 / j
    linear     K - j + 1
    top5       1 for j <= 5, else 0   (exposure over the first five positions)
    geometric  0.8^(j-1)

Only the weights change; the floor is still rho * n_g / k with k = len(groups).
This is a post-hoc re-analysis of saved outputs: no model was re-run and the
models never saw these weights.

Usage:  python3 code/position_sensitivity.py [--out results/position_sensitivity.csv]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from fairlib import floors, margin, weights  # noqa: E402

WEIGHTS = {
    "log2": lambda k: weights(k),
    "inverse": lambda k: 1.0 / np.arange(1, k + 1),
    "linear": lambda k: (k - np.arange(1, k + 1) + 1.0),
    "top5": lambda k: np.array([1.0 if j <= 5 else 0.0 for j in range(1, k + 1)]),
    "geometric": lambda k: 0.8 ** (np.arange(1, k + 1) - 1.0),
}


def rate(xs):
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "position_sensitivity.csv"))
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(os.path.join(ROOT, "results", "checked.jsonl"))]
    rows = [r for r in rows if r.get("valid") and r.get("claim") is not None
            and r.get("rank_groups") and r.get("cell") == "base"]
    out = []
    for model in sorted({r["model"] for r in rows}):
        for ds in sorted({r["dataset"] for r in rows}):
            sub = [r for r in rows if r["model"] == model and r["dataset"] == ds]
            if not sub:
                continue
            base = None
            for name, fn in WEIGHTS.items():
                vio, cl = [], []
                for r in sub:
                    k = r["k"]
                    w = fn(k)
                    fl = floors(list(r["rank_groups"]), r["rho"])
                    vio.append(margin(list(r["rank_groups"]), fl, w=w) < -1e-9)
                    cl.append(bool(r["claim"]))
                v = 100 * rate(vio)
                if name == "log2":
                    base = v
                out.append({
                    "model": model,
                    "dataset": ds,
                    "weighting": name,
                    "n": len(sub),
                    "violation": round(v, 2),
                    "claim": round(100 * rate(cl), 2),
                    "delta_vs_log2": None if base is None else round(v - base, 2),
                })
    # pooled across models, within each dataset, for a one-line summary
    for ds in ("ml1m", "mind"):
        sub = [r for r in rows if r["dataset"] == ds]
        if not sub:
            continue
        base = None
        for name, fn in WEIGHTS.items():
            vio, cl = [], []
            for r in sub:
                k = r["k"]
                w = fn(k)
                fl = floors(list(r["rank_groups"]), r["rho"])
                vio.append(margin(list(r["rank_groups"]), fl, w=w) < -1e-9)
                cl.append(bool(r["claim"]))
            v = 100 * rate(vio)
            if name == "log2":
                base = v
            out.append({
                "model": "all",
                "dataset": ds,
                "weighting": name,
                "n": len(sub),
                "violation": round(v, 2),
                "claim": round(100 * rate(cl), 2),
                "delta_vs_log2": None if base is None else round(v - base, 2),
            })

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fields = ["model", "dataset", "weighting", "n", "violation", "claim", "delta_vs_log2"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"position_sensitivity.csv: {len(out)} rows")
    for r in out:
        if r["model"] == "all":
            print(f'  {r["dataset"]:5s} {r["weighting"]:9s} n={r["n"]:4d} '
                  f'viol={r["violation"]:5.1f}% claim={r["claim"]:5.1f}% '
                  f'delta={r["delta_vs_log2"]:+5.1f}')


if __name__ == "__main__":
    main()

"""Which way do rerankers move the group that is furthest below its floor?
For every valid output: exposure share of that group in the input order and in the output.
Writes results/direction.csv (one row per model x dataset x cell)."""
import glob
import json
import os

import numpy as np
import pandas as pd

from fairlib import shares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for p in sorted(glob.glob(os.path.join(ROOT, "results", "raw", "*", "*", "*.jsonl"))):
    cell = os.path.basename(p)[:-6]
    if cell.startswith(("feedback_", "fidelity_", "ablation_", "audit", "count")) or "/mind_nanscores/" in p:
        continue
    d_in, d_out = [], []
    for line in open(p):
        r = json.loads(line)
        if not r.get("valid"):
            continue
        by = {c["cid"]: c for c in r["cands"]}
        base = sorted(r["cands"], key=lambda c: c["base_rank"])
        g_in = [c["group"] for c in base]
        g_out = [by[c]["group"] for c in r["ranking"]]
        if r["prompt"] in ("count", "countm"):
            g = r["floors"]["count_group"]
        else:
            sh = shares(g_in)
            g = min(r["floors"], key=lambda x: sh.get(x, 0) - r["floors"][x])
        d_in.append(shares(g_in).get(g, 0.0))
        d_out.append(shares(g_out).get(g, 0.0))
    if not d_in:
        continue
    d_in, d_out = np.array(d_in), np.array(d_out)
    rows.append({"model": r["model"], "dataset": os.path.basename(os.path.dirname(p)), "cell": cell, "n": len(d_in),
                 "share_in": d_in.mean(), "share_out": d_out.mean(),
                 "moved_up": float((d_out > d_in + 1e-9).mean()), "moved_down": float((d_out < d_in - 1e-9).mean())})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "direction.csv"), index=False)
print(df[df.cell.isin(["base", "count", "befair"])].round(3).to_string())

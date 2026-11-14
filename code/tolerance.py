"""Verdicts with the weights as displayed in the prompt (three decimals) against exact weights.
Writes results/tolerance.json."""
import glob
import json
import os

import numpy as np

from fairlib import satisfied, weights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
n = same = near = 0
for p in glob.glob(os.path.join(ROOT, "results", "raw", "*", "*", "*.jsonl")):
    ds = os.path.basename(os.path.dirname(p))
    cell = os.path.basename(p)
    if ds not in ("ml1m", "mind") or cell.startswith(("feedback_", "fidelity_", "ablation_", "count")):
        continue
    for line in open(p):
        r = json.loads(line)
        if not r.get("valid") or not isinstance(r.get("floors"), dict) or "count_group" in r["floors"]:
            continue
        by = {c["cid"]: c["group"] for c in r["cands"]}
        g = [by[i] for i in r["ranking"]]
        w_exact = weights(len(g))
        w_disp = np.round(w_exact, 3)
        a, b = satisfied(g, r["floors"], w_exact), satisfied(g, r["floors"], w_disp)
        n += 1
        same += a == b
        tot = w_exact.sum()
        sh = {}
        for gg, wj in zip(g, w_exact):
            sh[gg] = sh.get(gg, 0) + wj / tot
        near += min(abs(sh.get(k, 0) - f) for k, f in r["floors"].items()) < 0.002
out = {"outputs": n, "same_verdict": int(same), "within_0.002_of_a_floor": int(near)}
json.dump(out, open(os.path.join(ROOT, "results", "tolerance.json"), "w"), indent=1)
print(out)

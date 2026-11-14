"""Reasoning-first cell: compare the exposure shares the model states in its 'work' field with the
true shares of the ranking it returned. Writes results/traces.csv (one row per model x dataset)."""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from fairlib import satisfied, shares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for p in sorted(glob.glob(os.path.join(ROOT, "results", "raw", "*", "ml1m", "reason.jsonl")) + glob.glob(os.path.join(ROOT, "results", "raw", "*", "mind", "reason.jsonl"))):
    stats = {"outputs": 0, "violated": 0, "violated_claimed": 0, "stated_all": 0, "stated_sum_gt1": 0,
             "stated_off": 0, "stated_says_ok_truth_not": 0}
    model = dataset = None
    for line in open(p):
        r = json.loads(line)
        model, dataset = r["model"], r["dataset"]
        if not r.get("valid"):
            continue
        stats["outputs"] += 1
        by = {c["cid"]: c["group"] for c in r["cands"]}
        g = [by[i] for i in r["ranking"]]
        true = shares(g)
        viol = not satisfied(g, r["floors"])
        stats["violated"] += viol
        if not (viol and r.get("claim") is True):
            continue
        stats["violated_claimed"] += 1
        w = r.get("work") or ""
        stated = {}
        for grp in r["floors"]:
            m = re.findall(rf"[Gg]roup {grp}\b[^\n]*?=\s*(0?\.\d+|1\.0+)\b", w)
            if m:
                stated[grp] = float(m[0])
        if len(stated) == len(r["floors"]):
            stats["stated_all"] += 1
            stats["stated_sum_gt1"] += sum(stated.values()) > 1.01
            stats["stated_off"] += any(abs(stated[x] - true.get(x, 0)) > 0.02 for x in stated)
            stats["stated_says_ok_truth_not"] += all(stated[x] >= f - 1e-9 for x, f in r["floors"].items())
    rows.append({"model": model, "dataset": dataset, **stats})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "traces.csv"), index=False)
print(df.to_string())

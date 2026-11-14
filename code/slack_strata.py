#!/usr/bin/env python3
"""Slate-level max-min slack strata for the exposure floor.

Reads results/checked.jsonl only; no model call, no network.  For every valid
output we compute

    t* = max over permutations sigma of min_g (E_g(sigma) - floor_g),

the largest amount by which some ranking can clear every three-decimal floor.
t* depends only on the multiset of candidate groups, k, rho and the position
weights, so it is objective-free: it is not the margin of the nDCG-maximising
repair.  We solve the integer assignment with HiGHS through fairlib.maxmin_slack.

Strata: infeasible (t* < 0), near (0 <= t* < delta), wide (t* >= delta).  The
default delta is 0.02; it separates the five-group slates that can only just
meet the floor (t* around 0.005-0.02) from the default two-group cell (t* around
0.046-0.05).  The old exact-repair margin is kept as an auxiliary column so the
difference between "the repair's clearance" and "the best possible clearance"
is visible; it is not used for the strata.

Usage:  python3 code/slack_strata.py [--delta 0.02] [--out results/slack_strata.csv]
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
from fairlib import floors, margin, maxmin_slack  # noqa: E402

SMALL = ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"]
STRONG = ["qwen2.5:14b", "command-r:35b", "llama3.3:70b", "qwen3:8b"]
ALL = SMALL + STRONG
# cells whose constraint is the exposure floor (count/countm use a prefix-count rule
# and carry no repair_groups, so they cannot be stratified this way)
EXPOSURE = {"base", "paraphrase", "minbudget", "history", "reason", "audit", "befair",
            "rho050", "rho070", "rho080", "rho095", "k05", "k20", "G3", "G5",
            "order_random", "order_reverse", "audit_rho080",
            "feedback_retry", "feedback_memory"}
STRATA = ("infeasible", "near", "wide")


def repair_margin(r):
    """Clearance of the nDCG-maximising exact repair (old auxiliary definition)."""
    rg = r.get("repair_groups")
    if not rg:
        return None
    return margin(list(rg), floors(list(rg), r["rho"]))


def rate(xs):
    return float(np.mean(xs)) if xs else float("nan")


def stratum_of(r, delta):
    # The exact-repair flag is the ground truth for feasibility; t* < 0 should agree
    # with it, and using the flag avoids a numerical tie at t* = 0.
    if not r["feasible"]:
        return "infeasible"
    return "near" if r["tstar"] < delta else "wide"


def stratum_rows(rows, scope, model, delta, datasets):
    out = []
    for ds in datasets:
        d = rows if ds == "pooled" else [r for r in rows if r["dataset"] == ds]
        groups = {s: [] for s in STRATA}
        for r in d:
            groups[stratum_of(r, delta)].append(r)
        for s in STRATA:
            a = groups[s]
            if not a:
                continue
            vio = [bool(r["violated"]) for r in a]
            cl = [bool(r["claim"]) for r in a]
            rm = [repair_margin(r) for r in a if repair_margin(r) is not None]
            ts = [r["tstar"] for r in a]
            out.append({
                "scope": scope,
                "model": model,
                "dataset": ds,
                "delta": delta,
                "stratum": s,
                "n": len(a),
                "violation": round(100 * rate(vio), 2),
                "claim": round(100 * rate(cl), 2),
                "claim_given_violated": round(100 * rate([c for c, v in zip(cl, vio) if v]), 2),
                "claim_given_satisfied": round(100 * rate([c for c, v in zip(cl, vio) if not v]), 2),
                "tstar_min": round(float(min(ts)), 4),
                "tstar_median": round(float(np.median(ts)), 4),
                "repair_margin_median": round(float(np.median(rm)), 4) if rm else None,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "slack_strata.csv"))
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(os.path.join(ROOT, "results", "checked.jsonl"))]
    rows = [r for r in rows if r.get("valid") and r.get("claim") is not None
            and r.get("rank_groups")]
    cache = {}
    for r in rows:
        key = (r["k"], tuple(sorted(r["rank_groups"])), round(float(r["rho"]), 3))
        if key not in cache:
            cache[key] = maxmin_slack(list(r["rank_groups"]), r["rho"])
        r["tstar"] = cache[key]
    bad = sum(1 for r in rows if (r["tstar"] < -1e-6) != (not r["feasible"]))
    print(f"t*/feasibility disagreements: {bad} of {len(rows)}")

    out = []
    for scope, cell_filter in (("exposure", lambda r: r["cell"] in EXPOSURE),
                               ("table1_base", lambda r: r["cell"] == "base")):
        scope_rows = [r for r in rows if cell_filter(r)]
        model_groups = [("all", scope_rows), ("small", [r for r in scope_rows if r["model"] in SMALL]),
                        ("strong", [r for r in scope_rows if r["model"] in STRONG])]
        model_groups += [(m, [r for r in scope_rows if r["model"] == m]) for m in ALL]
        for model, sub in model_groups:
            if not sub:
                continue
            out += stratum_rows(sub, scope, model, a.delta, ("pooled", "ml1m", "mind"))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fields = ["scope", "model", "dataset", "delta", "stratum", "n", "violation", "claim",
              "claim_given_violated", "claim_given_satisfied", "tstar_min", "tstar_median",
              "repair_margin_median"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"slack_strata.csv: {len(out)} rows, delta={a.delta}")
    for r in out:
        if r["scope"] == "exposure" and r["dataset"] == "pooled" and r["model"] in ("small", "strong", "all"):
            print(f'  {r["model"]:6s} {r["stratum"]:10s} n={r["n"]:5d} '
                  f'viol={r["violation"]:5.1f}% claim={r["claim"]:5.1f}% '
                  f't*_med={r["tstar_median"]:+.4f}')


if __name__ == "__main__":
    main()

"""A4/B7: nDCG-violation frontier from saved per-slate outputs (no model calls).

Reads results/checked.jsonl and results/raw/**.jsonl, keeps slates that are
feasible and have a held-out positive in every tightness cell of a model/dataset,
and writes results/frontier.csv plus figures/fig_frontier.pdf.

B7 adds three model-independent baselines on the same feasible slates:
  greedy       minimal greedy swaps from the input order until the floor is met
  fair_ir      FA*IR through the released fairsearchcore wrapper in fairlib.py
  round_robin  equal round-robin over groups, each group in relevance order

Every row carries a `baseline` column (empty for the input/output/projection/
exact-repair policies).  Utility is the base-score utility on all feasible
slates; nDCG is averaged only over slates with a held-out positive.  Timing is
wall-clock and therefore not part of the byte-identical reproduction check.
"""
import json
import math
import os
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fairlib import fair_rerank, margin, ndcg, satisfied, score_utility  # noqa: E402
from paperfig import BLUE, ORANGE, AQUA, GRAY, MODEL_NAME, apply_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"]
DS = ["ml1m", "mind"]
DS_NAME = {"ml1m": "ML-1M", "mind": "MIND"}
CELLS = [(0.50, "rho050"), (0.70, "rho070"), (0.80, "rho080"), (0.90, "base"), (0.95, "rho095")]
POLICIES = [
    ("input", "base_ndcg", "binding", GRAY, "o", "--", "input order"),
    ("output", "ndcg", "violated", BLUE, "s", "-", "reranker output"),
    ("projection", "proj_ndcg", None, ORANGE, "^", ":", "projection"),
    ("repair", "repair_ndcg", None, AQUA, "D", "-.", "exact repair"),
]
BASELINES = [
    ("greedy", "#7b3294", "P", "greedy swaps"),
    ("fair_ir", "#b8860b", "X", "FA*IR"),
    ("round_robin", "#8c564b", "v", "round-robin"),
]
B = 2000


def finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


# ------------------------------------------------------------------ baselines
def greedy_swap(groups, scores, floors):
    """Best-improvement swap from the input order; stop as soon as the floor is met."""
    k = len(groups)
    order = list(range(k))
    if satisfied([groups[i] for i in order], floors):
        return order
    for _ in range(k):
        cur = margin([groups[i] for i in order], floors)
        best = None
        for i in range(k):
            for j in range(i + 1, k):
                cand = order.copy()
                cand[i], cand[j] = cand[j], cand[i]
                key = (margin([groups[x] for x in cand], floors),
                       score_utility([scores[x] for x in cand]))
                if best is None or key > best[0]:
                    best = (key, cand)
        if best is None or best[0][0] <= cur + 1e-12:
            break
        order = best[1]
        if satisfied([groups[i] for i in order], floors):
            break
    return order


def round_robin(groups):
    """Equal round-robin over groups, taking each group's best remaining item."""
    k = len(groups)
    by = defaultdict(list)
    for i in range(k):
        by[groups[i]].append(i)
    seq = []
    while len(seq) < k:
        moved = False
        for g in sorted(by):
            if by[g]:
                seq.append(by[g].pop(0))
                moved = True
                if len(seq) == k:
                    break
        if not moved:
            break
    return seq


def fair_ir(groups, scores):
    """FA*IR with the minority group as protected, through fairlib.fair_rerank.

    Ties for the minority group are broken by group name, not by set iteration
    order: `min(set(groups), key=groups.count)` depends on Python's per-process
    string hash seed and made the baseline row differ between two runs.
    """
    counts = {g: groups.count(g) for g in sorted(set(groups))}
    minority = min(counts, key=lambda g: (counts[g], g))
    p = counts[minority] / len(groups)
    return fair_rerank(scores, groups, minority, p=p)


def baseline_order(name, groups, scores, floors):
    if name == "greedy":
        return greedy_swap(groups, scores, floors)
    if name == "fair_ir":
        return fair_ir(groups, scores)
    if name == "round_robin":
        return round_robin(groups)
    raise ValueError(name)


def load_raw(models, datasets, cells):
    """(model, dataset, cell, sid) -> raw record with cands and floors."""
    out = {}
    for m in models:
        md = m.replace(":", "_")
        for ds in datasets:
            for _, cell in cells:
                p = os.path.join(ROOT, "results", "raw", md, ds, cell + ".jsonl")
                if not os.path.exists(p):
                    continue
                for line in open(p):
                    r = json.loads(line)
                    out[(m, ds, cell, r["sid"])] = r
    return out


def boot_ci(values, rng):
    """Mean and 95% percentile bootstrap CI; values may contain NaN."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    est = float(v.mean())
    idx = rng.integers(0, len(v), (B, len(v)))
    boot = v[idx].mean(axis=1)
    return est, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    rows = [json.loads(line) for line in open(os.path.join(ROOT, "results", "checked.jsonl"))]
    raw = load_raw(MODELS, DS, CELLS)
    rng = np.random.default_rng(0)
    out = []
    panels = {}
    # Cache the three baseline outputs by (dataset, cell, sid): they depend only
    # on the slate, not on which model produced the output.
    base_cache = {}
    for ds in DS:
        for m in MODELS:
            by_rho = {}
            for rho, cell in CELLS:
                sub = [r for r in rows if r["model"] == m and r["dataset"] == ds
                       and r["cell"] == cell and r.get("feasible")]
                by_rho[rho] = {r["sid"]: r for r in sub
                               if all(finite(r.get(k)) for k in ("ndcg", "base_ndcg", "proj_ndcg", "repair_ndcg"))}
            sids = sorted(set.intersection(*(set(d) for d in by_rho.values())))
            if not sids:
                continue
            panels[(ds, m)] = {}
            for rho, cell in CELLS:
                d = by_rho[rho]
                arr = {name: np.array([d[s][key] for s in sids], dtype=float)
                       for name, key, _, _, _, _, _ in POLICIES}
                util = {
                    "input": np.ones(len(sids)),
                    "output": np.array([d[s]["util"] for s in sids], dtype=float),
                    "projection": np.array([d[s]["proj_util"] for s in sids], dtype=float),
                    "repair": np.array([d[s]["repair_util"] for s in sids], dtype=float),
                }
                vio = {
                    "input": np.array([bool(d[s]["binding"]) for s in sids], dtype=float),
                    "output": np.array([bool(d[s]["violated"]) for s in sids], dtype=float),
                    "projection": np.zeros(len(sids)),
                    "repair": np.zeros(len(sids)),
                }
                point = {"ds": ds, "model": m, "rho": rho, "n": len(sids)}
                for name, _, _, _, _, _, _ in POLICIES:
                    nd = arr[name]
                    vv = vio[name]
                    uu = util[name]
                    boot_nd = np.empty(B)
                    boot_v = np.empty(B)
                    boot_u = np.empty(B)
                    for b in range(B):
                        idx = rng.integers(0, len(sids), len(sids))
                        boot_nd[b] = nd[idx].mean()
                        boot_v[b] = vv[idx].mean()
                        boot_u[b] = uu[idx].mean()
                    point[name] = {
                        "ndcg": float(nd.mean()),
                        "ndcg_lo": float(np.percentile(boot_nd, 2.5)),
                        "ndcg_hi": float(np.percentile(boot_nd, 97.5)),
                        "violation": float(vv.mean()),
                        "violation_lo": float(np.percentile(boot_v, 2.5)),
                        "violation_hi": float(np.percentile(boot_v, 97.5)),
                        "util": float(uu.mean()),
                        "util_lo": float(np.percentile(boot_u, 2.5)),
                        "util_hi": float(np.percentile(boot_u, 97.5)),
                    }
                    out.append({
                        "model": m, "dataset": ds, "rho": rho, "policy": name, "baseline": "",
                        "n_slates": len(sids), "n_labeled": len(sids),
                        "time_ms_median": (float(np.median([d[s]["repair_ms"] for s in sids]))
                                           if name == "repair" else np.nan),
                        **point[name],
                    })
                panels[(ds, m)][rho] = point

                # ---- B7 baselines on all feasible slates, not only the common labeled ones
                feasible = [r for r in rows if r["model"] == m and r["dataset"] == ds
                            and r["cell"] == cell and r.get("feasible")]
                for bname, _color, _marker, _label in BASELINES:
                    vals, times = [], []
                    for r in feasible:
                        key = (ds, cell, r["sid"])
                        if key not in base_cache:
                            rr = raw[(m, ds, cell, r["sid"])]
                            base = sorted(rr["cands"], key=lambda c: c["base_rank"])
                            bg = [c["group"] for c in base]
                            bs = [c["score"] for c in base]
                            labels = [c["label"] for c in base]
                            res = {}
                            for nm in ("greedy", "fair_ir", "round_robin"):
                                t0 = time.perf_counter()
                                order = baseline_order(nm, bg, bs, rr["floors"])
                                dt = 1000 * (time.perf_counter() - t0)
                                g = [bg[i] for i in order]
                                res[nm] = {
                                    "violated": not satisfied(g, rr["floors"]),
                                    "util": score_utility([bs[i] for i in order]),
                                    "ndcg": ndcg([labels[i] for i in order]),
                                    "ms": dt,
                                }
                            base_cache[key] = res
                        vals.append(base_cache[key][bname])
                        times.append(base_cache[key][bname]["ms"])
                    vv = np.array([v["violated"] for v in vals], dtype=float)
                    uu = np.array([v["util"] for v in vals], dtype=float)
                    nd = np.array([v["ndcg"] for v in vals], dtype=float)
                    v_est, v_lo, v_hi = boot_ci(vv, rng)
                    u_est, u_lo, u_hi = boot_ci(uu, rng)
                    n_est, n_lo, n_hi = boot_ci(nd, rng)
                    out.append({
                        "model": m, "dataset": ds, "rho": rho, "policy": "baseline",
                        "baseline": bname, "n_slates": len(vals),
                        "n_labeled": int((~np.isnan(nd)).sum()),
                        "time_ms_median": float(np.median(times)),
                        "ndcg": n_est, "ndcg_lo": n_lo, "ndcg_hi": n_hi,
                        "violation": v_est, "violation_lo": v_lo, "violation_hi": v_hi,
                        "util": u_est, "util_lo": u_lo, "util_hi": u_hi,
                    })
                    panels[(ds, m)][rho][bname] = {
                        "ndcg": n_est, "ndcg_lo": n_lo, "ndcg_hi": n_hi,
                        "violation": v_est, "violation_lo": v_lo, "violation_hi": v_hi,
                    }

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    pd.DataFrame(out).to_csv(os.path.join(ROOT, "results", "frontier.csv"), index=False)

    apply_style()
    fig, axes = plt.subplots(len(DS), len(MODELS), figsize=(7.2, 3.8), sharex=True, sharey="row")
    for i, ds in enumerate(DS):
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            p = panels.get((ds, m))
            if not p:
                ax.set_visible(False)
                continue
            for name, _, _, color, marker, ls, label in POLICIES:
                xs = [100 * p[rho][name]["violation"] for rho, _ in CELLS]
                ys = [p[rho][name]["ndcg"] for rho, _ in CELLS]
                lo = [ys[k] - p[rho][name]["ndcg_lo"] for k, (rho, _) in enumerate(CELLS)]
                hi = [p[rho][name]["ndcg_hi"] - ys[k] for k, (rho, _) in enumerate(CELLS)]
                ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, ls=ls,
                            capsize=1.2, elinewidth=0.6, label=label)
            for name, color, marker, label in BASELINES:
                xs = [100 * p[rho][name]["violation"] for rho, _ in CELLS]
                ys = [p[rho][name]["ndcg"] for rho, _ in CELLS]
                ax.plot(xs, ys, color=color, marker=marker, ls=":", lw=0.9,
                        markersize=3.0, label=label)
            ax.set_title(f"{MODEL_NAME[m]}, {DS_NAME[ds]}")
            ax.grid(True, axis="y")
            if i == len(DS) - 1:
                ax.set_xlabel("violation of the floor (%)")
            if j == 0:
                ax.set_ylabel("nDCG@10")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1))
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "fig_frontier.pdf"))
    plt.close(fig)
    print(f"frontier: {len(out)} points, {len(panels)} panels")


if __name__ == "__main__":
    main()

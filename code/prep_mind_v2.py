"""MIND-small slates with a non-degenerate relevance score.

The first version (prep_data.build_mind, kept as the 'mind_tied' ablation) displayed the
min-max-scaled item-kNN score, which is tied on most candidates (1.9 distinct values per ten).
Here the score blends the kNN score with log popularity, s = knn / max(knn) + beta * log1p(clicks) / max,
with beta chosen on a validation day (fit on 9-12 Nov, evaluate nDCG@10 on 13 Nov impressions),
then refit on 9-13 Nov and applied to the same 14 Nov impressions as before.
Writes data/slates_mind.json (the old file is kept as data/slates_mind_tied.json) and adds
'mind_v2' to results/recommender_quality.json.
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from implicit.nearest_neighbours import CosineRecommender  # noqa: E402

from prep_data import N_SLATES, RAW, ROOT, SEED, SLATE_LEN, ndcg_at  # noqa: E402

BETAS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def fit(b, test_day):
    prev, last = b[b.day < test_day], b[b.day == test_day]
    pairs = set()
    for u, h in prev[["u", "history"]].dropna().drop_duplicates().itertuples(index=False):
        pairs.update((u, n) for n in h.split())
    for u, imp in prev[["u", "imp"]].itertuples(index=False):
        pairs.update((u, x[:-2]) for x in imp.split() if x.endswith("-1"))
    for u, h in last[["u", "history"]].dropna().drop_duplicates().itertuples(index=False):
        pairs.update((u, n) for n in h.split())       # pre-impression history of test-day users
    us, ns = zip(*pairs)
    uids, nids = sorted(set(us)), sorted(set(ns))
    umap = {u: i for i, u in enumerate(uids)}
    nmap = {n: i for i, n in enumerate(nids)}
    X = sp.csr_matrix((np.ones(len(us), np.float32), ([umap[u] for u in us], [nmap[n] for n in ns])),
                      shape=(len(uids), len(nids)))
    model = CosineRecommender(K=100, num_threads=2)
    model.fit(X, show_progress=False)
    return last, X, umap, nmap, nids, np.asarray(X.sum(0)).ravel(), model


def impressions(last, X, umap, nmap, clicks, model, rng, limit):
    cand = last[last.u.isin(umap)].drop_duplicates("u")
    cand = cand.iloc[rng.permutation(len(cand))]
    for row in cand.itertuples(index=False):
        c = [(x[:-2], int(x[-1])) for x in row.imp.split()]
        c = [(n, lab) for n, lab in c if n in nmap and clicks[nmap[n]] >= 5]
        if len(c) < SLATE_LEN:
            continue
        ui = umap[row.u]
        idx = np.array([nmap[n] for n, _ in c])
        ids, sc = model.recommend(ui, X[ui], N=len(idx), items=idx, filter_already_liked_items=False)
        lab = {nmap[n]: v for n, v in c}
        yield row, ids, np.asarray(sc, float), lab
        limit -= 1
        if limit == 0:
            return


def blend(ids, sc, clicks, beta):
    # implicit returns -DBL_MAX for candidates outside every neighbourhood of the user's items;
    # their kNN similarity is zero (an earlier version propagated -inf and produced NaN scores)
    sc = np.where(sc < -1e300, 0.0, sc)
    pop = np.log1p(clicks[ids])
    return sc / (sc.max() + 1e-9) + beta * pop / (pop.max() + 1e-9)


if __name__ == "__main__":
    b = pd.read_csv(os.path.join(RAW, "mind-small", "train", "behaviors.tsv"), sep="\t",
                    names=["iid", "u", "time", "history", "imp"])
    b["day"] = pd.to_datetime(b.time, format="%m/%d/%Y %I:%M:%S %p").dt.day
    news = pd.read_csv(os.path.join(RAW, "mind-small", "train", "news.tsv"), sep="\t",
                       names=["nid", "category", "sub", "title", "abs", "url", "te", "ae"]).set_index("nid")
    # validation: choose beta on 13 Nov
    last, X, umap, nmap, nids, clicks, model = fit(b, 13)
    val = {beta: [] for beta in BETAS}
    for _, ids, sc, lab in impressions(last, X, umap, nmap, clicks, model, np.random.default_rng(SEED), 3000):
        for beta in BETAS:
            o = ids[np.argsort(-blend(ids, sc, clicks, beta), kind="stable")]
            val[beta].append(ndcg_at([lab[i] for i in o], 10))
    val = {beta: float(np.mean(v)) for beta, v in val.items()}
    best = max(BETAS, key=lambda x: val[x])
    print("validation nDCG@10", val, "beta", best)
    # test day: same impressions as the first version
    last, X, umap, nmap, nids, clicks, model = fit(b, 14)
    rng = np.random.default_rng(SEED)
    slates, nd, nd0, distinct = [], [], [], []
    for row, ids, sc, lab in impressions(last, X, umap, nmap, clicks, model, rng, 3000):
        s = blend(ids, sc, clicks, best)
        o = np.argsort(-s, kind="stable")
        ids_o, s_o = ids[o], s[o]
        nd.append(ndcg_at([lab[i] for i in ids_o], 10))
        nd0.append(ndcg_at([lab[i] for i in ids[np.lexsort((-clicks[ids], -sc))]], 10))
        assert np.isfinite(s).all()
        rng.permutation(ids)                       # keep the random stream aligned with the first version
        if len(slates) < N_SLATES:
            top, sv = ids_o[:SLATE_LEN], s_o[:SLATE_LEN]
            lo, hi = float(sv.min()), float(sv.max())
            items = [{"item": nids[i], "title": str(news.loc[nids[i]].title)[:90],
                      "category": news.loc[nids[i]].category,
                      "score": round((float(v) - lo) / (hi - lo + 1e-12) * 0.9 + 0.1, 3),
                      "label": lab[i]} for i, v in zip(top, sv)]
            slates.append({"sid": f"mind-{row.iid}", "items": items})
            distinct.append(len({round(it["score"], 2) for it in items[:10]}))
            assert all(np.isfinite(it["score"]) for it in items)
    old = os.path.join(ROOT, "data", "slates_mind.json")
    tied = os.path.join(ROOT, "data", "slates_mind_tied.json")
    if not os.path.exists(tied):
        os.rename(old, tied)
    nanv = os.path.join(ROOT, "data", "slates_mind_nanscores.json")
    if os.path.exists(old) and not os.path.exists(nanv):
        os.rename(old, nanv)
    json.dump(slates, open(old, "w"))
    qpath = os.path.join(ROOT, "results", "recommender_quality.json")
    q = json.load(open(qpath))
    q["mind_v2"] = {"beta_validation": val, "beta": best, "eval_impressions": len(nd),
                    "ndcg10_blend": float(np.mean(nd)), "ndcg10_itemknn_tiebreak": float(np.mean(nd0)),
                    "distinct_scores_top10_mean": float(np.mean(distinct)),
                    "fully_tied_top10": float(np.mean(np.array(distinct) == 1))}
    json.dump(q, open(qpath, "w"), indent=1)
    print(q["mind_v2"])
    old_s = json.load(open(tied))
    print("same impressions:", [s["sid"] for s in old_s] == [s["sid"] for s in slates])

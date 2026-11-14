"""Build frozen candidate slates from real data with classical recommenders.

MovieLens-1M : item-kNN (implicit CosineRecommender; chosen over BPR on validation) trained on each user's chronologically first 80%
               of positive ratings (>=4); slate = top-20 unseen items.
MIND-small   : item-kNN (implicit CosineRecommender) trained on clicks of
               9-13 Nov 2019; slates are real impressions of 14 Nov 2019,
               restricted to warm candidates and ranked by the kNN score.
Outputs data/slates_{ml1m,mind}.json and results/recommender_quality.json.
"""
import collections
import json
import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import implicit  # noqa: E402
from implicit.bpr import BayesianPersonalizedRanking  # noqa: E402
from implicit.nearest_neighbours import CosineRecommender  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
SEED = 20261005
N_SLATES = 400      # sampled per dataset; experiments use the first n of these
SLATE_LEN = 20
quality = {"implicit_version": implicit.__version__, "seed": SEED}


def ndcg_at(labels, k):
    labels = np.asarray(labels[:k], float)
    w = 1.0 / np.log2(np.arange(2, len(labels) + 2))
    ideal = np.sort(labels)[::-1]
    return float((labels * w).sum() / (ideal * w).sum()) if labels.sum() > 0 else 0.0


# ----------------------------------------------------------------- MovieLens
def build_ml1m():
    rng = np.random.default_rng(SEED)
    movies = pd.read_csv(os.path.join(RAW, "ml-1m", "movies.dat"), sep="::", engine="python",
                         names=["mid", "title", "genres"], encoding="latin-1")
    movies["year"] = movies.title.str.extract(r"\((\d{4})\)\s*$").astype(int)
    ratings = pd.read_csv(os.path.join(RAW, "ml-1m", "ratings.dat"), sep="::", engine="python",
                          names=["u", "mid", "r", "t"])
    pos = ratings[ratings.r >= 4].sort_values(["u", "t", "mid"])
    cnt = pos.groupby("u").size()
    pos = pos[pos.u.isin(cnt[cnt >= 10].index)]
    pos["rank"] = pos.groupby("u").cumcount()
    pos["n"] = pos.groupby("u").mid.transform("size")
    train = pos[pos["rank"] < (0.8 * pos.n).astype(int)]
    test = pos[pos["rank"] >= (0.8 * pos.n).astype(int)]
    uids = np.sort(pos.u.unique()); iids = np.sort(movies.mid.unique())
    umap = {u: i for i, u in enumerate(uids)}; imap = {m: i for i, m in enumerate(iids)}
    X = sp.csr_matrix((np.ones(len(train), np.float32),
                       (train.u.map(umap).values, train.mid.map(imap).values)),
                      shape=(len(uids), len(iids)))
    # item-kNN (cosine, K=20) won the validation comparison in select_recommender.py
    model = CosineRecommender(K=20, num_threads=2)
    model.fit(X, show_progress=False)
    ids, scores = model.recommend(np.arange(len(uids)), X, N=SLATE_LEN, filter_already_liked_items=True)
    test_sets = test.groupby("u").mid.apply(set).to_dict()
    # sanity: compare with popularity
    pop = np.asarray(X.sum(0)).ravel()
    hits_knn, hits_pop = [], []
    for ui, u in enumerate(uids):
        ts = {imap[m] for m in test_sets.get(u, ())}
        hits_knn.append(ndcg_at([int(i in ts) for i in ids[ui][:10]], 10))
        seen = set(X[ui].indices)
        top_pop = [i for i in np.argsort(-pop)[:400] if i not in seen][:10]
        hits_pop.append(ndcg_at([int(i in ts) for i in top_pop], 10))
    quality["ml1m"] = {"users": len(uids), "items": len(iids), "train_pos": len(train),
                       "test_pos": len(test), "ndcg10_itemknn": float(np.mean(hits_knn)),
                       "ndcg10_popularity": float(np.mean(hits_pop))}
    mrow = movies.set_index("mid")
    chosen = rng.permutation(len(uids))[:N_SLATES]
    slates = []
    for ui in chosen:
        u = uids[ui]; ts = test_sets.get(u, set())
        s = scores[ui]; lo, hi = float(s.min()), float(s.max())
        items = []
        for i, sc in zip(ids[ui], s):
            mid = int(iids[i]); row = mrow.loc[mid]
            items.append({"item": mid, "title": row.title, "year": int(row.year),
                          "score": round((float(sc) - lo) / (hi - lo + 1e-12) * 0.9 + 0.1, 3),
                          "label": int(mid in ts)})
        slates.append({"sid": f"ml1m-{int(u)}", "items": items})
    return slates


# ---------------------------------------------------------------------- MIND
def build_mind():
    rng = np.random.default_rng(SEED)
    b = pd.read_csv(os.path.join(RAW, "mind-small", "train", "behaviors.tsv"), sep="\t",
                    names=["iid", "u", "time", "history", "imp"])
    b["day"] = pd.to_datetime(b.time, format="%m/%d/%Y %I:%M:%S %p").dt.day
    news = pd.read_csv(os.path.join(RAW, "mind-small", "train", "news.tsv"), sep="\t",
                       names=["nid", "category", "sub", "title", "abs", "url", "te", "ae"]).set_index("nid")
    prev, last = b[b.day < 14], b[b.day == 14]
    pairs = set()
    for u, h in prev[["u", "history"]].dropna().drop_duplicates().itertuples(index=False):
        pairs.update((u, n) for n in h.split())
    for u, imp in prev[["u", "imp"]].itertuples(index=False):
        pairs.update((u, x[:-2]) for x in imp.split() if x.endswith("-1"))
    # users seen only on the last day still have a pre-impression history
    for u, h in last[["u", "history"]].dropna().drop_duplicates().itertuples(index=False):
        pairs.update((u, n) for n in h.split())
    us, ns = zip(*pairs)
    uids = sorted(set(us)); nids = sorted(set(ns))
    umap = {u: i for i, u in enumerate(uids)}; nmap = {n: i for i, n in enumerate(nids)}
    X = sp.csr_matrix((np.ones(len(us), np.float32), ([umap[u] for u in us], [nmap[n] for n in ns])),
                      shape=(len(uids), len(nids)))
    clicks = np.asarray(X.sum(0)).ravel()
    model = CosineRecommender(K=100, num_threads=2)
    model.fit(X, show_progress=False)
    cand = last[last.u.isin(umap)].drop_duplicates("u")
    cand = cand.iloc[rng.permutation(len(cand))]
    slates, nd_knn, nd_rand, nd_pop = [], [], [], []
    for row in cand.itertuples(index=False):
        c = [(x[:-2], int(x[-1])) for x in row.imp.split()]
        c = [(n, l) for n, l in c if n in nmap and clicks[nmap[n]] >= 5]
        if len(c) < SLATE_LEN:
            continue
        ui = umap[row.u]
        idx = np.array([nmap[n] for n, _ in c])
        lab = {nmap[n]: l for n, l in c}
        ids, sc = model.recommend(ui, X[ui], N=len(idx), items=idx, filter_already_liked_items=False)
        order = np.lexsort((-clicks[ids], -sc))  # kNN score, ties by popularity
        ids, sc = ids[order], sc[order]
        nd_knn.append(ndcg_at([lab[i] for i in ids], 10))
        nd_rand.append(ndcg_at([lab[i] for i in rng.permutation(idx)], 10))
        nd_pop.append(ndcg_at([lab[i] for i in idx[np.argsort(-clicks[idx])]], 10))
        if len(slates) < N_SLATES:
            top, s = ids[:SLATE_LEN], sc[:SLATE_LEN]
            lo, hi = float(s.min()), float(s.max())
            items = [{"item": nids[i], "title": str(news.loc[nids[i]].title)[:90],
                      "category": news.loc[nids[i]].category,
                      "score": round((float(v) - lo) / (hi - lo + 1e-12) * 0.9 + 0.1, 3),
                      "label": lab[i]} for i, v in zip(top, s)]
            slates.append({"sid": f"mind-{row.iid}", "items": items})
        if len(nd_knn) >= 3000:
            break
    quality["mind"] = {"users": len(uids), "items": len(nids), "train_clicks": len(us),
                       "eval_impressions": len(nd_knn), "ndcg10_itemknn": float(np.mean(nd_knn)),
                       "ndcg10_random": float(np.mean(nd_rand)), "ndcg10_popularity": float(np.mean(nd_pop))}
    return slates


if __name__ == "__main__":
    for name, fn in [("ml1m", build_ml1m), ("mind", build_mind)]:
        slates = fn()
        with open(os.path.join(ROOT, "data", f"slates_{name}.json"), "w") as f:
            json.dump(slates, f)
        print(name, len(slates), quality[name])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "recommender_quality.json"), "w") as f:
        json.dump(quality, f, indent=1)

"""User context for the 'history' prompt: the ten most recent items each slate's user engaged
with before the slate (ML-1M: last ten training positives; MIND: last ten clicked titles in the
impression's history field). Writes data/history_{ml1m,mind}.json keyed by slate id."""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
N = 10

# MovieLens-1M: same filtering and split as prep_data.py
movies = pd.read_csv(os.path.join(RAW, "ml-1m", "movies.dat"), sep="::", engine="python",
                     names=["mid", "title", "genres"], encoding="latin-1").set_index("mid")
ratings = pd.read_csv(os.path.join(RAW, "ml-1m", "ratings.dat"), sep="::", engine="python", names=["u", "mid", "r", "t"])
pos = ratings[ratings.r >= 4].sort_values(["u", "t", "mid"])
pos["rank"] = pos.groupby("u").cumcount()
pos["n"] = pos.groupby("u").mid.transform("size")
train = pos[pos["rank"] < (0.8 * pos.n).astype(int)]
last = train.groupby("u").tail(N)
hist = {u: [movies.loc[m].title for m in g.mid] for u, g in last.groupby("u")}
slates = json.load(open(os.path.join(ROOT, "data", "slates_ml1m.json")))
out = {s["sid"]: hist.get(int(s["sid"].split("-")[1]), []) for s in slates}
json.dump(out, open(os.path.join(ROOT, "data", "history_ml1m.json"), "w"))
print("ml1m", sum(len(v) > 0 for v in out.values()), "of", len(out))

# MIND: the impression row's own history field (clicks before the impression)
b = pd.read_csv(os.path.join(RAW, "mind-small", "train", "behaviors.tsv"), sep="\t",
                names=["iid", "u", "time", "history", "imp"]).set_index("iid")
news = pd.read_csv(os.path.join(RAW, "mind-small", "train", "news.tsv"), sep="\t",
                   names=["nid", "category", "sub", "title", "abs", "url", "te", "ae"]).set_index("nid")
slates = json.load(open(os.path.join(ROOT, "data", "slates_mind.json")))
out = {}
for s in slates:
    h = b.loc[int(s["sid"].split("-")[1])].history
    ids = [] if not isinstance(h, str) else h.split()[-N:]
    out[s["sid"]] = [str(news.loc[n].title)[:90] for n in ids if n in news.index]
json.dump(out, open(os.path.join(ROOT, "data", "history_mind.json"), "w"))
print("mind", sum(len(v) > 0 for v in out.values()), "of", len(out))

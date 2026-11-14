"""Model selection for the MovieLens-1M candidate generator on a validation split
(70% train / next 10% validation of each user's chronological positives).
Writes results/recommender_selection.json. The winner is hard-coded in prep_data.py."""
import os, numpy as np, pandas as pd, scipy.sparse as sp, itertools
os.environ["OPENBLAS_NUM_THREADS"]="1"
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender, BM25Recommender
r = pd.read_csv(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data/raw/ml-1m/ratings.dat', sep='::', engine='python', names=['u','mid','r','t'])
pos = r[r.r>=4].sort_values(['u','t','mid']); cnt=pos.groupby('u').size(); pos=pos[pos.u.isin(cnt[cnt>=10].index)]
pos['rank']=pos.groupby('u').cumcount(); pos['n']=pos.groupby('u').mid.transform('size')
tr = pos[pos['rank']<(0.7*pos.n).astype(int)]; va = pos[(pos['rank']>=(0.7*pos.n).astype(int))&(pos['rank']<(0.8*pos.n).astype(int))]
uids=np.sort(pos.u.unique()); iids=np.sort(r.mid.unique()); um={u:i for i,u in enumerate(uids)}; im={m:i for i,m in enumerate(iids)}
X=sp.csr_matrix((np.ones(len(tr),np.float32),(tr.u.map(um).values,tr.mid.map(im).values)),shape=(len(uids),len(iids)))
vs=va.groupby('u').mid.apply(lambda s:{im[m] for m in s}).to_dict()
w=1/np.log2(np.arange(2,12)); res={}
def ev(model):
    ids,_=model.recommend(np.arange(len(uids)),X,N=10,filter_already_liked_items=True)
    out=[]
    for ui,u in enumerate(uids):
        ts=vs.get(u,set()); 
        if not ts: continue
        l=np.array([i in ts for i in ids[ui]],float); out.append((l*w).sum()/w[:min(10,len(ts))].sum())
    return np.mean(out)
for K in [20,50,100,200]:
    m=CosineRecommender(K=K,num_threads=2); m.fit(X,show_progress=False); res[f'itemknn_cosine_K{K}']=ev(m)
    m=BM25Recommender(K=K,num_threads=2); m.fit(X,show_progress=False); res[f'itemknn_bm25_K{K}']=ev(m)
for f,lr,reg,it in [(64,0.05,0.01,200),(64,0.01,0.01,300),(128,0.02,0.005,300),(32,0.05,0.01,200),(64,0.1,0.001,200),(100,0.05,0.0001,300)]:
    m=BayesianPersonalizedRanking(factors=f,learning_rate=lr,regularization=reg,iterations=it,random_state=1,num_threads=2); m.fit(X,show_progress=False); res[f'bpr_f{f}_lr{lr}_reg{reg}_it{it}']=ev(m)
pop=np.asarray(X.sum(0)).ravel()
class P: 
    def recommend(self,u,X,N,filter_already_liked_items):
        out=[]
        for ui in u:
            seen=set(X[ui].indices); out.append([i for i in np.argsort(-pop)[:600] if i not in seen][:N])
        return np.array(out),None
res['popularity']=ev(P())
import json
json.dump({k:float(v) for k,v in res.items()},open(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/results/recommender_selection.json','w'),indent=1)
print(res)

"""Chess v3: maximize S_brier. Test model variants on 5-fold OOF -> Brier skill (S_brier) +
worst-group. Variants: hierarchical empirical-Bayes opening model (direct outcome-rate signal),
GBM, LR, and blends. The opening's historical rate is the strongest generalizable signal."""
import os, sys, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from chess_features import apply_moves
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr=pd.read_csv(os.path.join(HERE,'dataset','train.csv'))
RATE=['white_win_rate','draw_rate','black_win_rate']; N=len(tr); cnt=tr['cohort_game_count'].values.astype(float); Y=tr[RATE].values
prior=np.average(Y,0,weights=cnt)
first1=tr['move_prefix'].apply(lambda s:s.split()[0]).values
first2=tr['move_prefix'].apply(lambda s:' '.join(s.split()[:2])).values
plyb=np.where(tr['prefix_ply_count'].values<=10,'p10',np.where(tr['prefix_ply_count'].values<=12,'p12','p14+'))
toks=tr['move_prefix'].apply(lambda s:s.split())
OPS={k:toks.apply(lambda t:' '.join(t[:k])).values for k in [1,2,3,4,5]}
BK=['mat_diff','cap_diff','n_captures','n_checks','castle_diff','queens_on','queen_diff','pawn_diff','minor_diff','rook_diff','center_diff','dev_diff','mat_total']
BF=np.array([[apply_moves(s)[k] for k in BK] for s in tr['move_prefix']],np.float32)
BF=np.column_stack([BF, tr['prefix_ply_count'].values.astype(np.float32)])
BRIER_REF=np.mean(np.sum((prior[None]-Y)**2,1))
def brow(p,y): return np.sum((p-y)**2,1)
def sbrier(p): return 1-np.mean(brow(p,Y))/BRIER_REF
def worst(p):
    ws=[]
    for g in (first1,first2,plyb):
        for gg in np.unique(g):
            m=g==gg
            if m.sum()>=3: ws.append(1-np.mean(brow(p[m],Y[m]))/(np.mean(brow(np.tile(prior,(m.sum(),1)),Y[m]))+1e-9))
    return min(ws)

def hier_te(tri,ap,alphas=(30,25,20,15)):
    """empirical-Bayes backoff: first-5->4->3->2->prior, support-weighted."""
    def level(k):
        ss=defaultdict(lambda:np.zeros(3)); ws=defaultdict(float)
        for o,yy,ww in zip(OPS[k][tri],Y[tri],cnt[tri]): ss[o]+=yy*ww; ws[o]+=ww
        return ss,ws
    L={k:level(k) for k in [2,3,4,5]}
    out=np.zeros((len(ap),3),np.float32)
    for i in range(len(ap)):
        pred=prior.copy()
        for kk,a in zip([2,3,4,5],alphas):  # coarse->fine, each refines
            o=OPS[kk][ap][i]; ss,ws=L[kk]
            if o in ws: pred=(ss[o]+pred*a)/(ws[o]+a)
        out[i]=pred
    return out
def expand(X,idx):
    return np.vstack([X,X,X]),np.concatenate([np.zeros(len(idx)),np.ones(len(idx)),2*np.ones(len(idx))]),np.concatenate([Y[idx,0]*cnt[idx],Y[idx,1]*cnt[idx],Y[idx,2]*cnt[idx]])

kf=KFold(5,shuffle=True,random_state=42)
oof={}
oof['hierTE']=np.zeros((N,3)); oof['gbm']=np.zeros((N,3)); oof['lr']=np.zeros((N,3))
# features for gbm/lr: board + hier TE (as features)
for tri,vai in kf.split(np.arange(N)):
    te_tr=hier_te(tri,tri); te_va=hier_te(tri,vai)
    oof['hierTE'][vai]=te_va
    Xtr=np.concatenate([BF[tri],te_tr],1); Xva=np.concatenate([BF[vai],te_va],1)
    Xs,ys,ws=expand(Xtr,tri)
    gb=[HistGradientBoostingClassifier(random_state=s,max_leaf_nodes=8,min_samples_leaf=50,learning_rate=0.03,max_iter=350,l2_regularization=2.0).fit(Xs,ys,sample_weight=ws) for s in (0,1)]
    oof['gbm'][vai]=np.mean([g.predict_proba(Xva) for g in gb],0)
    sc=StandardScaler().fit(Xs); lr=LogisticRegression(C=0.1,max_iter=3000).fit(sc.transform(Xs),ys,sample_weight=ws)
    oof['lr'][vai]=lr.predict_proba(sc.transform(Xva))
print('=== S_brier / worst by model ===',flush=True)
for k,p in oof.items():
    print(f'  {k:8s} Sbrier={sbrier(p):.4f} worst={worst(p):.4f}',flush=True)
# blends
for wt,wg,wl in [(0.5,0.3,0.2),(0.6,0.25,0.15),(0.4,0.4,0.2),(0.7,0.2,0.1),(0.34,0.33,0.33)]:
    p=wt*oof['hierTE']+wg*oof['gbm']+wl*oof['lr']; p=p/p.sum(1,keepdims=True)
    print(f'  blend TE{wt}/gbm{wg}/lr{wl}: Sbrier={sbrier(p):.4f} worst={worst(p):.4f}',flush=True)

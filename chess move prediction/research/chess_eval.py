"""Honest chess evaluation: nested CV of the full composite metric (prior-based refs),
so calibration selection is not leaked. Lets me compare feature sets / models / calibration
against the real scoring rule instead of an optimistic in-fold BSS.
"""
import os, sys, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chess_features import apply_moves
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr=pd.read_csv(os.path.join(HERE,'dataset','train.csv'))
RATE=['white_win_rate','draw_rate','black_win_rate']
N=len(tr); cnt=tr['cohort_game_count'].values.astype(float); Y=tr[RATE].values
prior=np.average(Y,0,weights=cnt)
first2=tr['move_prefix'].apply(lambda s:' '.join(s.split()[:2])).values

BK=['mat_diff','cap_diff','n_captures','n_checks','castle_diff','w_castled','b_castled','queens_on',
    'queen_diff','pawn_diff','minor_diff','rook_diff','center_diff','dev_diff','dev_total','mat_total','n_promo','bishop_pair_w','bishop_pair_b']
BF=np.array([[apply_moves(s)[k] for k in BK] for s in tr['move_prefix']],np.float32)
BF=np.column_stack([BF, tr['prefix_ply_count'].values.astype(np.float32)])
OPKS=[2,3,4]  # only families that overlap test (first-6+ has 0% test overlap)
opv=lambda k: tr['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values
OP={k:opv(k) for k in OPKS}
def catd():
    out=[]
    for s in tr['move_prefix']:
        t=s.split(); d={'op2='+' '.join(t[:2]):1.,'op3='+' '.join(t[:3]):1.,'op4='+' '.join(t[:4]):1.}
        for i in range(min(4,len(t))): d[f'm{i}={t[i]}']=1.
        out.append(d)
    return out
dv=DictVectorizer(sparse=False); CAT=dv.fit_transform(catd()); keep=CAT.sum(0)>=10; CAT=CAT[:,keep]

def tenc(otr_idx, oap_vals, k, alpha):
    ss=defaultdict(lambda:np.zeros(3)); ws=defaultdict(float)
    for o,yy,ww in zip(OP[k][otr_idx],Y[otr_idx],cnt[otr_idx]): ss[o]+=yy*ww; ws[o]+=ww
    out=np.zeros((len(oap_vals),3),np.float32)
    for i,o in enumerate(oap_vals): out[i]=(ss[o]+prior*alpha)/(ws[o]+alpha) if o in ws else prior
    return out
def feats(tri, ap_idx, alpha=25.0):
    TE=np.concatenate([tenc(tri,OP[k][ap_idx],k,alpha) for k in OPKS],1)
    return np.concatenate([BF[ap_idx],CAT[ap_idx],TE],1)
def expand(X,idx):
    return (np.vstack([X,X,X]),np.concatenate([np.zeros(len(idx)),np.ones(len(idx)),2*np.ones(len(idx))]),
            np.concatenate([Y[idx,0]*cnt[idx],Y[idx,1]*cnt[idx],Y[idx,2]*cnt[idx]]))
GBM=dict(max_leaf_nodes=8,min_samples_leaf=60,learning_rate=0.03,max_iter=350,l2_regularization=2.0)
def fit_pred(tri, ap_idx):
    Xtr=feats(tri,tri); Xap=feats(tri,ap_idx); Xs,ys,ws=expand(Xtr,tri)
    sc=StandardScaler().fit(Xs)
    lr=LogisticRegression(C=0.08,max_iter=4000); lr.fit(sc.transform(Xs),ys,sample_weight=ws)
    gb=[HistGradientBoostingClassifier(random_state=s,**GBM).fit(Xs,ys,sample_weight=ws) for s in (0,1)]
    return lr.predict_proba(sc.transform(Xap)), np.mean([g.predict_proba(Xap) for g in gb],0)

# ---- metric ----
def brier_rows(p,y): return np.sum((p-y)**2,1)
def logexc_rows(p,y,eps=1e-12):
    p=np.clip(p,eps,1); return -np.sum(y*np.log(p),1) - (-np.sum(y*np.log(np.clip(y,eps,1)),1))
# prior-based refs (computed once on full train, as "prepared training split" constants)
BRIER_REF=np.average(brier_rows(np.tile(prior,(N,1)),Y),weights=None)
LOG_REF=np.mean(logexc_rows(np.tile(prior,(N,1)),Y))
CONF_REF=np.mean(np.abs(prior.max()-Y.max(1)))
def calib(p,lam,T):
    q=np.exp(np.log(np.clip(p,1e-9,1))/T); q/=q.sum(1,keepdims=True); q=(1-lam)*q+lam*prior[None]; return q/q.sum(1,keepdims=True)
def composite(p,conf,y,groups):
    sb=np.clip(1-np.mean(brier_rows(p,y))/BRIER_REF,0,1)
    sl=np.clip(1-np.mean(logexc_rows(p,y))/LOG_REF,0,1)
    scf=np.clip(1-np.mean(np.abs(conf-y.max(1)))/CONF_REF,0,1)
    # worst group brier skill
    wg=[]
    for g in np.unique(groups):
        m=groups==g
        if m.sum()>=3:
            bref=np.mean(brier_rows(np.tile(prior,(m.sum(),1)),y[m]))
            sk=1-np.mean(brier_rows(p[m],y[m]))/(bref+1e-9); wg.append(sk)
    sw=min(wg) if wg else 0.0
    comp=0.55*sb+0.20*sl+0.10*scf+0.15*sw
    return comp,sb,sl,scf,sw
def final(comp): return 0.12+0.88*comp

def nested_cv(lam_grid=(0.1,0.2,0.3,0.4), T_grid=(1.0,1.2), wg_grid=(0.6,0.75,0.9), conf_ab=None, seed=42):
    outer=KFold(5,shuffle=True,random_state=seed)
    OOF=np.zeros((N,3)); CONF=np.zeros(N)
    for tri,tei in outer.split(np.arange(N)):
        # inner OOF on tri to pick calibration
        inner=KFold(4,shuffle=True,random_state=seed+1)
        oof_lr=np.zeros((len(tri),3)); oof_g=np.zeros((len(tri),3))
        pos={idx:i for i,idx in enumerate(tri)}
        for itr,iva in inner.split(tri):
            a,b=tri[itr],tri[iva]
            plr,pg=fit_pred(a,b)
            for j,idx in enumerate(b): oof_lr[pos[idx]]=plr[j]; oof_g[pos[idx]]=pg[j]
        besti=(-9,None)
        for wg in wg_grid:
            bl=wg*oof_g+(1-wg)*oof_lr
            for lam in lam_grid:
                for T in T_grid:
                    c=calib(bl,lam,T); comp,_,_,_,_=composite(c, np.clip(0.5*c.max(1)+0.4,0,1), Y[tri], first2[tri])
                    if comp>besti[0]: besti=(comp,(wg,lam,T))
        wg,lam,T=besti[1]
        # fit conf map on inner oof
        cin=calib(wg*oof_g+(1-wg)*oof_lr,lam,T); mp=cin.max(1); my=Y[tri].max(1)
        A=np.vstack([mp,np.ones_like(mp)]).T; a,b=np.linalg.lstsq(A,my,rcond=None)[0]
        # predict outer test
        plr,pg=fit_pred(tri,tei); c=calib(wg*pg+(1-wg)*plr,lam,T)
        OOF[tei]=c; CONF[tei]=np.clip(a*c.max(1)+b,0,1)
    comp,sb,sl,scf,sw=composite(OOF,CONF,Y,first2)
    print(f'  HONEST nested-CV: composite={comp:.4f} final={final(comp):.4f} | Sbrier={sb:.3f} Slog={sl:.3f} Sconf={scf:.3f} Sworst={sw:.3f}',flush=True)
    return comp

if __name__=='__main__':
    print(f'refs: BRIER={BRIER_REF:.4f} LOG={LOG_REF:.4f} CONF={CONF_REF:.4f}')
    print('feat dims: board',BF.shape[1],'cat',CAT.shape[1])
    print('=== current-style pipeline, honest nested CV ===')
    nested_cv()

"""Chess v4: maximize the true composite. Blend hierarchical-TE + GBM + LR, then choose
shrinkage/temperature to trade S_brier against worst-group (the term that sank 0.173).
Reports every sub-score under coarse (realistic) and fine (pessimistic) groupings, and
builds the submission by refitting on all data."""
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
tr=pd.read_csv(os.path.join(HERE,'dataset','train.csv')); te=pd.read_csv(os.path.join(HERE,'dataset','test.csv'))
RATE=['white_win_rate','draw_rate','black_win_rate']; N=len(tr); cnt=tr['cohort_game_count'].values.astype(float); Y=tr[RATE].values
prior=np.average(Y,0,weights=cnt)
def grp1(df): return df['move_prefix'].apply(lambda s:s.split()[0]).values
def grp2(df): return df['move_prefix'].apply(lambda s:' '.join(s.split()[:2])).values
def grpp(df): v=df['prefix_ply_count'].values; return np.where(v<=10,'p10',np.where(v<=12,'p12','p14+'))
first1,first2,plyb=grp1(tr),grp2(tr),grpp(tr)
OPS_tr={k:tr['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values for k in [2,3,4,5]}
OPS_te={k:te['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values for k in [2,3,4,5]}
BK=['mat_diff','cap_diff','n_captures','n_checks','castle_diff','queens_on','queen_diff','pawn_diff','minor_diff','rook_diff','center_diff','dev_diff','mat_total']
def board(df):
    M=np.array([[apply_moves(s)[k] for k in BK] for s in df['move_prefix']],np.float32)
    return np.column_stack([M, df['prefix_ply_count'].values.astype(np.float32)])
BFtr,BFte=board(tr),board(te)
def catdicts(df):
    out=[]
    for s in df['move_prefix']:
        t=s.split(); d={'op2='+' '.join(t[:2]):1.,'op3='+' '.join(t[:3]):1.,'op4='+' '.join(t[:4]):1.}
        for i in range(min(4,len(t))): d[f'm{i}={t[i]}']=1.
        out.append(d)
    return out
dv=DictVectorizer(sparse=False); CATtr=dv.fit_transform(catdicts(tr)); CATte=dv.transform(catdicts(te)); keep=CATtr.sum(0)>=10; CATtr,CATte=CATtr[:,keep],CATte[:,keep]
def hier_te(tri, ops_ap, alphas=(30,25,20,15)):
    L={}
    for k in [2,3,4,5]:
        ss=defaultdict(lambda:np.zeros(3)); ws=defaultdict(float)
        for o,yy,ww in zip(OPS_tr[k][tri],Y[tri],cnt[tri]): ss[o]+=yy*ww; ws[o]+=ww
        L[k]=(ss,ws)
    n=len(ops_ap[2]); out=np.zeros((n,3),np.float32)
    for i in range(n):
        pred=prior.copy()
        for kk,a in zip([2,3,4,5],alphas):
            o=ops_ap[kk][i]; ss,ws=L[kk]
            if o in ws: pred=(ss[o]+pred*a)/(ws[o]+a)
        out[i]=pred
    return out
def expand(X,idx): return np.vstack([X,X,X]),np.concatenate([np.zeros(len(idx)),np.ones(len(idx)),2*np.ones(len(idx))]),np.concatenate([Y[idx,0]*cnt[idx],Y[idx,1]*cnt[idx],Y[idx,2]*cnt[idx]])
GBM=dict(max_leaf_nodes=8,min_samples_leaf=50,learning_rate=0.03,max_iter=350,l2_regularization=2.0)
def models(tri, te_tr, X_tr):
    Xs,ys,ws=expand(X_tr,tri); sc=StandardScaler().fit(Xs)
    lr=LogisticRegression(C=0.1,max_iter=3000).fit(sc.transform(Xs),ys,sample_weight=ws)
    gb=[HistGradientBoostingClassifier(random_state=s,**GBM).fit(Xs,ys,sample_weight=ws) for s in (0,1)]
    return sc,lr,gb
BRIER_REF=np.mean(np.sum((prior[None]-Y)**2,1)); LOG_REF=np.mean(-np.sum(Y*np.log(prior[None]),1)+np.sum(Y*np.log(np.clip(Y,1e-12,1)),1)); CONF_REF=np.mean(np.abs(prior.max()-Y.max(1)))
def brow(p,y): return np.sum((p-y)**2,1)
def wg(p,g):
    o=[]
    for gg in np.unique(g):
        m=g==gg
        if m.sum()>=3: o.append(1-np.mean(brow(p[m],Y[m]))/(np.mean(brow(np.tile(prior,(m.sum(),1)),Y[m]))+1e-9))
    return min(o) if o else 0.
def comp(p,conf,coarse=True):
    sb=np.clip(1-np.mean(brow(p,Y))/BRIER_REF,0,1)
    le=np.mean(-np.sum(Y*np.log(np.clip(p,1e-12,1)),1)+np.sum(Y*np.log(np.clip(Y,1e-12,1)),1)); sl=np.clip(1-le/LOG_REF,0,1)
    scf=np.clip(1-np.mean(np.abs(conf-Y.max(1)))/CONF_REF,0,1)
    sw=min(wg(p,first1),wg(p,plyb)) if coarse else min(wg(p,first1),wg(p,plyb),wg(p,first2))
    return 0.55*sb+0.20*sl+0.10*scf+0.15*sw, sb,sl,scf,sw
def calib(p,lam,T): q=np.exp(np.log(np.clip(p,1e-9,1))/T); q/=q.sum(1,keepdims=True); q=(1-lam)*q+lam*prior[None]; return q/q.sum(1,keepdims=True)

kf=KFold(5,shuffle=True,random_state=42); oof=np.zeros((N,3))
for tri,vai in kf.split(np.arange(N)):
    te_tr=hier_te(tri,{k:OPS_tr[k][tri] for k in [2,3,4,5]}); te_va=hier_te(tri,{k:OPS_tr[k][vai] for k in [2,3,4,5]})
    Xtr=np.concatenate([BFtr[tri],CATtr[tri],te_tr],1); Xva=np.concatenate([BFtr[vai],CATtr[vai],te_va],1)
    sc,lr,gb=models(tri,te_tr,Xtr)
    plr=lr.predict_proba(sc.transform(Xva)); pg=np.mean([g.predict_proba(Xva) for g in gb],0)
    oof[vai]=0.45*te_va+0.35*pg+0.20*plr
oof=oof/oof.sum(1,keepdims=True)
mp=oof.max(1); a,b=np.linalg.lstsq(np.vstack([mp,np.ones_like(mp)]).T,Y.max(1),rcond=None)[0]
best=(-9,None)
for lam in [0.1,0.2,0.3,0.4,0.5]:
    for T in [1.0,1.15,1.3,1.5]:
        c=calib(oof,lam,T); conf=np.clip(a*c.max(1)+b,0,1); cp,sb,sl,scf,sw=comp(c,conf,coarse=True)
        if cp>best[0]: best=(cp,(lam,T),(sb,sl,scf,sw))
cp,(lam,T),(sb,sl,scf,sw)=best
c=calib(oof,lam,T); conf=np.clip(a*c.max(1)+b,0,1); _,_,_,_,swf=comp(c,conf,coarse=False)
print(f'BEST lam={lam} T={T}: coarse comp={cp:.4f} final={0.12+0.88*cp:.4f}',flush=True)
print(f'  Sbrier={sb:.3f} Slog={sl:.3f} Sconf={scf:.3f} Sworst_coarse={sw:.3f} Sworst_fine={swf:.3f}',flush=True)

# ---- build submission: refit on all train, predict test ----
te_all=hier_te(np.arange(N),{k:OPS_tr[k] for k in [2,3,4,5]})
te_test=hier_te(np.arange(N),{k:OPS_te[k] for k in [2,3,4,5]})
Xall=np.concatenate([BFtr,CATtr,te_all],1); Xtest=np.concatenate([BFte,CATte,te_test],1)
sc,lr,gb=models(np.arange(N),te_all,Xall)
plr=lr.predict_proba(sc.transform(Xtest)); pg=np.mean([g.predict_proba(Xtest) for g in gb],0)
ptest=0.45*te_test+0.35*pg+0.20*plr; ptest=ptest/ptest.sum(1,keepdims=True)
ptest=calib(ptest,lam,T); conf_t=np.clip(a*ptest.max(1)+b,0,1)
sub=pd.DataFrame({'id':te['id'].values,'white_win_prob':ptest[:,0],'draw_prob':ptest[:,1],'black_win_prob':ptest[:,2],'confidence':conf_t})
os.makedirs(os.path.join(HERE,'working'),exist_ok=True); sub.to_csv(os.path.join(HERE,'working','submission.csv'),index=False)
print('wrote submission',sub.shape, 'lam',lam,'T',T,'a',round(a,3),'b',round(b,3),flush=True)

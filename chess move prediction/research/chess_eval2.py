"""Chess v2 diagnosis: fix the worst-group collapse. 5-fold OOF, then sweep calibration
(blend, shrinkage lam, temperature T) reporting the FULL composite + all 4 sub-scores +
per-group skill, so I can see the worst opening group and how shrinkage rescues it."""
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
first1=tr['move_prefix'].apply(lambda s:s.split()[0]).values
first2=tr['move_prefix'].apply(lambda s:' '.join(s.split()[:2])).values
plyb=np.where(tr['prefix_ply_count'].values<=10,'p10',np.where(tr['prefix_ply_count'].values<=12,'p12','p14+'))
BK=['mat_diff','cap_diff','n_captures','n_checks','castle_diff','w_castled','b_castled','queens_on',
    'queen_diff','pawn_diff','minor_diff','rook_diff','center_diff','dev_diff','dev_total','mat_total','n_promo','bishop_pair_w','bishop_pair_b']
BF=np.array([[apply_moves(s)[k] for k in BK] for s in tr['move_prefix']],np.float32)
BF=np.column_stack([BF, tr['prefix_ply_count'].values.astype(np.float32)])
OPKS=[2,3,4]; opv=lambda k: tr['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values
OP={k:opv(k) for k in OPKS}
def catd():
    out=[]
    for s in tr['move_prefix']:
        t=s.split(); d={'op2='+' '.join(t[:2]):1.,'op3='+' '.join(t[:3]):1.,'op4='+' '.join(t[:4]):1.}
        for i in range(min(4,len(t))): d[f'm{i}={t[i]}']=1.
        out.append(d)
    return out
dv=DictVectorizer(sparse=False); CAT=dv.fit_transform(catd()); keep=CAT.sum(0)>=10; CAT=CAT[:,keep]
def tenc(tri,ap,k,alpha):
    ss=defaultdict(lambda:np.zeros(3)); ws=defaultdict(float)
    for o,yy,ww in zip(OP[k][tri],Y[tri],cnt[tri]): ss[o]+=yy*ww; ws[o]+=ww
    out=np.zeros((len(ap),3),np.float32)
    for i,o in enumerate(ap): out[i]=(ss[o]+prior*alpha)/(ws[o]+alpha) if o in ws else prior
    return out
def feats(tri,ap):
    TE=np.concatenate([tenc(tri,OP[k][ap],k,25.) for k in OPKS],1)
    return np.concatenate([BF[ap],CAT[ap],TE],1)
def expand(X,idx):
    return np.vstack([X,X,X]),np.concatenate([np.zeros(len(idx)),np.ones(len(idx)),2*np.ones(len(idx))]),np.concatenate([Y[idx,0]*cnt[idx],Y[idx,1]*cnt[idx],Y[idx,2]*cnt[idx]])
GBM=dict(max_leaf_nodes=8,min_samples_leaf=60,learning_rate=0.03,max_iter=350,l2_regularization=2.0)
def fitp(tri,ap):
    Xtr=feats(tri,tri); Xap=feats(tri,ap); Xs,ys,ws=expand(Xtr,tri)
    sc=StandardScaler().fit(Xs); lr=LogisticRegression(C=0.08,max_iter=4000); lr.fit(sc.transform(Xs),ys,sample_weight=ws)
    gb=[HistGradientBoostingClassifier(random_state=s,**GBM).fit(Xs,ys,sample_weight=ws) for s in (0,1)]
    return lr.predict_proba(sc.transform(Xap)), np.mean([g.predict_proba(Xap) for g in gb],0)
BRIER_REF=np.mean(np.sum((prior[None]-Y)**2,1)); LOG_REF=np.mean(-np.sum(Y*np.log(prior[None]),1)+np.sum(Y*np.log(np.clip(Y,1e-12,1)),1)); CONF_REF=np.mean(np.abs(prior.max()-Y.max(1)))
def brow(p,y): return np.sum((p-y)**2,1)
def bss_grp(p,y,g):
    out={}
    for gg in np.unique(g):
        m=g==gg
        if m.sum()>=3: out[gg]=1-np.mean(brow(p[m],y[m]))/(np.mean(brow(np.tile(prior,(m.sum(),1)),y[m]))+1e-9)
    return out
def composite(p,conf):
    sb=np.clip(1-np.mean(brow(p,Y))/BRIER_REF,0,1)
    le=np.mean(-np.sum(Y*np.log(np.clip(p,1e-12,1)),1)+np.sum(Y*np.log(np.clip(Y,1e-12,1)),1)); sl=np.clip(1-le/LOG_REF,0,1)
    scf=np.clip(1-np.mean(np.abs(conf-Y.max(1)))/CONF_REF,0,1)
    sw=min(min(bss_grp(p,Y,first2).values()), min(bss_grp(p,Y,first1).values()), min(bss_grp(p,Y,plyb).values()))
    return 0.55*sb+0.20*sl+0.10*scf+0.15*sw, sb,sl,scf,sw
def calib(p,lam,T): q=np.exp(np.log(np.clip(p,1e-9,1))/T); q/=q.sum(1,keepdims=True); q=(1-lam)*q+lam*prior[None]; return q/q.sum(1,keepdims=True)

kf=KFold(5,shuffle=True,random_state=42); oof_lr=np.zeros((N,3)); oof_g=np.zeros((N,3))
for tri,vai in kf.split(np.arange(N)):
    plr,pg=fitp(tri,vai); oof_lr[vai]=plr; oof_g[vai]=pg
# confidence: fit a*maxp+b + inflation via CV oof
def best_conf(p):
    mp=p.max(1); A=np.vstack([mp,np.ones_like(mp)]).T; a,b=np.linalg.lstsq(A,Y.max(1),rcond=None)[0]; return np.clip(a*mp+b,0,1)
print(f'{"wg":>4} {"lam":>4} {"T":>4} | {"comp":>6} {"final":>6} | Sb Sl Sc Sw',flush=True)
best=(-9,None)
for wg in [0.5,0.7,0.85,1.0]:
    bl=wg*oof_g+(1-wg)*oof_lr
    for lam in [0.05,0.15,0.25,0.35,0.45,0.55]:
        for T in [1.0,1.2,1.5]:
            c=calib(bl,lam,T); conf=best_conf(c); comp,sb,sl,scf,sw=composite(c,conf)
            if comp>best[0]: best=(comp,(wg,lam,T),(sb,sl,scf,sw),c,conf)
comp,(wg,lam,T),(sb,sl,scf,sw),cbest,confbest=best
print(f'BEST wg={wg} lam={lam} T={T}: comp={comp:.4f} final={0.12+0.88*comp:.4f} | Sb={sb:.3f} Sl={sl:.3f} Sc={scf:.3f} Sw={sw:.3f}',flush=True)
# show worst groups
g2=bss_grp(cbest,Y,first2); worst=sorted(g2.items(),key=lambda x:x[1])[:6]
print('worst first2 groups (skill, n):',[(k,round(v,3),int((first2==k).sum())) for k,v in worst],flush=True)

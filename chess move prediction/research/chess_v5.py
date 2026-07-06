"""Chess v5 FINAL: blend position-CNN (piece placement) + hierarchical opening-rate + GBM/LR.
Optimize the true composite (S_brier vs worst-group), fit confidence, build the submission."""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from chess_features import apply_moves
torch.manual_seed(0); np.random.seed(0)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr=pd.read_csv(os.path.join(HERE,'dataset','train.csv')); te=pd.read_csv(os.path.join(HERE,'dataset','test.csv'))
RATE=['white_win_rate','draw_rate','black_win_rate']; N=len(tr); cnt=tr['cohort_game_count'].values.astype(np.float32); Y=tr[RATE].values.astype(np.float32)
prior=np.average(Y,0,weights=cnt)
first1=tr['move_prefix'].apply(lambda s:s.split()[0]).values
plyb=np.where(tr['prefix_ply_count'].values<=10,'p10',np.where(tr['prefix_ply_count'].values<=12,'p12','p14+'))
# ---- board planes + scalar board feats ----
def extract(df):
    R=[apply_moves(s,return_board=True) for s in df['move_prefix']]
    PL=np.stack([r[1] for r in R]); ply=df['prefix_ply_count'].values.astype(np.float32)
    return PL, ply
PLtr,PLYtr=extract(tr); PLte,PLYte=extract(te)
OPS_tr={k:tr['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values for k in [2,3,4,5]}
OPS_te={k:te['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values for k in [2,3,4,5]}
BK=['mat_diff','cap_diff','n_captures','n_checks','castle_diff','queens_on','queen_diff','pawn_diff','minor_diff','rook_diff','center_diff','dev_diff','mat_total']
def board_sc(df): return np.column_stack([np.array([[apply_moves(s)[k] for k in BK] for s in df['move_prefix']],np.float32), df['prefix_ply_count'].values.astype(np.float32)])
BFtr,BFte=board_sc(tr),board_sc(te)
def catd(df):
    out=[]
    for s in df['move_prefix']:
        t=s.split(); d={'op2='+' '.join(t[:2]):1.,'op3='+' '.join(t[:3]):1.,'op4='+' '.join(t[:4]):1.}
        for i in range(min(4,len(t))): d[f'm{i}={t[i]}']=1.
        out.append(d)
    return out
dv=DictVectorizer(sparse=False); CATtr=dv.fit_transform(catd(tr)); CATte=dv.transform(catd(te)); keep=CATtr.sum(0)>=10; CATtr,CATte=CATtr[:,keep],CATte[:,keep]
def hier_te(tri, ops):
    L={}
    for k in [2,3,4,5]:
        ss=defaultdict(lambda:np.zeros(3)); ws=defaultdict(float)
        for o,yy,ww in zip(OPS_tr[k][tri],Y[tri],cnt[tri]): ss[o]+=yy*ww; ws[o]+=ww
        L[k]=(ss,ws)
    n=len(ops[2]); out=np.zeros((n,3),np.float32)
    for i in range(n):
        p=prior.copy()
        for kk,a in zip([2,3,4,5],(30,25,20,15)):
            o=ops[kk][i]; ss,ws=L[kk]
            if o in ws: p=(ss[o]+p*a)/(ws[o]+a)
        out[i]=p
    return out
def expand(X,idx): return np.vstack([X,X,X]),np.concatenate([np.zeros(len(idx)),np.ones(len(idx)),2*np.ones(len(idx))]),np.concatenate([Y[idx,0]*cnt[idx],Y[idx,1]*cnt[idx],Y[idx,2]*cnt[idx]])
GBM=dict(max_leaf_nodes=8,min_samples_leaf=50,learning_rate=0.03,max_iter=350,l2_regularization=2.0)
def feat_models(tri, te_tr, X_tr):
    Xs,ys,ws=expand(X_tr,tri); sc=StandardScaler().fit(Xs)
    lr=LogisticRegression(C=0.1,max_iter=3000).fit(sc.transform(Xs),ys,sample_weight=ws)
    gb=[HistGradientBoostingClassifier(random_state=s,**GBM).fit(Xs,ys,sample_weight=ws) for s in (0,1)]
    return sc,lr,gb
class PNet(nn.Module):
    def __init__(s):
        super().__init__(); s.c=nn.Sequential(nn.Conv2d(12,48,3,padding=1),nn.ReLU(),nn.BatchNorm2d(48),nn.Conv2d(48,48,3,padding=1),nn.ReLU(),nn.BatchNorm2d(48),nn.AdaptiveAvgPool2d(1))
        s.f=nn.Sequential(nn.Flatten(),nn.Dropout(0.5),nn.Linear(49,32),nn.ReLU(),nn.Dropout(0.4),nn.Linear(32,3))
    def forward(s,x,p): return s.f(torch.cat([s.c(x).flatten(1),p],1))
def pos_train_pred(tri, PLp, PLYp, seeds=3, epochs=60):
    Xt=torch.tensor(PLtr[tri]).to(DEV); yt=torch.tensor(Y[tri]).to(DEV); wt=torch.tensor(np.sqrt(cnt[tri])).to(DEV); pt=torch.tensor(((PLYtr[tri]-11)/3)[:,None]).to(DEV)
    Xp=torch.tensor(PLp).to(DEV); pp=torch.tensor(((PLYp-11)/3)[:,None]).to(DEV); outs=[]
    for sd in range(seeds):
        torch.manual_seed(sd); m=PNet().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=3e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
        for ep in range(epochs):
            m.train(); perm=torch.randperm(len(tri))
            for k in range(0,len(tri),128):
                idx=perm[k:k+128]; lp=F.log_softmax(m(Xt[idx],pt[idx]),1); loss=-(wt[idx]*(yt[idx]*lp).sum(1)).sum()/wt[idx].sum()
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        m.eval()
        with torch.no_grad(): outs.append(F.softmax(m(Xp,pp),1).cpu().numpy())
    return np.mean(outs,0)
BRIER_REF=np.mean(np.sum((prior[None]-Y)**2,1)); LOG_REF=np.mean(-np.sum(Y*np.log(prior[None]),1)+np.sum(Y*np.log(np.clip(Y,1e-12,1)),1)); CONF_REF=np.mean(np.abs(prior.max()-Y.max(1)))
def brow(p,y): return np.sum((p-y)**2,1)
def wgm(p,g):
    o=[]
    for gg in np.unique(g):
        m=g==gg
        if m.sum()>=3: o.append(1-np.mean(brow(p[m],Y[m]))/(np.mean(brow(np.tile(prior,(m.sum(),1)),Y[m]))+1e-9))
    return min(o) if o else 0.
def comp(p,conf):
    sb=np.clip(1-np.mean(brow(p,Y))/BRIER_REF,0,1)
    le=np.mean(-np.sum(Y*np.log(np.clip(p,1e-12,1)),1)+np.sum(Y*np.log(np.clip(Y,1e-12,1)),1)); sl=np.clip(1-le/LOG_REF,0,1)
    scf=np.clip(1-np.mean(np.abs(conf-Y.max(1)))/CONF_REF,0,1)
    sw=min(wgm(p,first1),wgm(p,plyb))
    return 0.55*sb+0.20*sl+0.10*scf+0.15*sw, sb,sl,scf,sw
def calib(p,lam,T): q=np.exp(np.log(np.clip(p,1e-9,1))/T); q/=q.sum(1,keepdims=True); q=(1-lam)*q+lam*prior[None]; return q/q.sum(1,keepdims=True)

print('computing OOF...',flush=True)
kf=KFold(5,shuffle=True,random_state=42); feat_oof=np.zeros((N,3)); pos_oof=np.zeros((N,3))
for tri,vai in kf.split(np.arange(N)):
    te_tr=hier_te(tri,{k:OPS_tr[k][tri] for k in [2,3,4,5]}); te_va=hier_te(tri,{k:OPS_tr[k][vai] for k in [2,3,4,5]})
    Xtr=np.concatenate([BFtr[tri],CATtr[tri],te_tr],1); Xva=np.concatenate([BFtr[vai],CATtr[vai],te_va],1)
    sc,lr,gb=feat_models(tri,te_tr,Xtr)
    plr=lr.predict_proba(sc.transform(Xva)); pg=np.mean([g.predict_proba(Xva) for g in gb],0)
    feat_oof[vai]=0.5*te_va+0.35*pg+0.15*plr
    pos_oof[vai]=pos_train_pred(tri, PLtr[vai], PLYtr[vai])
feat_oof/=feat_oof.sum(1,keepdims=True); pos_oof/=pos_oof.sum(1,keepdims=True)
RD=os.path.dirname(os.path.abspath(__file__))
np.save(RD+'/a_feat_oof.npy',feat_oof); np.save(RD+'/a_pos_oof.npy',pos_oof)
best=(-9,None)
for wp in [0.3,0.4,0.5,0.6]:
    bl=wp*pos_oof+(1-wp)*feat_oof; bl/=bl.sum(1,keepdims=True)
    mp=bl.max(1); a,b=np.linalg.lstsq(np.vstack([mp,np.ones_like(mp)]).T,Y.max(1),rcond=None)[0]
    for lam in [0.15,0.25,0.35,0.45]:
        for T in [1.0,1.2,1.4]:
            c=calib(bl,lam,T); conf=np.clip(a*c.max(1)+b,0,1); cp,sb,sl,scf,sw=comp(c,conf)
            if cp>best[0]: best=(cp,(wp,lam,T,a,b),(sb,sl,scf,sw))
cp,(wp,lam,T,a,b),(sb,sl,scf,sw)=best
print(f'BEST wp={wp} lam={lam} T={T}: comp={cp:.4f} final={0.12+0.88*cp:.4f} | Sb={sb:.3f} Sl={sl:.3f} Sc={scf:.3f} Sw={sw:.3f}',flush=True)

# ---- submission ----
print('building submission...',flush=True)
te_all=hier_te(np.arange(N),{k:OPS_tr[k] for k in [2,3,4,5]}); te_test=hier_te(np.arange(N),{k:OPS_te[k] for k in [2,3,4,5]})
Xall=np.concatenate([BFtr,CATtr,te_all],1); Xtest=np.concatenate([BFte,CATte,te_test],1)
sc,lr,gb=feat_models(np.arange(N),te_all,Xall)
plr=lr.predict_proba(sc.transform(Xtest)); pg=np.mean([g.predict_proba(Xtest) for g in gb],0)
feat_test=0.5*te_test+0.35*pg+0.15*plr; feat_test/=feat_test.sum(1,keepdims=True)
pos_test=pos_train_pred(np.arange(N), PLte, PLYte, seeds=5); pos_test/=pos_test.sum(1,keepdims=True)
np.save(RD+'/a_feat_test.npy',feat_test); np.save(RD+'/a_pos_test.npy',pos_test)
bl=wp*pos_test+(1-wp)*feat_test; bl/=bl.sum(1,keepdims=True)
final=calib(bl,lam,T); conf_t=np.clip(a*final.max(1)+b,0,1)
sub=pd.DataFrame({'id':te['id'].values,'white_win_prob':final[:,0],'draw_prob':final[:,1],'black_win_prob':final[:,2],'confidence':conf_t})
os.makedirs(os.path.join(HERE,'working'),exist_ok=True); sub.to_csv(os.path.join(HERE,'working','submission.csv'),index=False)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),'v5_params.npy'),np.array([wp,lam,T,a,b]))
print('wrote submission',sub.shape,'| wp',wp,'lam',lam,'T',T,flush=True)

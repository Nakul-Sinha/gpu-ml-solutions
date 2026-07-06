"""Fast calibration iterator on saved v5 model outputs. Adds support-based ADAPTIVE shrinkage:
low-support (rare) openings shrink hard toward prior (rescues worst-group) while high-support
openings keep their signal (preserves S_brier). Optimizes the true composite, builds submission."""
import os, sys, numpy as np, pandas as pd
from collections import defaultdict
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RD=os.path.dirname(os.path.abspath(__file__))
tr=pd.read_csv(os.path.join(HERE,'dataset','train.csv')); te=pd.read_csv(os.path.join(HERE,'dataset','test.csv'))
RATE=['white_win_rate','draw_rate','black_win_rate']; N=len(tr); cnt=tr['cohort_game_count'].values.astype(float); Y=tr[RATE].values
prior=np.average(Y,0,weights=cnt)
first1=tr['move_prefix'].apply(lambda s:s.split()[0]).values
plyb=np.where(tr['prefix_ply_count'].values<=10,'p10',np.where(tr['prefix_ply_count'].values<=12,'p12','p14+'))
feat_oof=np.load(RD+'/a_feat_oof.npy'); pos_oof=np.load(RD+'/a_pos_oof.npy'); feat_test=np.load(RD+'/a_feat_test.npy'); pos_test=np.load(RD+'/a_pos_test.npy')
gru_oof=np.load(RD+'/a_gru_oof.npy'); gru_test=np.load(RD+'/a_gru_test.npy')
# ---- support: train game-count of each row's first-4 opening (back off to first-2) ----
def opk(df,k): return df['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values
sup4=defaultdict(float); sup2=defaultdict(float)
for o4,o2,c in zip(opk(tr,4),opk(tr,2),cnt): sup4[o4]+=c; sup2[o2]+=c
def support(df):
    s=np.zeros(len(df))
    for i,(o4,o2) in enumerate(zip(opk(df,4),opk(df,2))): s[i]=sup4.get(o4,0) if sup4.get(o4,0)>0 else sup2.get(o2,0)
    return s
sup_tr=support(tr); sup_te=support(te)
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
    scf=np.clip(1-np.mean(np.abs(conf-Y.max(1)))/CONF_REF,0,1); sw=min(wgm(p,first1),wgm(p,plyb))
    return 0.55*sb+0.20*sl+0.10*scf+0.15*sw, sb,sl,scf,sw
def acalib(p,sup,lo,hi,tau,T):
    lam=lo+(hi-lo)*np.exp(-sup/tau)
    q=np.exp(np.log(np.clip(p,1e-9,1))/T); q/=q.sum(1,keepdims=True); q=(1-lam[:,None])*q+lam[:,None]*prior[None]; return q/q.sum(1,keepdims=True)
WTS=[(wp,wg,round(1-wp-wg,3)) for wp in [0.25,0.3,0.35,0.4] for wg in [0.25,0.3,0.35,0.4] if 0.15<=1-wp-wg<=0.5]
best=(-9,None)
for wp,wg,wf in WTS:
    bl=wp*pos_oof+wg*gru_oof+wf*feat_oof; bl/=bl.sum(1,keepdims=True)
    mp=bl.max(1); a,b=np.linalg.lstsq(np.vstack([mp,np.ones_like(mp)]).T,Y.max(1),rcond=None)[0]
    for lo in [0.03,0.08,0.13]:
        for hi in [0.45,0.6,0.75,0.9]:
            for tau in [60,120,250]:
                for T in [1.0,1.15,1.3]:
                    c=acalib(bl,sup_tr,lo,hi,tau,T); conf=np.clip(a*c.max(1)+b,0,1); cp,sb,sl,scf,sw=comp(c,conf)
                    if cp>best[0]: best=(cp,(wp,wg,wf,lo,hi,tau,T,a,b),(sb,sl,scf,sw))
cp,(wp,wg,wf,lo,hi,tau,T,a,b),(sb,sl,scf,sw)=best
print(f'BEST wp={wp} wg={wg} wf={wf} lo={lo} hi={hi} tau={tau} T={T}: comp={cp:.4f} final={0.12+0.88*cp:.4f}',flush=True)
print(f'  Sbrier={sb:.3f} Slog={sl:.3f} Sconf={scf:.3f} Sworst={sw:.3f}',flush=True)
# submission
bl=wp*pos_test+wg*gru_test+wf*feat_test; bl/=bl.sum(1,keepdims=True)
final=acalib(bl,sup_te,lo,hi,tau,T); conf_t=np.clip(a*final.max(1)+b,0,1)
sub=pd.DataFrame({'id':te['id'].values,'white_win_prob':final[:,0],'draw_prob':final[:,1],'black_win_prob':final[:,2],'confidence':conf_t})
os.makedirs(os.path.join(HERE,'working'),exist_ok=True); sub.to_csv(os.path.join(HERE,'working','submission.csv'),index=False)
np.save(RD+'/v6_params.npy',np.array([wp,wg,wf,lo,hi,tau,T,a,b]))
print('wrote submission',sub.shape,flush=True)

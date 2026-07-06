"""Chess v7: ROBUST calibration. The 0.266-OOF model scored 0.245 live because the adaptive
shrinkage overfit the OOF worst-group. Fix: shrink rows where the three models DISAGREE
(grouping-independent uncertainty) or the opening is rare, and select shrinkage against a
HARSHER worst-group proxy (finer opening groups + cohort-size buckets) so it stops overfitting."""
import os, numpy as np, pandas as pd
from collections import defaultdict
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RD=os.path.dirname(os.path.abspath(__file__))
tr=pd.read_csv(os.path.join(HERE,'dataset','train.csv')); te=pd.read_csv(os.path.join(HERE,'dataset','test.csv'))
RATE=['white_win_rate','draw_rate','black_win_rate']; N=len(tr); cnt=tr['cohort_game_count'].values.astype(float); Y=tr[RATE].values
prior=np.average(Y,0,weights=cnt)
fo=np.load(RD+'/a_feat_oof.npy'); po=np.load(RD+'/a_pos_oof.npy'); go=np.load(RD+'/a_gru_oof.npy')
ft=np.load(RD+'/a_feat_test.npy'); pt=np.load(RD+'/a_pos_test.npy'); gt=np.load(RD+'/a_gru_test.npy')
def gk(df,k): return df['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values
first1=df1=tr['move_prefix'].apply(lambda s:s.split()[0]).values
first2=gk(tr,2); first3=gk(tr,3)
plyb=np.where(tr['prefix_ply_count'].values<=10,'p10',np.where(tr['prefix_ply_count'].values<=12,'p12','p14+'))
sizeb=np.where(cnt<=14,'s1',np.where(cnt<=18,'s2',np.where(cnt<=26,'s3',np.where(cnt<=45,'s4','s5'))))
sup4=defaultdict(float); sup2=defaultdict(float)
for o4,o2,c in zip(gk(tr,4),gk(tr,2),cnt): sup4[o4]+=c; sup2[o2]+=c
def support(df):
    return np.array([sup4.get(o4,0) if sup4.get(o4,0)>0 else sup2.get(o2,0) for o4,o2 in zip(gk(df,4),gk(df,2))])
sup_tr=support(tr); sup_te=support(te)
BRIER_REF=np.mean(np.sum((prior[None]-Y)**2,1)); LOG_REF=np.mean(-np.sum(Y*np.log(prior[None]),1)+np.sum(Y*np.log(np.clip(Y,1e-12,1)),1)); CONF_REF=np.mean(np.abs(prior.max()-Y.max(1)))
def brow(p,y): return np.sum((p-y)**2,1)
def wgm(p,g,minsz=3):
    o=[]
    for gg in np.unique(g):
        m=g==gg
        if m.sum()>=minsz: o.append(1-np.mean(brow(p[m],Y[m]))/(np.mean(brow(np.tile(prior,(m.sum(),1)),Y[m]))+1e-9))
    return min(o) if o else 0.
def worst_harsh(p):
    return min(wgm(p,first1),wgm(p,first2,8),wgm(p,first3,8),wgm(p,plyb),wgm(p,sizeb))
def comp(p,conf):
    sb=np.clip(1-np.mean(brow(p,Y))/BRIER_REF,0,1)
    le=np.mean(-np.sum(Y*np.log(np.clip(p,1e-12,1)),1)+np.sum(Y*np.log(np.clip(Y,1e-12,1)),1)); sl=np.clip(1-le/LOG_REF,0,1)
    scf=np.clip(1-np.mean(np.abs(conf-Y.max(1)))/CONF_REF,0,1); sw=worst_harsh(p)
    return 0.55*sb+0.20*sl+0.10*scf+0.15*sw, sb,sl,scf,sw
def disagree(pa,pb,pc):
    return (np.abs(pa-pb).sum(1)+np.abs(pa-pc).sum(1)+np.abs(pb-pc).sum(1))/3.0
def acalib(p,feat,sup,dis,lo,hi,tau,dsc,tau2,T):
    rar=np.exp(-sup/tau); dn=np.clip(dis/dsc,0,1); lam=lo+(hi-lo)*np.maximum(rar,dn)
    wsup=(sup/(sup+tau2))[:,None]; target=wsup*feat+(1-wsup)*prior[None]
    q=np.exp(np.log(np.clip(p,1e-9,1))/T); q/=q.sum(1,keepdims=True); q=(1-lam[:,None])*q+lam[:,None]*target; return q/q.sum(1,keepdims=True)
dis_tr=disagree(po,go,fo); dis_te=disagree(pt,gt,ft)
best=(-9,None)
for wp,wg,wf in [(0.4,0.35,0.25),(0.35,0.35,0.3),(0.4,0.3,0.3),(0.45,0.35,0.2),(0.3,0.4,0.3)]:
    bl=wp*po+wg*go+wf*fo; bl/=bl.sum(1,keepdims=True)
    mp=bl.max(1); a,b=np.linalg.lstsq(np.vstack([mp,np.ones_like(mp)]).T,Y.max(1),rcond=None)[0]
    for lo in [0.05,0.1,0.15]:
        for hi in [0.5,0.65,0.8]:
            for tau in [80,150]:
                for dsc in [0.4,0.6]:
                    for tau2 in [100,250,600]:
                        for T in [1.0,1.15]:
                            c=acalib(bl,fo,sup_tr,dis_tr,lo,hi,tau,dsc,tau2,T); conf=np.clip(a*c.max(1)+b,0,1); cp,sb,sl,scf,sw=comp(c,conf)
                            if cp>best[0]: best=(cp,(wp,wg,wf,lo,hi,tau,dsc,tau2,T,a,b),(sb,sl,scf,sw))
cp,(wp,wg,wf,lo,hi,tau,dsc,tau2,T,a,b),(sb,sl,scf,sw)=best
print(f'BEST wp={wp} wg={wg} wf={wf} lo={lo} hi={hi} tau={tau} dsc={dsc} tau2={tau2} T={T}',flush=True)
print(f'  HARSH-proxy: comp={cp:.4f} final={0.12+0.88*cp:.4f} | Sb={sb:.3f} Sl={sl:.3f} Sc={scf:.3f} Sw_harsh={sw:.3f}',flush=True)
# also report against the OLD lenient proxy for reference
def worst_lenient(p): return min(wgm(p,first1),wgm(p,plyb))
bl=wp*po+wg*go+wf*fo; bl/=bl.sum(1,keepdims=True); c=acalib(bl,fo,sup_tr,dis_tr,lo,hi,tau,dsc,tau2,T); conf=np.clip(a*c.max(1)+b,0,1)
print(f'  Sw_lenient(first1/ply)={worst_lenient(c):.3f} | worst first3(min8)={wgm(c,first3,8):.3f}',flush=True)
# submission
blt=wp*pt+wg*gt+wf*ft; blt/=blt.sum(1,keepdims=True)
fin=acalib(blt,ft,sup_te,dis_te,lo,hi,tau,dsc,tau2,T); conf_t=np.clip(a*fin.max(1)+b,0,1)
sub=pd.DataFrame({'id':te['id'].values,'white_win_prob':fin[:,0],'draw_prob':fin[:,1],'black_win_prob':fin[:,2],'confidence':conf_t})
os.makedirs(os.path.join(HERE,'working'),exist_ok=True); sub.to_csv(os.path.join(HERE,'working','submission.csv'),index=False)
np.save(RD+'/v7_params.npy',np.array([wp,wg,wf,lo,hi,tau,dsc,tau2,T,a,b]))
print('wrote submission',sub.shape,flush=True)

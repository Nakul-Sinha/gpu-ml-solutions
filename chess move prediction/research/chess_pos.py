"""Position CNN: learn outcome distribution from the 12x8x8 piece-placement planes of the
final position (reconstructed from the SAN prefix). Trained with count-weighted soft cross
entropy. Tests whether piece placement carries more signal than crude material features."""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from sklearn.model_selection import KFold
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from chess_features import apply_moves
torch.manual_seed(0); np.random.seed(0)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr=pd.read_csv(os.path.join(HERE,'dataset','train.csv'))
RATE=['white_win_rate','draw_rate','black_win_rate']; N=len(tr); cnt=tr['cohort_game_count'].values.astype(np.float32); Y=tr[RATE].values.astype(np.float32)
prior=np.average(Y,0,weights=cnt)
PL=np.stack([apply_moves(s,return_board=True)[1] for s in tr['move_prefix']])  # (N,12,8,8)
first1=tr['move_prefix'].apply(lambda s:s.split()[0]).values
plyf=tr['prefix_ply_count'].values.astype(np.float32)
BRIER_REF=np.mean(np.sum((prior[None]-Y)**2,1))
def brow(p,y): return np.sum((p-y)**2,1)
def sbrier(p): return 1-np.mean(brow(p,Y))/BRIER_REF
def worst(p):
    o=[]
    for g in (first1, np.where(plyf<=10,'a',np.where(plyf<=12,'b','c'))):
        for gg in np.unique(g):
            m=g==gg
            if m.sum()>=3: o.append(1-np.mean(brow(p[m],Y[m]))/(np.mean(brow(np.tile(prior,(m.sum(),1)),Y[m]))+1e-9))
    return min(o)
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c=nn.Sequential(nn.Conv2d(12,48,3,padding=1),nn.ReLU(),nn.BatchNorm2d(48),
                          nn.Conv2d(48,48,3,padding=1),nn.ReLU(),nn.BatchNorm2d(48),nn.AdaptiveAvgPool2d(1))
        s.f=nn.Sequential(nn.Flatten(),nn.Dropout(0.5),nn.Linear(48+1,32),nn.ReLU(),nn.Dropout(0.4),nn.Linear(32,3))
    def forward(s,x,ply): return s.f(torch.cat([s.c(x).flatten(1),ply],1))
def train(tri,vai,epochs=60):
    Xt=torch.tensor(PL[tri]).to(DEV); yt=torch.tensor(Y[tri]).to(DEV); wt=torch.tensor(np.sqrt(cnt[tri])).to(DEV); pt=torch.tensor(((plyf[tri]-11)/3)[:,None]).to(DEV)
    Xv=torch.tensor(PL[vai]).to(DEV); pv=torch.tensor(((plyf[vai]-11)/3)[:,None]).to(DEV)
    m=Net().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=3e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    for ep in range(epochs):
        m.train(); perm=torch.randperm(len(tri))
        for k in range(0,len(tri),128):
            idx=perm[k:k+128]; lp=F.log_softmax(m(Xt[idx],pt[idx]),1)
            loss=-(wt[idx]*(yt[idx]*lp).sum(1)).sum()/wt[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad(): return F.softmax(m(Xv,pv),1).cpu().numpy()
if __name__=='__main__':
    kf=KFold(5,shuffle=True,random_state=42); oof=np.zeros((N,3))
    for tri,vai in kf.split(np.arange(N)): oof[vai]=train(tri,vai)
    print(f'POSITION CNN: Sbrier={sbrier(oof):.4f} worst={worst(oof):.4f}',flush=True)
    # blend with prior (shrink) to check calibrated
    for lam in [0,0.2,0.4]:
        q=(1-lam)*oof+lam*prior[None]
        print(f'  +shrink lam={lam}: Sbrier={sbrier(q):.4f} worst={worst(q):.4f}',flush=True)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),'pos_oof.npy'),oof)

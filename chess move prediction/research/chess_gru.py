"""GRU over the SAN token sequence -> outcome distribution (count-weighted soft CE).
Complementary to the final-position CNN and the opening-rate model. Saves OOF + test preds."""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import KFold
torch.manual_seed(0); np.random.seed(0); DEV='cuda' if torch.cuda.is_available() else 'cpu'
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RD=os.path.dirname(os.path.abspath(__file__))
tr=pd.read_csv(os.path.join(HERE,'dataset','train.csv')); te=pd.read_csv(os.path.join(HERE,'dataset','test.csv'))
RATE=['white_win_rate','draw_rate','black_win_rate']; N=len(tr); cnt=tr['cohort_game_count'].values.astype(np.float32); Y=tr[RATE].values.astype(np.float32)
prior=np.average(Y,0,weights=cnt)
# vocab
toks=set()
for s in pd.concat([tr.move_prefix,te.move_prefix]):
    toks.update(s.split())
vocab={t:i+1 for i,t in enumerate(sorted(toks))}; V=len(vocab)+1; MAXLEN=20
def enc(df):
    X=np.zeros((len(df),MAXLEN),np.int64); L=np.zeros(len(df),np.int64)
    for i,s in enumerate(df.move_prefix):
        t=s.split()[:MAXLEN]; X[i,:len(t)]=[vocab[x] for x in t]; L[i]=len(t)
    return X,L
Xtr,Ltr=enc(tr); Xte,Lte=enc(te)
BRIER_REF=np.mean(np.sum((prior[None]-Y)**2,1))
def sbrier(p): return 1-np.mean(np.sum((p-Y)**2,1))/BRIER_REF
class GRU(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Embedding(V,28,padding_idx=0); s.drop=nn.Dropout(0.4)
        s.gru=nn.GRU(28,40,batch_first=True); s.fc=nn.Sequential(nn.Dropout(0.4),nn.Linear(40,3))
    def forward(s,x,l):
        e=s.drop(s.emb(x)); o,h=s.gru(e); return s.fc(h[-1])
def train_pred(tri, Xp, seeds=3, epochs=45):
    Xt=torch.tensor(Xtr[tri]).to(DEV); Lt=torch.tensor(Ltr[tri]).to(DEV); yt=torch.tensor(Y[tri]).to(DEV); wt=torch.tensor(np.sqrt(cnt[tri])).to(DEV)
    Xpp=torch.tensor(Xp).to(DEV); outs=[]
    for sd in range(seeds):
        torch.manual_seed(sd); m=GRU().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=2e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
        for ep in range(epochs):
            m.train(); perm=torch.randperm(len(tri))
            for k in range(0,len(tri),128):
                idx=perm[k:k+128]; lp=F.log_softmax(m(Xt[idx],None),1); loss=-(wt[idx]*(yt[idx]*lp).sum(1)).sum()/wt[idx].sum()
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        m.eval()
        with torch.no_grad(): outs.append(F.softmax(m(Xpp,None),1).cpu().numpy())
    return np.mean(outs,0)
kf=KFold(5,shuffle=True,random_state=42); oof=np.zeros((N,3))
for tri,vai in kf.split(np.arange(N)): oof[vai]=train_pred(tri,Xtr[vai])
print(f'GRU: Sbrier(raw)={sbrier(oof):.4f}',flush=True)
for lam in [0.2,0.4]: print(f'  shrink {lam}: Sbrier={sbrier((1-lam)*oof+lam*prior[None]):.4f}',flush=True)
np.save(RD+'/a_gru_oof.npy',oof); np.save(RD+'/a_gru_test.npy',train_pred(np.arange(N),Xte,seeds=5))
print('saved gru preds',flush=True)

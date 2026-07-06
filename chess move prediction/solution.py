import os, re, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
try: torch.cuda.manual_seed_all(SEED)
except Exception: pass
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

PIECE_VALUE = {'P':1,'N':3,'B':3,'R':5,'Q':9,'K':0}
FILES = 'abcdefgh'
SAN_RE = re.compile(r'^([KQRBN])?([a-h])?([1-8])?(x)?([a-h][1-8])(=[QRBN])?([+#])?$')
def initial_board():
    b=[[None]*8 for _ in range(8)]; back=['R','N','B','Q','K','B','N','R']
    for f in range(8):
        b[0][f]=('w',back[f]); b[1][f]=('w','P'); b[6][f]=('b','P'); b[7][f]=('b',back[f])
    return b
def _knight(r,f):
    for dr,df in [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]:
        nr,nf=r+dr,f+df
        if 0<=nr<8 and 0<=nf<8: yield nr,nf
def _king(r,f):
    for dr in (-1,0,1):
        for df in (-1,0,1):
            if dr or df:
                nr,nf=r+dr,f+df
                if 0<=nr<8 and 0<=nf<8: yield nr,nf
_SLIDE={'B':[(1,1),(1,-1),(-1,1),(-1,-1)],'R':[(1,0),(-1,0),(0,1),(0,-1)],'Q':[(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)]}
def _slide(board,piece,color,tr,tf):
    out=[]
    for dr,df in _SLIDE[piece]:
        r,f=tr+dr,tf+df
        while 0<=r<8 and 0<=f<8:
            cell=board[r][f]
            if cell is not None:
                if cell==(color,piece): out.append((r,f))
                break
            r+=dr; f+=df
    return out
def _src(board,color,piece,tr,tf,dfile,drank):
    if piece=='N': c=[(r,f) for r,f in _knight(tr,tf) if board[r][f]==(color,'N')]
    elif piece=='K': c=[(r,f) for r,f in _king(tr,tf) if board[r][f]==(color,'K')]
    else: c=_slide(board,piece,color,tr,tf)
    if dfile is not None: c=[x for x in c if x[1]==dfile]
    if drank is not None: c=[x for x in c if x[0]==drank]
    return c[0] if c else None
def replay(move_prefix):
    board=initial_board(); tokens=move_prefix.split()
    ncap=nchk=npr=0; castle={'w':0,'b':0}; captured={'w':0,'b':0}; ep=None
    for i,tok in enumerate(tokens):
        color='w' if i%2==0 else 'b'; new_ep=None
        if tok.startswith(('O-O-O','0-0-0')):
            r=0 if color=='w' else 7; board[r][2]=(color,'K'); board[r][3]=(color,'R'); board[r][4]=None; board[r][0]=None
            castle[color]=2; nchk+=('+' in tok or '#' in tok); ep=None; continue
        if tok.startswith(('O-O','0-0')):
            r=0 if color=='w' else 7; board[r][6]=(color,'K'); board[r][5]=(color,'R'); board[r][4]=None; board[r][7]=None
            castle[color]=1; nchk+=('+' in tok or '#' in tok); ep=None; continue
        m=SAN_RE.match(tok)
        if not m: ep=None; continue
        piece,dfile,drank,capx,dest,promo,chk=m.groups(); tf=FILES.index(dest[0]); tr=int(dest[1])-1
        if chk: nchk+=1
        is_cap=capx is not None
        if piece is None:
            d=1 if color=='w' else -1
            if is_cap:
                sf=FILES.index(dfile) if dfile else tf; sr=tr-d; tgt=board[tr][tf]
                if tgt is None and ep==(tr,tf):
                    cr=tr-d
                    if board[cr][tf] is not None: captured[color]+=PIECE_VALUE['P']; ncap+=1; board[cr][tf]=None
                elif tgt is not None: captured[color]+=PIECE_VALUE[tgt[1]]; ncap+=1
                if 0<=sr<8: board[sr][sf]=None
                board[tr][tf]=(color,promo[1] if promo else 'P'); npr+=bool(promo)
            else:
                sr=tr-d
                if 0<=sr<8 and board[sr][tf]==(color,'P'): board[sr][tf]=None
                else:
                    sr2=tr-2*d
                    if 0<=sr2<8 and board[sr2][tf]==(color,'P'): board[sr2][tf]=None; new_ep=(tr-d,tf)
                board[tr][tf]=(color,promo[1] if promo else 'P'); npr+=bool(promo)
        else:
            src=_src(board,color,piece,tr,tf,FILES.index(dfile) if dfile else None,(int(drank)-1) if drank else None)
            if is_cap and board[tr][tf] is not None: captured[color]+=PIECE_VALUE[board[tr][tf][1]]; ncap+=1
            if src is not None: board[src[0]][src[1]]=None
            board[tr][tf]=(color,piece)
        ep=new_ep
    planes=np.zeros((12,8,8),np.float32); pidx={'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5}
    mat={'w':0,'b':0}; ct={(c,p):0 for c in 'wb' for p in 'PNBRQK'}; cen={'w':0,'b':0}; dev={'w':0,'b':0}; csq={(3,3),(3,4),(4,3),(4,4)}
    for r in range(8):
        for f in range(8):
            cell=board[r][f]
            if cell is None: continue
            col,pc=cell; planes[pidx[pc]+(0 if col=='w' else 6),r,f]=1.0; ct[(col,pc)]+=1; mat[col]+=PIECE_VALUE[pc]
            if (r,f) in csq: cen[col]+=1
            if pc in ('N','B') and r!=(0 if col=='w' else 7): dev[col]+=1
    feat=[mat['w']-mat['b'],captured['w']-captured['b'],ncap,nchk,(castle['w']>0)-(castle['b']>0),ct[('w','Q')]+ct[('b','Q')],
          ct[('w','Q')]-ct[('b','Q')],ct[('w','P')]-ct[('b','P')],(ct[('w','N')]+ct[('w','B')])-(ct[('b','N')]+ct[('b','B')]),
          ct[('w','R')]-ct[('b','R')],cen['w']-cen['b'],dev['w']-dev['b'],mat['w']+mat['b']]
    return planes, np.array(feat,np.float32)

def find_ds():
    here=os.path.dirname(os.path.abspath(__file__))
    for c in [os.path.join(here,'dataset','public'),os.path.join(here,'dataset'),'./dataset/public','./dataset']:
        if os.path.exists(os.path.join(c,'train.csv')) and os.path.exists(os.path.join(c,'test.csv')): return c
    raise FileNotFoundError('dataset')
DS=find_ds()
tr=pd.read_csv(os.path.join(DS,'train.csv')); te=pd.read_csv(os.path.join(DS,'test.csv'))
RATE=['white_win_rate','draw_rate','black_win_rate']; N=len(tr); cnt=tr['cohort_game_count'].values.astype(np.float32); Y=tr[RATE].values.astype(np.float32)
prior=np.average(Y,0,weights=cnt)
Rtr=[replay(s) for s in tr['move_prefix']]; Rte=[replay(s) for s in te['move_prefix']]
PLtr=np.stack([r[0] for r in Rtr]); PLte=np.stack([r[0] for r in Rte])
PLYtr=tr['prefix_ply_count'].values.astype(np.float32); PLYte=te['prefix_ply_count'].values.astype(np.float32)
BFtr=np.column_stack([np.stack([r[1] for r in Rtr]),PLYtr]); BFte=np.column_stack([np.stack([r[1] for r in Rte]),PLYte])
OPS_tr={k:tr['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values for k in [2,3,4,5]}
OPS_te={k:te['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values for k in [2,3,4,5]}
def catd(df):
    out=[]
    for s in df['move_prefix']:
        t=s.split(); d={'op2='+' '.join(t[:2]):1.,'op3='+' '.join(t[:3]):1.,'op4='+' '.join(t[:4]):1.}
        for i in range(min(4,len(t))): d[f'm{i}={t[i]}']=1.
        out.append(d)
    return out
dv=DictVectorizer(sparse=False); CATtr=dv.fit_transform(catd(tr)); CATte=dv.transform(catd(te)); keep=CATtr.sum(0)>=10; CATtr,CATte=CATtr[:,keep],CATte[:,keep]
vocab={}
for s in pd.concat([tr.move_prefix,te.move_prefix]):
    for t in s.split(): vocab.setdefault(t,len(vocab)+1)
Vsz=len(vocab)+1; ML=20
def seqenc(df):
    X=np.zeros((len(df),ML),np.int64)
    for i,s in enumerate(df.move_prefix):
        t=s.split()[:ML]; X[i,:len(t)]=[vocab[x] for x in t]
    return X
SXtr=seqenc(tr); SXte=seqenc(te)
def hier_te(tri,ops):
    L={}
    for k in [2,3,4,5]:
        ss=defaultdict(lambda:np.zeros(3)); ws=defaultdict(float)
        for o,yy,ww in zip(OPS_tr[k][tri],Y[tri],cnt[tri]): ss[o]+=yy*ww; ws[o]+=ww
        L[k]=(ss,ws)
    n=len(ops[2]); out=np.zeros((n,3),np.float32)
    for i in range(n):
        p=prior.copy()
        for kk,al in zip([2,3,4,5],(30,25,20,15)):
            o=ops[kk][i]; ss,ws=L[kk]
            if o in ws: p=(ss[o]+p*al)/(ws[o]+al)
        out[i]=p
    return out
def expand(X,idx): return np.vstack([X,X,X]),np.concatenate([np.zeros(len(idx)),np.ones(len(idx)),2*np.ones(len(idx))]),np.concatenate([Y[idx,0]*cnt[idx],Y[idx,1]*cnt[idx],Y[idx,2]*cnt[idx]])
GBM=dict(max_leaf_nodes=8,min_samples_leaf=50,learning_rate=0.03,max_iter=350,l2_regularization=2.0)
def feat_predict(tri,ap_idx):
    te_tr=hier_te(tri,{k:OPS_tr[k][tri] for k in [2,3,4,5]}); te_ap=hier_te(tri,{k:OPS_tr[k][ap_idx] for k in [2,3,4,5]})
    Xtr=np.concatenate([BFtr[tri],CATtr[tri],te_tr],1); Xap=np.concatenate([BFtr[ap_idx],CATtr[ap_idx],te_ap],1)
    Xs,ys,ws=expand(Xtr,tri); sc=StandardScaler().fit(Xs)
    lr=LogisticRegression(C=0.1,max_iter=3000).fit(sc.transform(Xs),ys,sample_weight=ws)
    gb=[HistGradientBoostingClassifier(random_state=s,**GBM).fit(Xs,ys,sample_weight=ws) for s in (0,1)]
    p=0.5*te_ap+0.35*np.mean([g.predict_proba(Xap) for g in gb],0)+0.15*lr.predict_proba(sc.transform(Xap)); return p/p.sum(1,keepdims=True)
def feat_predict_test():
    te_tr=hier_te(np.arange(N),{k:OPS_tr[k] for k in [2,3,4,5]}); te_ap=hier_te(np.arange(N),{k:OPS_te[k] for k in [2,3,4,5]})
    Xtr=np.concatenate([BFtr,CATtr,te_tr],1); Xap=np.concatenate([BFte,CATte,te_ap],1)
    Xs,ys,ws=expand(Xtr,np.arange(N)); sc=StandardScaler().fit(Xs)
    lr=LogisticRegression(C=0.1,max_iter=3000).fit(sc.transform(Xs),ys,sample_weight=ws)
    gb=[HistGradientBoostingClassifier(random_state=s,**GBM).fit(Xs,ys,sample_weight=ws) for s in (0,1)]
    p=0.5*te_ap+0.35*np.mean([g.predict_proba(Xap) for g in gb],0)+0.15*lr.predict_proba(sc.transform(Xap)); return p/p.sum(1,keepdims=True)
class PNet(nn.Module):
    def __init__(s):
        super().__init__(); s.c=nn.Sequential(nn.Conv2d(12,48,3,padding=1),nn.ReLU(),nn.BatchNorm2d(48),nn.Conv2d(48,48,3,padding=1),nn.ReLU(),nn.BatchNorm2d(48),nn.AdaptiveAvgPool2d(1))
        s.f=nn.Sequential(nn.Flatten(),nn.Dropout(0.5),nn.Linear(49,32),nn.ReLU(),nn.Dropout(0.4),nn.Linear(32,3))
    def forward(s,x,p): return s.f(torch.cat([s.c(x).flatten(1),p],1))
class GNet(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Embedding(Vsz,28,padding_idx=0); s.drop=nn.Dropout(0.4); s.gru=nn.GRU(28,40,batch_first=True); s.fc=nn.Sequential(nn.Dropout(0.4),nn.Linear(40,3))
    def forward(s,x): o,h=s.gru(s.drop(s.emb(x))); return s.fc(h[-1])
def train_net(kind,tri,pred_pack,seeds):
    yt=torch.tensor(Y[tri]).to(DEV); wt=torch.tensor(np.sqrt(cnt[tri])).to(DEV); outs=[]
    epochs=60 if kind=='pos' else 45
    if kind=='pos': Xt=torch.tensor(PLtr[tri]).to(DEV); pt=torch.tensor(((PLYtr[tri]-11)/3)[:,None]).to(DEV)
    else: Xt=torch.tensor(SXtr[tri]).to(DEV)
    for sd in range(seeds):
        torch.manual_seed(sd); m=(PNet() if kind=='pos' else GNet()).to(DEV)
        opt=torch.optim.AdamW(m.parameters(),lr=2e-3 if kind=='pos' else 3e-3,weight_decay=3e-3 if kind=='pos' else 2e-3)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
        for ep in range(epochs):
            m.train(); perm=torch.randperm(len(tri))
            for k in range(0,len(tri),128):
                idx=perm[k:k+128]
                out=m(Xt[idx],pt[idx]) if kind=='pos' else m(Xt[idx])
                loss=-(wt[idx]*(yt[idx]*F.log_softmax(out,1)).sum(1)).sum()/wt[idx].sum()
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        m.eval()
        with torch.no_grad():
            out=m(*pred_pack) if kind=='pos' else m(pred_pack[0])
            outs.append(F.softmax(out,1).cpu().numpy())
    return np.mean(outs,0)
def pos_pack(idx): return (torch.tensor(PLtr[idx]).to(DEV),torch.tensor(((PLYtr[idx]-11)/3)[:,None]).to(DEV))
def gru_pack(idx): return (torch.tensor(SXtr[idx]).to(DEV),)
def net_oof(kind):
    oof=np.zeros((N,3)); kf=KFold(5,shuffle=True,random_state=42)
    for tri,vai in kf.split(np.arange(N)):
        oof[vai]=train_net(kind,tri,pos_pack(vai) if kind=='pos' else gru_pack(vai),3)
    return oof

def main():
    print('computing OOF...',flush=True)
    kf=KFold(5,shuffle=True,random_state=42); feat_oof=np.zeros((N,3))
    for tri,vai in kf.split(np.arange(N)): feat_oof[vai]=feat_predict(tri,vai)
    pos_oof=net_oof('pos'); gru_oof=net_oof('gru')
    print('training final models...',flush=True)
    feat_te=feat_predict_test()
    pos_te=train_net('pos',np.arange(N),(torch.tensor(PLte).to(DEV),torch.tensor(((PLYte-11)/3)[:,None]).to(DEV)),5)
    gru_te=train_net('gru',np.arange(N),(torch.tensor(SXte).to(DEV),),5)
    wp,wg,wf,lo,hi,tau,T=0.35,0.35,0.30,0.03,0.75,60.0,1.0
    def opk(df,k): return df['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values
    sup4=defaultdict(float); sup2=defaultdict(float)
    for o4,o2,c in zip(opk(tr,4),opk(tr,2),cnt): sup4[o4]+=c; sup2[o2]+=c
    def support(df):
        s=np.zeros(len(df))
        for i,(o4,o2) in enumerate(zip(opk(df,4),opk(df,2))): s[i]=sup4.get(o4,0) if sup4.get(o4,0)>0 else sup2.get(o2,0)
        return s
    def acalib(p,sup):
        lam=lo+(hi-lo)*np.exp(-sup/tau); q=np.exp(np.log(np.clip(p,1e-9,1))/T); q/=q.sum(1,keepdims=True)
        q=(1-lam[:,None])*q+lam[:,None]*prior[None]; return q/q.sum(1,keepdims=True)
    bl_oof=wp*pos_oof+wg*gru_oof+wf*feat_oof; bl_oof/=bl_oof.sum(1,keepdims=True)
    c_oof=acalib(bl_oof,support(tr)); mp=c_oof.max(1); a,b=np.linalg.lstsq(np.vstack([mp,np.ones_like(mp)]).T,Y.max(1),rcond=None)[0]
    bl=wp*pos_te+wg*gru_te+wf*feat_te; bl/=bl.sum(1,keepdims=True)
    fin=acalib(bl,support(te)); conf=np.clip(a*fin.max(1)+b,0,1)
    sub=pd.DataFrame({'id':te['id'].values,'white_win_prob':fin[:,0],'draw_prob':fin[:,1],'black_win_prob':fin[:,2],'confidence':conf})
    os.makedirs('./working',exist_ok=True); sub.to_csv('./working/submission.csv',index=False)
    print('wrote ./working/submission.csv',sub.shape,flush=True)

if __name__=='__main__':
    main()

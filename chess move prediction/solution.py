import os, re, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import RepeatedKFold

SEED = 42
np.random.seed(SEED)

PIECE_VALUE = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
FILES = 'abcdefgh'
SAN_RE = re.compile(r'^([KQRBN])?([a-h])?([1-8])?(x)?([a-h][1-8])(=[QRBN])?([+#])?$')

def initial_board():
    b = [[None]*8 for _ in range(8)]
    back = ['R','N','B','Q','K','B','N','R']
    for f in range(8):
        b[0][f] = ('w', back[f]); b[1][f] = ('w','P')
        b[6][f] = ('b','P');      b[7][f] = ('b', back[f])
    return b

def _knight(r, f):
    for dr, df in [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]:
        nr, nf = r+dr, f+df
        if 0 <= nr < 8 and 0 <= nf < 8: yield nr, nf

def _king(r, f):
    for dr in (-1,0,1):
        for df in (-1,0,1):
            if dr or df:
                nr, nf = r+dr, f+df
                if 0 <= nr < 8 and 0 <= nf < 8: yield nr, nf

_SLIDE = {'B':[(1,1),(1,-1),(-1,1),(-1,-1)], 'R':[(1,0),(-1,0),(0,1),(0,-1)],
          'Q':[(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)]}

def _slide_sources(board, piece, color, tr, tf):
    out = []
    for dr, df in _SLIDE[piece]:
        r, f = tr+dr, tf+df
        while 0 <= r < 8 and 0 <= f < 8:
            cell = board[r][f]
            if cell is not None:
                if cell == (color, piece): out.append((r, f))
                break
            r += dr; f += df
    return out

def _find_source(board, color, piece, tr, tf, dfile, drank):
    if piece == 'N':
        c = [(r,f) for r,f in _knight(tr,tf) if board[r][f]==(color,'N')]
    elif piece == 'K':
        c = [(r,f) for r,f in _king(tr,tf) if board[r][f]==(color,'K')]
    else:
        c = _slide_sources(board, piece, color, tr, tf)
    if dfile is not None: c = [x for x in c if x[1]==dfile]
    if drank is not None: c = [x for x in c if x[0]==drank]
    return c[0] if c else None

def board_features(move_prefix):
    board = initial_board()
    tokens = move_prefix.split()
    n_captures = n_checks = n_promo = 0
    castle = {'w':0,'b':0}; captured = {'w':0,'b':0}; ep = None
    for i, tok in enumerate(tokens):
        color = 'w' if i % 2 == 0 else 'b'
        new_ep = None
        if tok.startswith(('O-O-O','0-0-0')):
            r = 0 if color=='w' else 7
            board[r][2]=(color,'K'); board[r][3]=(color,'R'); board[r][4]=None; board[r][0]=None
            castle[color]=2; n_checks += ('+' in tok or '#' in tok); ep=None; continue
        if tok.startswith(('O-O','0-0')):
            r = 0 if color=='w' else 7
            board[r][6]=(color,'K'); board[r][5]=(color,'R'); board[r][4]=None; board[r][7]=None
            castle[color]=1; n_checks += ('+' in tok or '#' in tok); ep=None; continue
        m = SAN_RE.match(tok)
        if not m: ep=None; continue
        piece, dfile, drank, capx, dest, promo, chk = m.groups()
        tf = FILES.index(dest[0]); tr = int(dest[1]) - 1
        if chk: n_checks += 1
        is_cap = capx is not None
        if piece is None:
            direction = 1 if color=='w' else -1
            if is_cap:
                sf = FILES.index(dfile) if dfile else tf
                sr = tr - direction
                tgt = board[tr][tf]
                if tgt is None and ep == (tr, tf):
                    cr = tr - direction
                    if board[cr][tf] is not None:
                        captured[color]+=PIECE_VALUE['P']; n_captures+=1; board[cr][tf]=None
                elif tgt is not None:
                    captured[color]+=PIECE_VALUE[tgt[1]]; n_captures+=1
                if 0 <= sr < 8: board[sr][sf]=None
                board[tr][tf]=(color, promo[1] if promo else 'P'); n_promo += bool(promo)
            else:
                sr = tr - direction
                if 0 <= sr < 8 and board[sr][tf]==(color,'P'):
                    board[sr][tf]=None
                else:
                    sr2 = tr - 2*direction
                    if 0 <= sr2 < 8 and board[sr2][tf]==(color,'P'):
                        board[sr2][tf]=None; new_ep=(tr-direction, tf)
                board[tr][tf]=(color, promo[1] if promo else 'P'); n_promo += bool(promo)
        else:
            src = _find_source(board, color, piece, tr, tf,
                               FILES.index(dfile) if dfile else None,
                               (int(drank)-1) if drank else None)
            if is_cap and board[tr][tf] is not None:
                captured[color]+=PIECE_VALUE[board[tr][tf][1]]; n_captures+=1
            if src is not None: board[src[0]][src[1]]=None
            board[tr][tf]=(color, piece)
        ep = new_ep
    mat={'w':0,'b':0}; cnt={(c,p):0 for c in 'wb' for p in 'PNBRQK'}
    center={'w':0,'b':0}; dev={'w':0,'b':0}; csq={(3,3),(3,4),(4,3),(4,4)}
    for r in range(8):
        for f in range(8):
            cell=board[r][f]
            if cell is None: continue
            col,pc=cell; cnt[(col,pc)]+=1; mat[col]+=PIECE_VALUE[pc]
            if (r,f) in csq: center[col]+=1
            if pc in ('N','B') and r != (0 if col=='w' else 7): dev[col]+=1
    return {
        'mat_diff':mat['w']-mat['b'],'cap_diff':captured['w']-captured['b'],'n_captures':n_captures,
        'n_checks':n_checks,'castle_diff':(castle['w']>0)-(castle['b']>0),
        'w_castled':int(castle['w']>0),'b_castled':int(castle['b']>0),
        'queens_on':cnt[('w','Q')]+cnt[('b','Q')],'queen_diff':cnt[('w','Q')]-cnt[('b','Q')],
        'pawn_diff':cnt[('w','P')]-cnt[('b','P')],
        'minor_diff':(cnt[('w','N')]+cnt[('w','B')])-(cnt[('b','N')]+cnt[('b','B')]),
        'rook_diff':cnt[('w','R')]-cnt[('b','R')],'center_diff':center['w']-center['b'],
        'dev_diff':dev['w']-dev['b'],'dev_total':dev['w']+dev['b'],'mat_total':mat['w']+mat['b'],
        'n_promo':n_promo,'bishop_pair_w':int(cnt[('w','B')]>=2),'bishop_pair_b':int(cnt[('b','B')]>=2)}

def find_dataset():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.path.join(here,'dataset','public'), os.path.join(here,'dataset'),
              './dataset/public', './dataset']:
        if os.path.exists(os.path.join(c,'train.csv')) and os.path.exists(os.path.join(c,'test.csv')):
            return c
    raise FileNotFoundError('Could not locate dataset train.csv/test.csv')

def main():
    DS = find_dataset()
    tr = pd.read_csv(os.path.join(DS,'train.csv'))
    te = pd.read_csv(os.path.join(DS,'test.csv'))
    RATE = ['white_win_rate','draw_rate','black_win_rate']
    N = len(tr); cnt = tr['cohort_game_count'].values.astype(float); Y = tr[RATE].values
    prior = np.average(Y, 0, weights=cnt)

    BK = ['mat_diff','cap_diff','n_captures','n_checks','castle_diff','w_castled','b_castled',
          'queens_on','queen_diff','pawn_diff','minor_diff','rook_diff','center_diff','dev_diff',
          'dev_total','mat_total','n_promo','bishop_pair_w','bishop_pair_b']
    def board_mat(df):
        R=[board_features(s) for s in df['move_prefix']]
        M=np.array([[r[k] for k in BK] for r in R], np.float32)
        return np.column_stack([M, df['prefix_ply_count'].values.astype(np.float32)])
    Btr, Bte = board_mat(tr), board_mat(te)
    OPKS=[2,3,4,6]
    opv=lambda df,k: df['move_prefix'].apply(lambda s:' '.join(s.split()[:k])).values
    op_tr={k:opv(tr,k) for k in OPKS}; op_te={k:opv(te,k) for k in OPKS}
    def cat_dicts(df):
        out=[]
        for s in df['move_prefix']:
            t=s.split(); d={'op2='+' '.join(t[:2]):1.0,'op4='+' '.join(t[:4]):1.0}
            for i in range(min(6,len(t))): d[f'm{i}={t[i]}']=1.0
            out.append(d)
        return out
    dv=DictVectorizer(sparse=False); Ctr=dv.fit_transform(cat_dicts(tr)); Cte=dv.transform(cat_dicts(te))
    keep=Ctr.sum(0)>=8; Ctr,Cte=Ctr[:,keep],Cte[:,keep]

    def tenc(otr, oap, idx, alpha=20.0):
        ss=defaultdict(lambda:np.zeros(3)); ws=defaultdict(float)
        for o,yy,ww in zip(otr[idx],Y[idx],cnt[idx]): ss[o]+=yy*ww; ws[o]+=ww
        out=np.zeros((len(oap),3),np.float32)
        for i,o in enumerate(oap): out[i]=(ss[o]+prior*alpha)/(ws[o]+alpha) if o in ws else prior
        return out
    def feats_tr(idx_tr, idx_ap):
        TE=np.concatenate([tenc(op_tr[k],op_tr[k][idx_ap],idx_tr) for k in OPKS],1)
        return np.concatenate([Btr[idx_ap],Ctr[idx_ap],TE],1)
    def feats_te(idx_tr):
        TE=np.concatenate([tenc(op_tr[k],op_te[k],idx_tr) for k in OPKS],1)
        return np.concatenate([Bte,Cte,TE],1)
    def expand(X, idx):
        return (np.vstack([X,X,X]),
                np.concatenate([np.zeros(len(idx)),np.ones(len(idx)),2*np.ones(len(idx))]),
                np.concatenate([Y[idx,0]*cnt[idx],Y[idx,1]*cnt[idx],Y[idx,2]*cnt[idx]]))
    GBM=dict(max_leaf_nodes=8,min_samples_leaf=50,learning_rate=0.03,max_iter=400,l2_regularization=1.0)
    def fit(idx):
        X=feats_tr(idx,idx); Xs,ys,ws=expand(X,idx)
        sc=StandardScaler().fit(Xs)
        lr=LogisticRegression(C=0.1,max_iter=4000); lr.fit(sc.transform(Xs),ys,sample_weight=ws)
        gbms=[HistGradientBoostingClassifier(random_state=s,**GBM).fit(Xs,ys,sample_weight=ws) for s in (0,1,2)]
        return sc,lr,gbms
    def pred(models, X):
        sc,lr,gbms=models
        return lr.predict_proba(sc.transform(X)), np.mean([g.predict_proba(X) for g in gbms],0)

    def brier(p,y): return np.mean(np.sum((p-y)**2,1))
    def bss(p,y,w):
        pr=np.average(y,0,weights=w); return 1-brier(p,y)/np.mean(np.sum((pr[None]-y)**2,1))
    def calib(p,lam,T):
        q=np.exp(np.log(np.clip(p,1e-9,1))/T); q/=q.sum(1,keepdims=True)
        q=(1-lam)*q+lam*prior[None]; return q/q.sum(1,keepdims=True)

    print('running repeated CV for calibration...', flush=True)
    rkf=RepeatedKFold(n_splits=5,n_repeats=3,random_state=SEED)
    oof_lr=np.zeros((N,3)); oof_g=np.zeros((N,3)); ck=np.zeros(N)
    for tri,vai in rkf.split(np.arange(N)):
        models=fit(tri); Xv=feats_tr(tri,vai); plr,pg=pred(models,Xv)
        oof_lr[vai]+=plr; oof_g[vai]+=pg; ck[vai]+=1
    oof_lr/=ck[:,None]; oof_g/=ck[:,None]
    best=(-9,None)
    for wg in np.linspace(0,1,11):
        bl=wg*oof_g+(1-wg)*oof_lr
        for lam in [0,0.02,0.05,0.08,0.12,0.16,0.2,0.25]:
            for T in [0.8,0.9,1.0,1.1,1.25,1.5]:
                s=bss(calib(bl,lam,T),Y,cnt)
                if s>best[0]: best=(s,(wg,lam,T))
    s,(wg,lam,T)=best
    oofc=calib(wg*oof_g+(1-wg)*oof_lr,lam,T)
    mp=oofc.max(1); my=Y.max(1)
    a,b=np.linalg.lstsq(np.vstack([mp,np.ones_like(mp)]).T,my,rcond=None)[0]
    print(f'selected wg={wg:.2f} lam={lam} T={T} | OOF BSS={s:.4f} confMAE={np.mean(np.abs(np.clip(a*mp+b,0,1)-my)):.4f}', flush=True)

    models=fit(np.arange(N)); Xte=feats_te(np.arange(N)); plr,pg=pred(models,Xte)
    p=calib(wg*pg+(1-wg)*plr,lam,T)
    conf=np.clip(a*p.max(1)+b,0.0,1.0)
    sub=pd.DataFrame({'id':te['id'].values,'white_win_prob':p[:,0],'draw_prob':p[:,1],
                      'black_win_prob':p[:,2],'confidence':conf})
    os.makedirs('./working', exist_ok=True)
    sub.to_csv('./working/submission.csv', index=False)
    print('wrote ./working/submission.csv', sub.shape, flush=True)
    print(sub.head().to_string(index=False))

if __name__ == '__main__':
    main()

"""Chess final recipe: GBM+LR blend over generalizable features (board replay + opening
target-encoding + coarse one-hots), calibrated (temperature + prior shrinkage), with a
fitted confidence map. Repeated CV to pick calibration; refit on all data for test.
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import RepeatedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chess_features import apply_moves

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(HERE, 'dataset', 'train.csv'))
te = pd.read_csv(os.path.join(HERE, 'dataset', 'test.csv'))
RATE = ['white_win_rate', 'draw_rate', 'black_win_rate']
N = len(tr); cnt = tr['cohort_game_count'].values.astype(float); Y = tr[RATE].values
prior = np.average(Y, 0, weights=cnt)

BK = ['mat_diff','cap_diff','n_captures','n_checks','castle_diff','w_castled','b_castled','queens_on',
      'queen_diff','pawn_diff','minor_diff','rook_diff','center_diff','dev_diff','dev_total','mat_total',
      'n_promo','bishop_pair_w','bishop_pair_b']
def board_mat(df):
    R = [apply_moves(s) for s in df['move_prefix']]
    M = np.array([[r[k] for k in BK] for r in R], np.float32)
    return np.column_stack([M, df['prefix_ply_count'].values.astype(np.float32)])
Btr, Bte = board_mat(tr), board_mat(te)
OPKS = [2, 3, 4, 6]
opv = lambda df, k: df['move_prefix'].apply(lambda s: ' '.join(s.split()[:k])).values
op_tr = {k: opv(tr, k) for k in OPKS}; op_te = {k: opv(te, k) for k in OPKS}

def cat_dicts(df):
    out = []
    for s in df['move_prefix']:
        t = s.split(); d = {'op2=' + ' '.join(t[:2]): 1.0, 'op4=' + ' '.join(t[:4]): 1.0}
        for i in range(min(6, len(t))): d[f'm{i}={t[i]}'] = 1.0
        out.append(d)
    return out
dv = DictVectorizer(sparse=False)
Ctr = dv.fit_transform(cat_dicts(tr)); Cte = dv.transform(cat_dicts(te))
keep = Ctr.sum(0) >= 8; Ctr, Cte = Ctr[:, keep], Cte[:, keep]

def tenc(otr, oap, idx, alpha=20.0):
    from collections import defaultdict
    ss = defaultdict(lambda: np.zeros(3)); ws = defaultdict(float)
    for o, yy, ww in zip(otr[idx], Y[idx], cnt[idx]):
        ss[o] += yy * ww; ws[o] += ww
    out = np.zeros((len(oap), 3), np.float32)
    for i, o in enumerate(oap):
        out[i] = (ss[o] + prior * alpha) / (ws[o] + alpha) if o in ws else prior
    return out
def te_block(idx_tr, which):  # which: 'tr' or 'te'
    src = op_tr if which == 'tr' else op_te
    idxmap = idx_tr if which == 'tr' else np.arange(len(te))
    return np.concatenate([tenc(op_tr[k], src[k][idxmap] if which=='tr' else op_te[k], idx_tr) for k in OPKS], 1)

def feats(idx_tr, target='tr', idx_ap=None):
    if target == 'tr':
        ap = idx_ap
        B, C = Btr[ap], Ctr[ap]
        TE = np.concatenate([tenc(op_tr[k], op_tr[k][ap], idx_tr) for k in OPKS], 1)
    else:
        B, C = Bte, Cte
        TE = np.concatenate([tenc(op_tr[k], op_te[k], idx_tr) for k in OPKS], 1)
    return np.concatenate([B, C, TE], 1)

def expand(X, idx):
    return (np.vstack([X, X, X]),
            np.concatenate([np.zeros(len(idx)), np.ones(len(idx)), 2*np.ones(len(idx))]),
            np.concatenate([Y[idx,0]*cnt[idx], Y[idx,1]*cnt[idx], Y[idx,2]*cnt[idx]]))

GBM_KW = dict(max_leaf_nodes=8, min_samples_leaf=50, learning_rate=0.03, max_iter=400, l2_regularization=1.0)
def fit_models(idx_tr, seeds=(0,1,2)):
    Xtr = feats(idx_tr, 'tr', idx_tr); Xs, ys, ws = expand(Xtr, idx_tr)
    sc = StandardScaler().fit(Xs)
    lr = LogisticRegression(C=0.1, max_iter=4000); lr.fit(sc.transform(Xs), ys, sample_weight=ws)
    gbms = []
    for s in seeds:
        g = HistGradientBoostingClassifier(random_state=s, **GBM_KW); g.fit(Xs, ys, sample_weight=ws); gbms.append(g)
    return sc, lr, gbms
def predict_models(models, X):
    sc, lr, gbms = models
    plr = lr.predict_proba(sc.transform(X))
    pg = np.mean([g.predict_proba(X) for g in gbms], 0)
    return plr, pg

# metrics
def brier(p,y): return np.mean(np.sum((p-y)**2,1))
def bss(p,y,w):
    pr=np.average(y,0,weights=w); return 1-brier(p,y)/np.mean(np.sum((pr[None]-y)**2,1))
def logexc(p,y,eps=1e-12):
    p=np.clip(p,eps,1); return np.mean(-np.sum(y*np.log(p),1)+np.sum(y*np.log(np.clip(y,eps,1)),1))
def calib(p,lam,T):
    q=np.exp(np.log(np.clip(p,1e-9,1))/T); q/=q.sum(1,keepdims=True); q=(1-lam)*q+lam*prior[None]; return q/q.sum(1,keepdims=True)

def run():
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    oof_lr = np.zeros((N,3)); oof_g = np.zeros((N,3)); ck = np.zeros(N)
    for tri, vai in rkf.split(np.arange(N)):
        models = fit_models(tri)
        Xv = feats(tri, 'tr', vai)
        plr, pg = predict_models(models, Xv)
        oof_lr[vai] += plr; oof_g[vai] += pg; ck[vai] += 1
    oof_lr /= ck[:,None]; oof_g /= ck[:,None]
    # search blend + calibration to maximize BSS
    best = (-9, None)
    for wg in np.linspace(0,1,11):
        blend = wg*oof_g + (1-wg)*oof_lr
        for lam in [0,0.02,0.05,0.08,0.12,0.16,0.2]:
            for T in [0.8,0.9,1.0,1.1,1.25,1.5]:
                c = calib(blend, lam, T); s = bss(c, Y, cnt)
                if s > best[0]: best = (s, (wg,lam,T))
    s,(wg,lam,T) = best
    blend = wg*oof_g+(1-wg)*oof_lr; oofc = calib(blend,lam,T)
    # confidence map: conf = clip(a*max(p)+b) to predict max(y)
    mp = oofc.max(1); my = Y.max(1)
    A = np.vstack([mp, np.ones_like(mp)]).T
    a,b = np.linalg.lstsq(A, my, rcond=None)[0]
    conf = np.clip(a*mp+b, 0, 1)
    firsttok = tr['move_prefix'].apply(lambda s:s.split()[0]).values
    grp = np.where(firsttok=='e4','e4',np.where(firsttok=='d4','d4','other'))
    gb = {g: round(bss(oofc[grp==g],Y[grp==g],cnt[grp==g]),3) for g in ['e4','d4']}
    print(f'BEST blend wg={wg:.1f} lam={lam} T={T}')
    print(f'  BSS={s:.4f} logexc={logexc(oofc,Y):.4f} grp={gb}')
    print(f'  conf map a={a:.3f} b={b:.3f} mean|conf-maxy|={np.mean(np.abs(conf-my)):.4f} (raw maxp={np.mean(np.abs(mp-my)):.4f})')
    return wg,lam,T,a,b

if __name__ == '__main__':
    run()

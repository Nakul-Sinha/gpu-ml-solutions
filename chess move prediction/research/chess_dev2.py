"""Chess v2: generalizable features only.
- Board features from SAN replay (material, captures, castling, checks, development).
- Out-of-fold target-encoding of opening families (first-k moves) -> smoothed historical
  white/draw/black rates for that opening (generalizes; exact prefixes never recur in test).
- Coarse opening one-hots + early-move one-hots.
Models: regularized multinomial LR + HistGBM (both via count-weighted class expansion).
Blend + calibrate (prior-shrinkage + temperature) tuned to the composite proxy.
"""
import os, sys, numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chess_features import apply_moves

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(HERE, 'dataset', 'train.csv'))
te = pd.read_csv(os.path.join(HERE, 'dataset', 'test.csv'))
RATE = ['white_win_rate', 'draw_rate', 'black_win_rate']
N = len(tr); cnt = tr['cohort_game_count'].values.astype(float)
Y = tr[RATE].values
prior = np.average(Y, axis=0, weights=cnt)

BOARD_KEYS = ['mat_diff','cap_diff','n_captures','n_checks','castle_diff','w_castled','b_castled',
              'queens_on','queen_diff','pawn_diff','minor_diff','rook_diff','center_diff','dev_diff',
              'dev_total','mat_total','n_promo','bishop_pair_w','bishop_pair_b']

def board_matrix(df):
    rows = [apply_moves(s) for s in df['move_prefix']]
    return np.array([[r[k] for k in BOARD_KEYS] for r in rows], np.float32)

def opening(df, k):
    return df['move_prefix'].apply(lambda s: ' '.join(s.split()[:k])).values

Btr = board_matrix(tr); Bte = board_matrix(te)
# add ply
Btr = np.column_stack([Btr, tr['prefix_ply_count'].values.astype(np.float32)])
Bte = np.column_stack([Bte, te['prefix_ply_count'].values.astype(np.float32)])

OPKS = [2, 3, 4, 6]
op_tr = {k: opening(tr, k) for k in OPKS}
op_te = {k: opening(te, k) for k in OPKS}

def target_encode(op_train_vals, op_apply_vals, y, w, alpha=20.0):
    """smoothed count-weighted mean rates per opening key -> (M,3)."""
    from collections import defaultdict
    ssum = defaultdict(lambda: np.zeros(3)); wsum = defaultdict(float)
    for o, yy, ww in zip(op_train_vals, y, w):
        ssum[o] += yy * ww; wsum[o] += ww
    out = np.zeros((len(op_apply_vals), 3), np.float32)
    for i, o in enumerate(op_apply_vals):
        if o in wsum:
            out[i] = (ssum[o] + prior * alpha) / (wsum[o] + alpha)
        else:
            out[i] = prior
    return out

def build_te_features(tri_idx, apply_op_tr, apply_op_ap):
    """out-of-fold safe: encode using only tri_idx rows."""
    feats = []
    for k in OPKS:
        enc = target_encode(op_tr[k][tri_idx], apply_op_ap[k], Y[tri_idx], cnt[tri_idx])
        feats.append(enc)
    return np.concatenate(feats, 1)

# one-hot of frequent openings (first-2, first-4) + early moves
from sklearn.feature_extraction import DictVectorizer
def cat_dicts(df):
    out = []
    for s in df['move_prefix']:
        toks = s.split(); d = {}
        d['op2=' + ' '.join(toks[:2])] = 1.0
        d['op4=' + ' '.join(toks[:4])] = 1.0
        for i in range(min(6, len(toks))):
            d[f'm{i}={toks[i]}'] = 1.0
        out.append(d)
    return out
dv = DictVectorizer(sparse=False)
Ctr = dv.fit_transform(cat_dicts(tr)); Cte = dv.transform(cat_dicts(te))
# prune rare one-hots (freq<8) to reduce overfit
freq = Ctr.sum(0); keep = freq >= 8
Ctr = Ctr[:, keep]; Cte = Cte[:, keep]
print('board', Btr.shape, 'cat(kept)', Ctr.shape)

# ---- metrics ----
def brier(p, y): return np.mean(np.sum((p - y)**2, 1))
def bss(p, y, w):
    pr = np.average(y, 0, weights=w)
    return 1 - brier(p, y) / np.mean(np.sum((pr[None] - y)**2, 1))
def logexc(p, y, eps=1e-12):
    p = np.clip(p, eps, 1); ce = -np.sum(y*np.log(p),1); en=-np.sum(y*np.log(np.clip(y,eps,1)),1)
    return np.mean(ce-en)
def calib(p, lam, T):
    q = np.exp(np.log(np.clip(p,1e-9,1))/T); q/=q.sum(1,keepdims=True)
    q = (1-lam)*q + lam*prior[None]; return q/q.sum(1,keepdims=True)

def expand(X, idx):
    Xs = np.vstack([X, X, X])
    ys = np.concatenate([np.zeros(len(idx)), np.ones(len(idx)), 2*np.ones(len(idx))])
    ws = np.concatenate([Y[idx,0]*cnt[idx], Y[idx,1]*cnt[idx], Y[idx,2]*cnt[idx]])
    return Xs, ys, ws

def fit_predict(Xtr_parts_fn, idx_tr, idx_ap, kind='lr', C=0.3, gbm_kw=None):
    Xtr = np.concatenate(Xtr_parts_fn(idx_tr, idx_tr), 1)
    Xap = np.concatenate(Xtr_parts_fn(idx_tr, idx_ap), 1)
    Xs, ys, ws = expand(Xtr, idx_tr)
    if kind == 'lr':
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xs)
        m = LogisticRegression(C=C, max_iter=3000)
        m.fit(sc.transform(Xs), ys, sample_weight=ws)
        return m.predict_proba(sc.transform(Xap))
    else:
        m = HistGradientBoostingClassifier(**(gbm_kw or {}))
        m.fit(Xs, ys, sample_weight=ws)
        return m.predict_proba(Xap)

# feature assembler: returns list of matrices [board, catoh, te_features]
def parts(idx_tr, idx_ap):
    B = Btr[idx_ap]; C = Ctr[idx_ap]
    TE = build_te_features(idx_tr, op_tr, {k: op_tr[k][idx_ap] for k in OPKS})
    return [B, C, TE]

def run_cv(kind='lr', C=0.3, gbm_kw=None, lam=0.0, T=1.0, seed=42, folds=5, verbose=True):
    kf = KFold(folds, shuffle=True, random_state=seed)
    oof = np.zeros((N,3))
    for tri, vai in kf.split(np.arange(N)):
        # target-encode using tri only, applied to tri (for training X) and vai
        oof[vai] = fit_predict(parts, tri, vai, kind, C, gbm_kw)
    oofc = calib(oof, lam, T)
    s = bss(oofc, Y, cnt); le = logexc(oofc, Y)
    # group BSS by first move and ply bucket
    fm = op_tr[2]  # first2 as proxy; use first token
    firsttok = tr['move_prefix'].apply(lambda s: s.split()[0]).values
    grp = np.where(firsttok=='e4','e4', np.where(firsttok=='d4','d4','other'))
    gb = {g: bss(oofc[grp==g], Y[grp==g], cnt[grp==g]) for g in ['e4','d4','other']}
    plyb = np.where(tr['prefix_ply_count'].values<=10,'p10','p12+')
    pb = {g: bss(oofc[plyb==g], Y[plyb==g], cnt[plyb==g]) for g in np.unique(plyb)}
    worst = min(list(gb.values())+list(pb.values()))
    if verbose:
        print(f'  {kind} C={C} lam={lam} T={T}: BSS={s:.4f} logexc={le:.4f} worst_grp={worst:.4f} grp={ {k:round(v,3) for k,v in gb.items()} } ply={ {k:round(v,3) for k,v in pb.items()} }')
    return oof, oofc, s

if __name__ == '__main__':
    print('prior', prior.round(4))
    print('\n=== LR feature/C sweep ===')
    for C in [0.1, 0.2, 0.3, 0.5]:
        run_cv('lr', C=C)
    print('\n=== GBM sweep ===')
    for mln, msl, lr, it in [(8,50,0.03,400),(15,40,0.03,500),(15,30,0.05,400),(31,20,0.05,400)]:
        run_cv('gbm', gbm_kw=dict(max_leaf_nodes=mln,min_samples_leaf=msl,learning_rate=lr,max_iter=it,l2_regularization=1.0,early_stopping=False))

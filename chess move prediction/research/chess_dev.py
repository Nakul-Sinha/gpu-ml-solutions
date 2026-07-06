"""Chess cohort outcome-distribution CV development.
Approach: expand each cohort into weighted class observations (rate_k * cohort_count),
train a weighted multinomial model -> calibrated P(outcome|features). Blend LR + GBM,
then tune prior-shrinkage + temperature on the exact composite proxy via grouped CV.
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chess_features import apply_moves

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(HERE, 'dataset', 'train.csv'))
te = pd.read_csv(os.path.join(HERE, 'dataset', 'test.csv'))
RATE_COLS = ['white_win_rate', 'draw_rate', 'black_win_rate']

def build_feats(df, max_ply=12):
    rows = []
    for _, r in df.iterrows():
        toks = r['move_prefix'].split()
        d = {}
        # positional move tokens (capture the exact opening line, shared across cohorts)
        for i in range(min(len(toks), max_ply)):
            d[f'm{i}={toks[i]}'] = 1.0
        # move bag (which moves appear) + consecutive 2-grams
        for i, t in enumerate(toks):
            d[f'bag={t}'] = 1.0
            if i + 1 < len(toks):
                d[f'bg={toks[i]}>{toks[i+1]}'] = 1.0
        # board-derived numeric features
        bf = apply_moves(r['move_prefix'])
        for k, v in bf.items():
            d[f'f_{k}'] = float(v)
        d['ply'] = float(r['prefix_ply_count'])
        d['n_tokens'] = float(len(toks))
        rows.append(d)
    return rows

print('building features...')
tr_feats = build_feats(tr)
te_feats = build_feats(te)
dv = DictVectorizer(sparse=True)
Xtr = dv.fit_transform(tr_feats)
Xte = dv.transform(te_feats)
print('feature matrix:', Xtr.shape, 'test', Xte.shape)

Y = tr[RATE_COLS].values  # (N,3) soft targets
N = len(tr)
cnt = tr['cohort_game_count'].values.astype(float)
# global weighted prior
prior = np.average(Y, axis=0, weights=cnt)
print('weighted prior:', prior.round(4))

def expand(X, Yr, w):
    """Stack 3 class copies with sample_weight = rate_k * cohort_count."""
    import scipy.sparse as sp
    Xs = sp.vstack([X, X, X])
    ys = np.concatenate([np.zeros(X.shape[0]), np.ones(X.shape[0]), 2*np.ones(X.shape[0])])
    ws = np.concatenate([Yr[:, 0]*w, Yr[:, 1]*w, Yr[:, 2]*w])
    return Xs, ys, ws

def fit_lr(X, Yr, w, C=1.0):
    Xs, ys, ws = expand(X, Yr, w)
    m = LogisticRegression(C=C, max_iter=2000, multi_class='multinomial', class_weight=None)
    m.fit(Xs, ys, sample_weight=ws)
    return m

def fit_gbm(X, Yr, w, **kw):
    Xs, ys, ws = expand(X.toarray() if hasattr(X, 'toarray') else X, Yr, w)
    m = HistGradientBoostingClassifier(**kw)
    m.fit(Xs, ys, sample_weight=ws)
    return m

# ---------- metrics ----------
def brier(p, y):
    return np.mean(np.sum((p - y)**2, axis=1))

def logloss_excess(p, y, eps=1e-12):
    p = np.clip(p, eps, 1)
    ce = -np.sum(y * np.log(p), axis=1)
    ent = -np.sum(y * np.log(np.clip(y, eps, 1)), axis=1)
    return np.mean(ce - ent)

def bss(p, y, w=None):
    pr = np.average(y, axis=0, weights=w) if w is not None else y.mean(0)
    b = brier(p, y)
    b0 = np.mean(np.sum((pr[None, :] - y)**2, axis=1))
    return 1 - b / b0, b, b0

def calibrate(p, prior, lam, T):
    """temperature then shrink toward prior."""
    q = np.clip(p, 1e-9, 1)
    logit = np.log(q)
    q = np.exp(logit / T); q = q / q.sum(1, keepdims=True)
    q = (1 - lam) * q + lam * prior[None, :]
    return q / q.sum(1, keepdims=True)

# ---------- CV ----------
def run_cv(model_fns, blend_w=None, lam=0.0, T=1.0, seed=42, folds=5, verbose=True):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros((N, 3))
    for tri, vai in kf.split(np.arange(N)):
        preds = []
        for fn in model_fns:
            m = fn(Xtr[tri], Y[tri], cnt[tri])
            if hasattr(m, 'predict_proba'):
                pp = m.predict_proba(Xtr[vai] if 'LogReg' in type(m).__name__ or isinstance(m, LogisticRegression) else Xtr[vai].toarray())
            preds.append(pp)
        if blend_w is None:
            p = np.mean(preds, axis=0)
        else:
            p = sum(w*pp for w, pp in zip(blend_w, preds))
        oof[vai] = p
    oofc = calibrate(oof, prior, lam, T)
    sc, b, b0 = bss(oofc, Y, cnt)
    ll = logloss_excess(oofc, Y)
    if verbose:
        print(f'  BSS={sc:.4f} brier={b:.4f} (prior {b0:.4f}) logexcess={ll:.4f}')
    return oof, oofc

if __name__ == '__main__':
    print('\n=== prior-only baseline ===')
    p0 = np.tile(prior, (N, 1))
    print('  BSS=0 by definition, brier=%.4f' % brier(p0, Y))

    print('\n=== LogisticRegression (C=1) ===')
    oof_lr, _ = run_cv([lambda X, Yr, w: fit_lr(X, Yr, w, C=1.0)])
    print('\n=== LR C sweep ===')
    for C in [0.2, 0.5, 1.0, 2.0]:
        print(f'C={C}', end=' ')
        run_cv([lambda X, Yr, w, C=C: fit_lr(X, Yr, w, C=C)])

    print('\n=== calibration sweep on LR C=0.5 ===')
    best = (-9, None)
    for lam in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]:
        for T in [0.8, 1.0, 1.2, 1.5]:
            oof, oofc = run_cv([lambda X, Yr, w: fit_lr(X, Yr, w, C=0.5)], lam=lam, T=T, verbose=False)
            sc, _, _ = bss(oofc, Y, cnt)
            if sc > best[0]:
                best = (sc, (lam, T))
    print('best cal:', best)

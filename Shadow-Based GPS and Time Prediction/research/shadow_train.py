"""Shadow -> GPS + hour. CNN regressor that learns solar geometry IMPLICITLY from pixels
(no analytical sun equations, no shadow-angle measurement, no ephemeris). doy is provided
as a conditioning input (sin/cos). Longitude & hour use circular (sin/cos) targets; latitude
is regressed directly. Loss blends coordinate MSE with a differentiable Haversine term and a
circular-hour term aligned to the competition score. Config via env vars for local/H100 use.
"""
import os, sys, math, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def env(k, d): return type(d)(os.environ.get(k, d))
SEED = env('SEED', 42)
torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_ds():
    sd = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(HERE, 'dataset', 'public'), os.path.join(HERE, 'dataset'),
             os.path.join(sd, 'dataset', 'public'), os.path.join(sd, 'dataset'),
             './dataset/public', './dataset']
    for c in cands:
        if os.path.exists(os.path.join(c, 'train.csv')): return c
    raise FileNotFoundError('dataset')
DS = find_ds()
BACKBONE = os.environ.get('BB', 'convnext_tiny')
IMG = env('IMG', 256)
EPOCHS = env('EPOCHS', 30)
BS = env('BS', 64)
LR = env('LR', 3e-4)
SUBSET = env('SUBSET', 0)        # >0 -> use only N train images (fast local proto)
VAL_FRAC = env('VAL_FRAC', 0.12)
W_GEO = env('W_GEO', 1.0)
W_HOUR = env('W_HOUR', 1.5)
W_HAV = env('W_HAV', 0.5)
TAG = os.environ.get('TAG', 'run')
PRELOAD = env('PRELOAD', 1)

tr = pd.read_csv(os.path.join(DS, 'train.csv'))
if SUBSET > 0:
    tr = tr.sample(SUBSET, random_state=SEED).reset_index(drop=True)
TRAIN_DIR = os.path.join(DS, 'train')

def doy_feats(doy):
    a = 2 * math.pi * (doy.astype(np.float32) - 1) / 365.0
    return np.stack([np.sin(a), np.cos(a)], 1)

def img_path(folder, iid):
    for ext in ('.png', '.jpg', '.jpeg'):
        p = os.path.join(folder, iid + ext)
        if os.path.exists(p): return p
    return os.path.join(folder, iid + '.png')

def load_img(path, size):
    im = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)

# preload resized images to RAM
def preload(df, folder, size):
    arr = np.zeros((len(df), size, size, 3), np.uint8)
    for i, iid in enumerate(df['image_id'].values):
        arr[i] = load_img(img_path(folder, iid), size)
        if i % 1000 == 0: print(f'  preload {i}/{len(df)}', flush=True)
    return arr

LOAD = env('LOADSZ', IMG + 32)  # load a bit larger for random-crop augmentation
print(f'[{TAG}] backbone={BACKBONE} img={IMG} load={LOAD} epochs={EPOCHS} bs={BS} n={len(tr)} dev={DEV}', flush=True)
IMGS = preload(tr, TRAIN_DIR, LOAD) if PRELOAD else None

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class ShadowDS(Dataset):
    def __init__(self, idx, train=True):
        self.idx = idx; self.train = train
        self.lat = tr['latitude'].values.astype(np.float32)
        self.lon = tr['longitude'].values.astype(np.float32)
        self.hour = tr['hour'].values.astype(np.float32)
        self.df = doy_feats(tr['doy'].values)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        img = IMGS[j]
        # augmentation: random crop (translation) from LOAD->IMG, NO flips/rot (would corrupt azimuth)
        if self.train:
            mx = LOAD - IMG
            ox, oy = np.random.randint(0, mx + 1), np.random.randint(0, mx + 1)
        else:
            ox = oy = (LOAD - IMG) // 2
        img = img[oy:oy + IMG, ox:ox + IMG, :]
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        if self.train:  # mild photometric jitter only (preserve elevation/azimuth cues)
            t = torch.clamp(t * (0.9 + 0.2 * np.random.rand()) + (np.random.rand() - 0.5) * 0.05, 0, 1)
        t = (t - MEAN) / STD
        lat = self.lat[j]; lon = self.lon[j]; hr = self.hour[j]
        lonr = math.radians(lon); hra = 2 * math.pi * hr / 24.0
        tgt = torch.tensor([lat / 90.0, math.sin(lonr), math.cos(lonr),
                            math.sin(hra), math.cos(hra), lat, lon, hr], dtype=torch.float32)
        return t, torch.tensor(self.df[j]), tgt

class Net(nn.Module):
    def __init__(self, name, ncond=2):
        super().__init__()
        self.bb = timm.create_model(name, pretrained=True, num_classes=0, global_pool='avg')
        f = self.bb.num_features
        self.head = nn.Sequential(nn.Linear(f + ncond, 512), nn.SiLU(), nn.Dropout(env('DROP', 0.5)),
                                  nn.Linear(512, 256), nn.SiLU(), nn.Dropout(env('DROP', 0.5) * 0.6),
                                  nn.Linear(256, 5))
    def forward(self, x, cond):
        f = self.bb(x)
        return self.head(torch.cat([f, cond], 1))

def decode(out):
    lat = out[:, 0] * 90.0
    lon = torch.atan2(out[:, 1], out[:, 2]) * 180.0 / math.pi
    ha = torch.atan2(out[:, 3], out[:, 4])
    hour = (ha * 24.0 / (2 * math.pi)) % 24.0
    return lat, lon, hour

def hav_km(lat1, lon1, lat2, lon2):
    r1, r2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    dphi = r2 - r1; dl = torch.deg2rad(lon2 - lon1)
    a = torch.sin(dphi / 2)**2 + torch.cos(r1) * torch.cos(r2) * torch.sin(dl / 2)**2
    return 6371.0 * 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0, 1)))

def loss_fn(out, tgt):
    # Stable smooth targets only: regress raw sin/cos (decode uses atan2, no unit-norm needed).
    # Avoid haversine/arcsin/sqrt AND the x/||x|| normalization in the loss -- under fp16 both
    # produce NaN gradients near the origin and the GradScaler silently skips every step.
    l_lat = F.mse_loss(out[:, 0], tgt[:, 0])
    l_lon = F.mse_loss(out[:, 1:3], tgt[:, 1:3])
    l_hr = F.mse_loss(out[:, 3:5], tgt[:, 3:5])
    return W_GEO * (l_lat + l_lon) + W_HOUR * l_hr

def comp_score(lat_p, lon_p, hr_p, lat_t, lon_t, hr_t):
    d = hav_km(torch.tensor(lat_p), torch.tensor(lon_p), torch.tensor(lat_t), torch.tensor(lon_t)).numpy()
    dpen = np.where(d > 10000, d * 1.5, d)
    dh = np.abs(hr_p - hr_t); dh = np.minimum(dh, 24 - dh)
    geo = 1 / (1 + dpen.mean() / 500); tim = 1 / (1 + dh.mean() / 3)
    lat_mae = np.abs(lat_p - lat_t).mean()
    dlon = np.abs(lon_p - lon_t); dlon = np.minimum(dlon, 360 - dlon); lon_mae = dlon.mean()
    return 0.5 * geo + 0.5 * tim, dpen.mean(), dh.mean(), lat_mae, lon_mae

@torch.no_grad()
def predict(model, dl):
    model.eval(); LA, LO, HR = [], [], []
    for t, c, _ in dl:
        o = model(t.to(DEV), c.to(DEV))
        la, lo, hr = decode(o)
        LA.append(la.cpu().numpy()); LO.append(lo.cpu().numpy()); HR.append(hr.cpu().numpy())
    return np.concatenate(LA), np.concatenate(LO), np.concatenate(HR)

def main():
    n = len(tr); rng = np.random.RandomState(SEED); perm = rng.permutation(n)
    nval = int(n * VAL_FRAC); vai, tri = perm[:nval], perm[nval:]
    dltr = DataLoader(ShadowDS(tri, True), batch_size=BS, shuffle=True, num_workers=env('NW', 4), pin_memory=True, drop_last=True)
    dlva = DataLoader(ShadowDS(vai, False), batch_size=BS, shuffle=False, num_workers=env('NW', 4), pin_memory=True)
    model = Net(BACKBONE).to(DEV)
    lr_bb = env('LR_BB', LR); lr_head = env('LR_HEAD', LR)
    opt = torch.optim.AdamW([{'params': model.bb.parameters(), 'lr': lr_bb},
                             {'params': model.head.parameters(), 'lr': lr_head}], weight_decay=env('WD', 0.05))
    SCHED = os.environ.get('SCHED', 'onecycle'); AMP = env('AMP', 1)
    if SCHED == 'onecycle':
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_bb, lr_head], epochs=EPOCHS, steps_per_epoch=len(dltr), pct_start=0.1)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * len(dltr))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(AMP))
    best = -1
    latv = tr['latitude'].values[vai]; lonv = tr['longitude'].values[vai]; hrv = tr['hour'].values[vai]
    lattr = tr['latitude'].values[tri][:400]; lontr = tr['longitude'].values[tri][:400]; hrtr = tr['hour'].values[tri][:400]
    dltr_eval = DataLoader(ShadowDS(tri[:400], False), batch_size=BS, shuffle=False, num_workers=env('NW', 4))
    for ep in range(EPOCHS):
        model.train(); t0 = time.time()
        for t, c, tg in dltr:
            t, c, tg = t.to(DEV, non_blocking=True), c.to(DEV, non_blocking=True), tg.to(DEV, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(AMP)):
                loss = loss_fn(model(t, c), tg)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
        la, lo, hr = predict(model, dlva)
        sc, dkm, dh, latmae, lonmae = comp_score(la, lo, hr, latv, lonv, hrv)
        lat2, lo2, hr2 = predict(model, dltr_eval)
        sctr, _, _, _, _ = comp_score(lat2, lo2, hr2, lattr, lontr, hrtr)
        best = max(best, sc)
        print(f'[{TAG}] ep{ep} loss{loss.item():.3f} val{sc:.4f} tr{sctr:.4f} geo{dkm:.0f} hrMAE{dh:.2f} latMAE{latmae:.1f} lonMAE{lonmae:.0f} {time.time()-t0:.0f}s', flush=True)
    print(f'[{TAG}] BEST val score = {best:.4f}', flush=True)

if __name__ == '__main__':
    main()

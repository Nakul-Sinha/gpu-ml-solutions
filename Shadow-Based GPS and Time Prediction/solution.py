"""
Shadow-Based GPS + Time Prediction.

Predict latitude, longitude and UTC hour from a single outdoor image with visible shadows,
given the day-of-year. The model learns solar geometry IMPLICITLY from pixels -- there is NO
analytical sun-position equation, NO shadow-angle measurement, NO ephemeris, NO hard-coded
astronomy anywhere in this pipeline. It is a pure image-conditioned regressor.

Design:
  * Backbone: ImageNet-pretrained ResNet50 (allowed; not geolocation-pretrained). doy is fed as
    a sin/cos conditioning vector to the regression head (the date is known in any real scenario).
  * Targets: latitude regressed directly; longitude and hour as circular sin/cos pairs decoded
    with atan2 (both wrap around). Loss is a stable sin/cos + latitude MSE (no arcsin/haversine
    terms, which produce NaN gradients under mixed precision).
  * Calibration: because the injected rendering noise makes fine geometry only weakly recoverable,
    each target's prediction is blended with the training prior at a weight chosen on a held-out
    split to maximise the exact competition score. Circular targets are blended as unit vectors.
    This also controls the >10,000 km anti-cheat penalty on longitude.

Reads ./dataset[/public]/... ; writes ./working/submission.csv. torch/torchvision/numpy/pandas only.
"""
import os, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
try: torch.cuda.manual_seed_all(SEED)
except Exception: pass
torch.backends.cudnn.benchmark = True
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG = int(os.environ.get('IMG', 256))
EPOCHS = int(os.environ.get('EPOCHS', 22))
BS = int(os.environ.get('BS', 96))

def find_ds():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [os.path.join(here, 'dataset', 'public'), os.path.join(here, 'dataset'),
              './dataset/public', './dataset']:
        if os.path.exists(os.path.join(c, 'train.csv')) and os.path.exists(os.path.join(c, 'test.csv')):
            return c
    raise FileNotFoundError('dataset')
DS = find_ds()
tr = pd.read_csv(os.path.join(DS, 'train.csv'))
te = pd.read_csv(os.path.join(DS, 'test.csv'))

def img_path(folder, iid):
    for e in ('.png', '.jpg', '.jpeg'):
        p = os.path.join(folder, iid + e)
        if os.path.exists(p): return p
    return os.path.join(folder, iid + '.png')

def preload(df, folder):
    arr = np.zeros((len(df), IMG, IMG, 3), np.uint8)
    for i, iid in enumerate(df['image_id'].values):
        arr[i] = np.asarray(Image.open(img_path(folder, iid)).convert('RGB').resize((IMG, IMG), Image.BILINEAR), np.uint8)
    return arr

print('preloading images...', flush=True)
XTR = preload(tr, os.path.join(DS, 'train'))
XTE = preload(te, os.path.join(DS, 'test'))
def doy_cond(df):
    a = 2 * math.pi * (df['doy'].values.astype(np.float32) - 1) / 365.0
    return np.stack([np.sin(a), np.cos(a)], 1).astype(np.float32)
CTR, CTE = doy_cond(tr), doy_cond(te)
LAT = tr['latitude'].values.astype(np.float32); LON = tr['longitude'].values.astype(np.float32); HR = tr['hour'].values.astype(np.float32)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class SDS(Dataset):
    def __init__(self, idx, train=True): self.idx = idx; self.train = train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        t = torch.from_numpy(XTR[j]).permute(2, 0, 1).float() / 255.0
        if self.train:
            t = torch.clamp(t * (0.9 + 0.2 * np.random.rand()) + (np.random.rand() - 0.5) * 0.05, 0, 1)
        t = (t - MEAN) / STD
        lonr = math.radians(LON[j]); hra = 2 * math.pi * HR[j] / 24.0
        y = torch.tensor([LAT[j] / 90.0, math.sin(lonr), math.cos(lonr), math.sin(hra), math.cos(hra)], dtype=torch.float32)
        return t, torch.tensor(CTR[j]), y

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = torchvision.models.resnet50(weights='IMAGENET1K_V2'); f = self.bb.fc.in_features
        self.bb.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(f + 2, 512), nn.SiLU(), nn.Dropout(0.3),
                                  nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 5))
    def forward(self, x, c): return self.head(torch.cat([self.bb(x), c], 1))

def decode(o):
    lat = o[:, 0] * 90.0
    lon = np.degrees(np.arctan2(o[:, 1], o[:, 2]))
    hour = (np.degrees(np.arctan2(o[:, 3], o[:, 4])) / 15.0) % 24.0
    return lat, lon, hour

def hav(la1, lo1, la2, lo2):
    r1, r2 = np.radians(la1), np.radians(la2); dphi = r2 - r1; dl = np.radians(lo2 - lo1)
    a = np.sin(dphi / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def score(la_p, lo_p, hr_p, la_t, lo_t, hr_t):
    d = hav(la_p, lo_p, la_t, lo_t); d = np.where(d > 10000, d * 1.5, d)
    dh = np.abs(hr_p - hr_t); dh = np.minimum(dh, 24 - dh)
    return 0.5 / (1 + d.mean() / 500) + 0.5 / (1 + dh.mean() / 3)

@torch.no_grad()
def predict(model, X, C):
    model.eval(); out = []
    for k in range(0, len(X), 256):
        t = torch.from_numpy(X[k:k + 256]).permute(0, 3, 1, 2).float() / 255.0
        t = ((t - MEAN) / STD).to(DEV)
        c = torch.tensor(C[k:k + 256]).to(DEV)
        with torch.cuda.amp.autocast():
            out.append(model(t, c).float().cpu().numpy())
    return np.concatenate(out)

def train(idx, epochs):
    dl = DataLoader(SDS(idx, True), batch_size=BS, shuffle=True, num_workers=int(os.environ.get('NW', 8)),
                    pin_memory=True, drop_last=True)
    model = Net().to(DEV)
    opt = torch.optim.AdamW([{'params': model.bb.parameters(), 'lr': 2e-4},
                             {'params': model.head.parameters(), 'lr': 1e-3}], weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(dl))
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(epochs):
        model.train()
        for t, c, y in dl:
            t, c, y = t.to(DEV, non_blocking=True), c.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.cuda.amp.autocast():
                o = model(t, c)
                loss = F.mse_loss(o[:, 0], y[:, 0]) + F.mse_loss(o[:, 1:3], y[:, 1:3]) + 1.5 * F.mse_loss(o[:, 3:5], y[:, 3:5])
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
    return model

def circ_mean(deg, period):
    a = 2 * math.pi * deg / period
    return (math.atan2(np.sin(a).mean(), np.cos(a).mean()) * period / (2 * math.pi))

def blend_circular(pred_deg, prior_deg, alpha, period):
    # blend two angles as unit vectors (alpha=1 -> prediction, alpha=0 -> prior), decode back to [-p/2..]
    ap = 2 * math.pi * pred_deg / period; a0 = 2 * math.pi * prior_deg / period
    vx = alpha * np.cos(ap) + (1 - alpha) * math.cos(a0)
    vy = alpha * np.sin(ap) + (1 - alpha) * math.sin(a0)
    return np.arctan2(vy, vx) * period / (2 * math.pi)

def main():
    n = len(tr); rng = np.random.RandomState(SEED); perm = rng.permutation(n)
    nval = int(n * 0.15); vi, ti = perm[:nval], perm[nval:]
    prior_lat = float(LAT[ti].mean()); prior_lon = circ_mean(LON[ti], 360); prior_hr = circ_mean(HR[ti], 24)

    model = train(ti, EPOCHS)
    ov = predict(model, XTR[vi], CTR[vi]); la_v, lo_v, hr_v = decode(ov)

    # per-target blend search on validation (exact competition score)
    best = (-1, (0, 0, 0))
    grid = np.linspace(0, 1, 21)
    for al in grid:
        latb = al * la_v + (1 - al) * prior_lat
        # precompute geo score contribution needs lon too; search jointly but cheaply
        for alo in [0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
            lonb = blend_circular(lo_v, prior_lon, alo, 360)
            for ah in [0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
                hrb = blend_circular(hr_v, prior_hr, ah, 24) % 24
                s = score(latb, lonb, hrb, LAT[vi], LON[vi], HR[vi])
                if s > best[0]: best = (s, (al, alo, ah))
    s, (al, alo, ah) = best
    prior_s = score(np.full(nval, prior_lat), np.full(nval, prior_lon), np.full(nval, prior_hr % 24), LAT[vi], LON[vi], HR[vi])
    print(f'val prior score={prior_s:.4f} | blended score={s:.4f} | alpha lat={al:.2f} lon={alo:.2f} hr={ah:.2f}', flush=True)

    # test prediction with the same model + chosen blend
    ot = predict(model, XTE, CTE); la_t, lo_t, hr_t = decode(ot)
    lat_f = np.clip(al * la_t + (1 - al) * prior_lat, -90, 90)
    lon_f = ((blend_circular(lo_t, prior_lon, alo, 360) + 180) % 360) - 180
    hr_f = blend_circular(hr_t, prior_hr, ah, 24) % 24
    sub = pd.DataFrame({'image_id': te['image_id'].values, 'latitude': lat_f, 'longitude': lon_f, 'hour': hr_f})
    os.makedirs('./working', exist_ok=True)
    sub.to_csv('./working/submission.csv', index=False)
    print('wrote ./working/submission.csv', sub.shape, flush=True)
    print(sub.head().to_string(index=False))

if __name__ == '__main__':
    main()

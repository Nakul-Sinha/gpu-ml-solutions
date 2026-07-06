import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
try:
    torch.cuda.manual_seed_all(SEED)
except Exception:
    pass
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG = 256
EPOCHS = int(os.environ.get('EPOCHS', 70))
NFOLD = 5

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
ztr = np.load(os.path.join(DS, 'train', 'images.npz')); RGB, TH = ztr['rgb'], ztr['thermal']
zte = np.load(os.path.join(DS, 'test', 'images.npz')); RGBt, THt = zte['rgb'], zte['thermal']
BOX = tr[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(np.float32)
IDX = tr['array_index'].values
IDXt = te['array_index'].values
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def get6(rgb_arr, th_arr, ai):
    return np.concatenate([rgb_arr[ai].astype(np.float32) / 255., th_arr[ai].astype(np.float32) / 255.], 2)

def norm_t(t):
    for c in range(3):
        t[c] = (t[c] - MEAN[c]) / STD[c]; t[3 + c] = (t[3 + c] - 0.5) / 0.25
    return t

class WDS(Dataset):
    def __init__(self, idxs, boxes, train=True):
        self.idxs = idxs; self.boxes = boxes; self.train = train
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        ai = self.idxs[i]; x0, y0, x1, y1 = self.boxes[i].copy()
        img = get6(RGB, TH, ai)
        if self.train:
            if np.random.rand() < 0.5: img = img[:, ::-1, :]; x0, x1 = 1 - x1, 1 - x0
            if np.random.rand() < 0.5: img = img[::-1, :, :]; y0, y1 = 1 - y1, 1 - y0
            for _ in range(np.random.randint(0, 4)):
                img = np.rot90(img, 1, axes=(0, 1)); x0, y0, x1, y1 = y0, 1 - x1, y1, 1 - x0
            img = np.ascontiguousarray(img)
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(IMG, IMG), mode='bilinear', align_corners=False)[0]
        if self.train and np.random.rand() < 0.7:
            t[:3] = torch.clamp(t[:3] * (0.8 + 0.4 * np.random.rand()) + (np.random.rand() - 0.5) * 0.1, 0, 1)
        t = norm_t(t)
        box = torch.clamp(torch.tensor([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], dtype=torch.float32), 0, 1)
        return t, box

def make_backbone():
    m = torchvision.models.resnet34(weights='IMAGENET1K_V1'); f = 512
    w = m.conv1.weight.data
    nc = nn.Conv2d(6, w.shape[0], 7, 2, 3, bias=False)
    with torch.no_grad():
        nc.weight[:, :3] = w; nc.weight[:, 3:6] = w * 0.5
    m.conv1 = nc; m.fc = nn.Identity()
    return m, f

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb, f = make_backbone()
        self.head = nn.Sequential(nn.Linear(f, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4))
    def forward(self, x): return torch.sigmoid(self.head(self.bb(x)))

def giou_loss(p, t):
    px0, py0, px1, py1 = p.unbind(1); tx0, ty0, tx1, ty1 = t.unbind(1)
    px1 = torch.max(px1, px0 + 1e-4); py1 = torch.max(py1, py0 + 1e-4)
    ix0 = torch.max(px0, tx0); iy0 = torch.max(py0, ty0); ix1 = torch.min(px1, tx1); iy1 = torch.min(py1, ty1)
    inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
    ap = (px1 - px0) * (py1 - py0); at = (tx1 - tx0) * (ty1 - ty0); union = ap + at - inter + 1e-7
    iou = inter / union
    cx0 = torch.min(px0, tx0); cy0 = torch.min(py0, ty0); cx1 = torch.max(px1, tx1); cy1 = torch.max(py1, ty1)
    carea = (cx1 - cx0) * (cy1 - cy0) + 1e-7
    return (1 - (iou - (carea - union) / carea)).mean()

def metric_rows(p, t):
    ix = np.maximum(0, np.minimum(p[:, 2], t[:, 2]) - np.maximum(p[:, 0], t[:, 0]))
    iy = np.maximum(0, np.minimum(p[:, 3], t[:, 3]) - np.maximum(p[:, 1], t[:, 1])); inter = ix * iy
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]); at = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    iou = inter / (ap + at - inter + 1e-9); cmae = np.mean(np.abs(p - t), 1)
    return 100 * (0.65 * (1 - iou) + 0.35 * cmae)

@torch.no_grad()
def predict_tta(model, rgb_arr, th_arr, idxs):
    model.eval(); out = []
    for ai in idxs:
        img0 = get6(rgb_arr, th_arr, ai); acc = []
        for tf in ['id', 'h', 'v', 'r180']:
            img = img0.copy()
            if tf == 'h': img = img[:, ::-1, :]
            elif tf == 'v': img = img[::-1, :, :]
            elif tf == 'r180': img = img[::-1, ::-1, :]
            img = np.ascontiguousarray(img)
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(IMG, IMG), mode='bilinear', align_corners=False)[0]
            t = norm_t(t)
            p = model(t.unsqueeze(0).to(DEV)).cpu().numpy()[0]; x0, y0, x1, y1 = p
            if tf == 'h': x0, x1 = 1 - x1, 1 - x0
            elif tf == 'v': y0, y1 = 1 - y1, 1 - y0
            elif tf == 'r180': x0, y0, x1, y1 = 1 - x1, 1 - y1, 1 - x0, 1 - y0
            acc.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
        out.append(np.mean(acc, 0))
    return np.array(out)

def train_fold(tri, epochs):
    dl = DataLoader(WDS(IDX[tri], BOX[tri], True), batch_size=16, shuffle=True, num_workers=0, drop_last=len(tri) > 16)
    model = Net().to(DEV)
    opt = torch.optim.AdamW([{'params': model.bb.parameters(), 'lr': 1.5e-4},
                             {'params': model.head.parameters(), 'lr': 1.2e-3}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV); p = model(x)
            loss = 0.7 * giou_loss(p, y) + 0.3 * F.l1_loss(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model

def main():
    kf = KFold(NFOLD, shuffle=True, random_state=SEED)
    cv = []; test_preds = []
    for fi, (tri, vai) in enumerate(kf.split(IDX)):
        model = train_fold(tri, EPOCHS)
        pv = predict_tta(model, RGB, TH, IDX[vai])
        m = metric_rows(pv, BOX[vai]).mean(); cv.append(m)
        pt = predict_tta(model, RGBt, THt, IDXt); test_preds.append(pt)
        print(f'fold{fi} val_metric={m:.3f}', flush=True)
    print(f'CV metric = {np.mean(cv):.3f} +- {np.std(cv):.3f} (lower better)', flush=True)
    pred = np.mean(test_preds, 0)

    pred = np.clip(pred, 0, 1)
    for j in range(len(pred)):
        if pred[j, 2] - pred[j, 0] < 1e-3: pred[j, 0], pred[j, 2] = max(0, pred[j, 0] - 5e-3), min(1, pred[j, 2] + 5e-3)
        if pred[j, 3] - pred[j, 1] < 1e-3: pred[j, 1], pred[j, 3] = max(0, pred[j, 1] - 5e-3), min(1, pred[j, 3] + 5e-3)
    sub = pd.DataFrame({'id': te['id'].values, 'x_min': pred[:, 0], 'y_min': pred[:, 1],
                        'x_max': pred[:, 2], 'y_max': pred[:, 3]})
    os.makedirs('./working', exist_ok=True)
    sub.to_csv('./working/submission.csv', index=False)
    print('wrote ./working/submission.csv', sub.shape, flush=True)

if __name__ == '__main__':
    main()

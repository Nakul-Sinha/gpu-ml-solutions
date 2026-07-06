"""Wildlife bbox localization: 6-channel (RGB+thermal) early-fusion CNN with pretrained
backbone, aerial augmentation (flips/rot90/affine + box transforms), GIoU+L1 loss.
5-fold CV reporting the exact competition metric. Local (RTX 4050) friendly.
"""
import os, sys, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(HERE, 'dataset')
IMG = 256
BACKBONE = os.environ.get('BB', 'resnet34')
EPOCHS = int(os.environ.get('EPOCHS', 80))

tr = pd.read_csv(os.path.join(DS, 'train.csv'))
z = np.load(os.path.join(DS, 'train', 'images.npz'))
RGB, TH = z['rgb'], z['thermal']  # (N,300,300,3) uint8
BOX = tr[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(np.float32)
IDX = tr['array_index'].values

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

def to6(ai):
    rgb = RGB[ai].astype(np.float32) / 255.0
    th = TH[ai].astype(np.float32) / 255.0
    return rgb, th  # each (300,300,3)

class DS6(Dataset):
    def __init__(self, idxs, boxes, train=True):
        self.idxs = idxs; self.boxes = boxes; self.train = train
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        ai = self.idxs[i]; box = self.boxes[i].copy()
        rgb, th = to6(ai)
        img = np.concatenate([rgb, th], axis=2)  # (300,300,6)
        x0, y0, x1, y1 = box
        if self.train:
            # hflip
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]; x0, x1 = 1 - x1, 1 - x0
            # vflip
            if np.random.rand() < 0.5:
                img = img[::-1, :, :]; y0, y1 = 1 - y1, 1 - y0
            # rot90 k times
            k = np.random.randint(0, 4)
            for _ in range(k):
                img = np.rot90(img, 1, axes=(0, 1))
                # (x,y) -> (y, 1-x) for 90deg CCW on normalized coords
                nx0, ny0, nx1, ny1 = y0, 1 - x1, y1, 1 - x0
                x0, y0, x1, y1 = nx0, ny0, nx1, ny1
            img = np.ascontiguousarray(img)
        # resize to IMG
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(IMG, IMG), mode='bilinear', align_corners=False)[0]
        # color jitter on rgb (first 3) only
        if self.train and np.random.rand() < 0.7:
            t[:3] = torch.clamp(t[:3] * (0.8 + 0.4 * np.random.rand()) + (np.random.rand() - 0.5) * 0.1, 0, 1)
        # normalize: rgb with imagenet, thermal with 0.5
        for c in range(3):
            t[c] = (t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            t[3 + c] = (t[3 + c] - 0.5) / 0.25
        box = torch.tensor([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], dtype=torch.float32)
        box = torch.clamp(box, 0, 1)
        return t, box

def make_backbone(name):
    if name == 'resnet18':
        m = torchvision.models.resnet18(weights='IMAGENET1K_V1'); feat = 512
    elif name == 'resnet34':
        m = torchvision.models.resnet34(weights='IMAGENET1K_V1'); feat = 512
    elif name == 'resnet50':
        m = torchvision.models.resnet50(weights='IMAGENET1K_V2'); feat = 2048
    w = m.conv1.weight.data  # (64,3,7,7)
    newc = nn.Conv2d(6, w.shape[0], 7, 2, 3, bias=False)
    with torch.no_grad():
        newc.weight[:, :3] = w
        newc.weight[:, 3:6] = w * 0.5  # init thermal like rgb
    m.conv1 = newc
    m.fc = nn.Identity()
    return m, feat

class Net(nn.Module):
    def __init__(self, name='resnet34'):
        super().__init__()
        self.bb, feat = make_backbone(name)
        self.head = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4))
    def forward(self, x):
        f = self.bb(x)
        return torch.sigmoid(self.head(f))  # xyxy in [0,1]

def giou_loss(p, t):
    # p,t: (B,4) xyxy
    px0, py0, px1, py1 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    tx0, ty0, tx1, ty1 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    px1 = torch.max(px1, px0 + 1e-4); py1 = torch.max(py1, py0 + 1e-4)
    ix0 = torch.max(px0, tx0); iy0 = torch.max(py0, ty0)
    ix1 = torch.min(px1, tx1); iy1 = torch.min(py1, ty1)
    iw = (ix1 - ix0).clamp(min=0); ih = (iy1 - iy0).clamp(min=0)
    inter = iw * ih
    ap = (px1 - px0) * (py1 - py0); at = (tx1 - tx0) * (ty1 - ty0)
    union = ap + at - inter + 1e-7
    iou = inter / union
    cx0 = torch.min(px0, tx0); cy0 = torch.min(py0, ty0)
    cx1 = torch.max(px1, tx1); cy1 = torch.max(py1, ty1)
    carea = (cx1 - cx0) * (cy1 - cy0) + 1e-7
    giou = iou - (carea - union) / carea
    return (1 - giou).mean(), iou.mean()

def metric_rows(p, t):
    # exact competition metric per row -> mean
    inter_x = np.maximum(0, np.minimum(p[:, 2], t[:, 2]) - np.maximum(p[:, 0], t[:, 0]))
    inter_y = np.maximum(0, np.minimum(p[:, 3], t[:, 3]) - np.maximum(p[:, 1], t[:, 1]))
    inter = inter_x * inter_y
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    at = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    iou = inter / (ap + at - inter + 1e-9)
    cmae = np.mean(np.abs(p - t), axis=1)
    return 100 * (0.65 * (1 - iou) + 0.35 * cmae)

@torch.no_grad()
def predict_tta(model, idxs):
    model.eval()
    preds = []
    for ai in idxs:
        rgb, th = to6(ai)
        img0 = np.concatenate([rgb, th], axis=2)
        variants = []
        # identity + hflip + vflip + rot180 (each with inverse box map)
        tfs = ['id', 'h', 'v', 'r180']
        outs = []
        for tf in tfs:
            img = img0.copy()
            if tf == 'h': img = img[:, ::-1, :]
            elif tf == 'v': img = img[::-1, :, :]
            elif tf == 'r180': img = img[::-1, ::-1, :]
            img = np.ascontiguousarray(img)
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(IMG, IMG), mode='bilinear', align_corners=False)[0]
            for c in range(3):
                t[c] = (t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
                t[3 + c] = (t[3 + c] - 0.5) / 0.25
            p = model(t.unsqueeze(0).to(DEV)).cpu().numpy()[0]
            x0, y0, x1, y1 = p
            if tf == 'h': x0, x1 = 1 - x1, 1 - x0
            elif tf == 'v': y0, y1 = 1 - y1, 1 - y0
            elif tf == 'r180': x0, y0, x1, y1 = 1 - x1, 1 - y1, 1 - x0, 1 - y0
            outs.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
        preds.append(np.mean(outs, axis=0))
    return np.array(preds)

def train_fold(tri, vai, epochs=EPOCHS, verbose=False):
    dl = DataLoader(DS6(IDX[tri], BOX[tri], True), batch_size=16, shuffle=True, num_workers=0, drop_last=False)
    model = Net(BACKBONE).to(DEV)
    opt = torch.optim.AdamW([
        {'params': model.bb.parameters(), 'lr': 1.5e-4},
        {'params': model.head.parameters(), 'lr': 1.2e-3}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            gl, _ = giou_loss(model(x), y)
            l1 = F.l1_loss(model(x), y)
            loss = 0.7 * gl + 0.3 * l1
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    pv = predict_tta(model, IDX[vai])
    return model, pv

if __name__ == '__main__':
    from sklearn.model_selection import KFold
    print(f'backbone={BACKBONE} img={IMG} epochs={EPOCHS} dev={DEV}')
    kf = KFold(5, shuffle=True, random_state=SEED)
    all_m = []
    for fi, (tri, vai) in enumerate(kf.split(IDX)):
        _, pv = train_fold(tri, vai)
        m = metric_rows(pv, BOX[vai])
        all_m.append(m.mean())
        print(f'fold{fi} metric={m.mean():.3f}')
    print(f'CV metric = {np.mean(all_m):.3f} +- {np.std(all_m):.3f}  (lower better)')

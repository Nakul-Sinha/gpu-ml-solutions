"""Wildlife v2: 6-channel RGB+thermal, pretrained backbone, crop-around-box + flip/rot90
augmentation, GIoU+L1 loss. CV to measure the exact metric; ensemble+TTA to predict test.
Config via env: BB, IMG, EPOCHS, MODE (cv|final).
"""
import os, sys, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
def env(k,d): return type(d)(os.environ.get(k,d))
SEED=env('SEED',42); torch.manual_seed(SEED); np.random.seed(SEED)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def find_ds():
    for c in [os.path.join(HERE,'dataset','public'),os.path.join(HERE,'dataset'),'./dataset/public','./dataset']:
        if os.path.exists(os.path.join(c,'train.csv')): return c
    raise FileNotFoundError('dataset')
DS=find_ds()
IMG=env('IMG',288); BB=os.environ.get('BB','resnet34'); EPOCHS=env('EPOCHS',110); MODE=os.environ.get('MODE','cv')
tr=pd.read_csv(os.path.join(DS,'train.csv'))
z=np.load(os.path.join(DS,'train','images.npz')); RGB,TH=z['rgb'],z['thermal']
BOX=tr[['x_min','y_min','x_max','y_max']].values.astype(np.float32); IDX=tr['array_index'].values
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)

def get6(ai):
    return np.concatenate([RGB[ai].astype(np.float32)/255., TH[ai].astype(np.float32)/255.],2)  # (300,300,6)

def norm_t(t):
    for c in range(3):
        t[c]=(t[c]-MEAN[c])/STD[c]; t[3+c]=(t[3+c]-0.5)/0.25
    return t

class WDS(Dataset):
    def __init__(self, idxs, boxes, train=True): self.idxs=idxs; self.boxes=boxes; self.train=train
    def __len__(self): return len(self.idxs)
    def __getitem__(self,i):
        ai=self.idxs[i]; x0,y0,x1,y1=self.boxes[i].copy(); img=get6(ai)
        H,W=img.shape[:2]
        if self.train:
            # crop-around-box: choose a window that fully contains the box (scale/pan aug)
            bw,bh=x1-x0,y1-y0
            cw=min(1.0, max(bw*1.05, (bw+ (1-bw)*np.random.uniform(0.15,1.0))))
            ch=min(1.0, max(bh*1.05, (bh+ (1-bh)*np.random.uniform(0.15,1.0))))
            lx=np.random.uniform(max(0,x1-cw), min(x0, 1-cw)) if 1-cw>0 else 0.0
            ly=np.random.uniform(max(0,y1-ch), min(y0, 1-ch)) if 1-ch>0 else 0.0
            lx=float(np.clip(lx,0,max(0,1-cw))); ly=float(np.clip(ly,0,max(0,1-ch)))
            px0,py0,px1,py1=int(lx*W),int(ly*H),int((lx+cw)*W),int((ly+ch)*H)
            px1=max(px1,px0+8); py1=max(py1,py0+8)
            img=img[py0:py1, px0:px1, :]
            # remap box to crop
            nw,nh=(px1-px0)/W,(py1-py0)/H
            x0=(x0-lx)/nw; x1=(x1-lx)/nw; y0=(y0-ly)/nh; y1=(y1-ly)/nh
            if np.random.rand()<0.5: img=img[:,::-1,:]; x0,x1=1-x1,1-x0
            if np.random.rand()<0.5: img=img[::-1,:,:]; y0,y1=1-y1,1-y0
            k=np.random.randint(0,4)
            for _ in range(k):
                img=np.rot90(img,1,axes=(0,1)); x0,y0,x1,y1=y0,1-x1,y1,1-x0
            img=np.ascontiguousarray(img)
        t=torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
        t=F.interpolate(t,size=(IMG,IMG),mode='bilinear',align_corners=False)[0]
        if self.train and np.random.rand()<0.7:
            t[:3]=torch.clamp(t[:3]*(0.75+0.5*np.random.rand())+(np.random.rand()-0.5)*0.15,0,1)
            t[3:]=torch.clamp(t[3:]*(0.8+0.4*np.random.rand()),0,1)
        t=norm_t(t)
        box=torch.clamp(torch.tensor([min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)],dtype=torch.float32),0,1)
        return t, box

def backbone(name):
    if name=='resnet18': m=torchvision.models.resnet18(weights='IMAGENET1K_V1'); f=512
    elif name=='resnet34': m=torchvision.models.resnet34(weights='IMAGENET1K_V1'); f=512
    elif name=='resnet50': m=torchvision.models.resnet50(weights='IMAGENET1K_V2'); f=2048
    w=m.conv1.weight.data; nc=nn.Conv2d(6,w.shape[0],7,2,3,bias=False)
    with torch.no_grad(): nc.weight[:,:3]=w; nc.weight[:,3:6]=w*0.5
    m.conv1=nc; m.fc=nn.Identity(); return m,f
class Net(nn.Module):
    def __init__(self,name):
        super().__init__(); self.bb,f=backbone(name)
        self.head=nn.Sequential(nn.Linear(f,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,4))
    def forward(self,x): return torch.sigmoid(self.head(self.bb(x)))

def giou_loss(p,t):
    px0,py0,px1,py1=p.unbind(1); tx0,ty0,tx1,ty1=t.unbind(1)
    px1=torch.max(px1,px0+1e-4); py1=torch.max(py1,py0+1e-4)
    ix0=torch.max(px0,tx0);iy0=torch.max(py0,ty0);ix1=torch.min(px1,tx1);iy1=torch.min(py1,ty1)
    inter=(ix1-ix0).clamp(min=0)*(iy1-iy0).clamp(min=0)
    ap=(px1-px0)*(py1-py0);at=(tx1-tx0)*(ty1-ty0);union=ap+at-inter+1e-7;iou=inter/union
    cx0=torch.min(px0,tx0);cy0=torch.min(py0,ty0);cx1=torch.max(px1,tx1);cy1=torch.max(py1,ty1)
    carea=(cx1-cx0)*(cy1-cy0)+1e-7
    return (1-(iou-(carea-union)/carea)).mean()

def metric_rows(p,t):
    ix=np.maximum(0,np.minimum(p[:,2],t[:,2])-np.maximum(p[:,0],t[:,0]))
    iy=np.maximum(0,np.minimum(p[:,3],t[:,3])-np.maximum(p[:,1],t[:,1]));inter=ix*iy
    ap=(p[:,2]-p[:,0])*(p[:,3]-p[:,1]);at=(t[:,2]-t[:,0])*(t[:,3]-t[:,1])
    iou=inter/(ap+at-inter+1e-9);cmae=np.mean(np.abs(p-t),1)
    return 100*(0.65*(1-iou)+0.35*cmae)

@torch.no_grad()
def predict_tta(model, arrs):
    model.eval(); out=[]
    for img0 in arrs:
        acc=[]
        for tf in ['id','h','v','r180']:
            img=img0.copy()
            if tf=='h': img=img[:,::-1,:]
            elif tf=='v': img=img[::-1,:,:]
            elif tf=='r180': img=img[::-1,::-1,:]
            img=np.ascontiguousarray(img)
            t=torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
            t=F.interpolate(t,size=(IMG,IMG),mode='bilinear',align_corners=False)[0]
            t=norm_t(t)
            p=model(t.unsqueeze(0).to(DEV)).cpu().numpy()[0]; x0,y0,x1,y1=p
            if tf=='h': x0,x1=1-x1,1-x0
            elif tf=='v': y0,y1=1-y1,1-y0
            elif tf=='r180': x0,y0,x1,y1=1-x1,1-y1,1-x0,1-y0
            acc.append([min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)])
        out.append(np.mean(acc,0))
    return np.array(out)

def train_one(tri, epochs=EPOCHS):
    dl=DataLoader(WDS(IDX[tri],BOX[tri],True),batch_size=16,shuffle=True,num_workers=0,drop_last=len(tri)>16)
    model=Net(BB).to(DEV)
    opt=torch.optim.AdamW([{'params':model.bb.parameters(),'lr':1.5e-4},
                           {'params':model.head.parameters(),'lr':1.2e-3}],weight_decay=1e-2)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    for ep in range(epochs):
        model.train()
        for x,y in dl:
            x,y=x.to(DEV),y.to(DEV); p=model(x)
            loss=0.7*giou_loss(p,y)+0.3*F.l1_loss(p,y)
            opt.zero_grad();loss.backward();opt.step()
        sched.step()
    return model

if __name__=='__main__':
    from sklearn.model_selection import KFold
    print(f'BB={BB} IMG={IMG} EPOCHS={EPOCHS} MODE={MODE} dev={DEV}',flush=True)
    if MODE=='cv':
        kf=KFold(5,shuffle=True,random_state=SEED); ms=[]
        for fi,(tri,vai) in enumerate(kf.split(IDX)):
            model=train_one(tri)
            arrs=[get6(ai) for ai in IDX[vai]]
            pv=predict_tta(model,arrs); m=metric_rows(pv,BOX[vai]).mean(); ms.append(m)
            print(f'fold{fi} metric={m:.3f}',flush=True)
        print(f'CV metric={np.mean(ms):.3f} +- {np.std(ms):.3f}',flush=True)

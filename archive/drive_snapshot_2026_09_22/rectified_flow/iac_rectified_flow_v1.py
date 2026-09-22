from __future__ import annotations

import argparse, json, math, random, time, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Reuse ONLY the already-validated metric code from old IAC-B.
# No old bridge/model/train code is imported.
IACB_ROOT = Path('/content/drive/MyDrive/ToothFairy/ToothFairy3/iacb')
if IACB_ROOT.exists():
    sys.path.insert(0, str(IACB_ROOT))
from iacb.common import prior_side_masks, side_metrics


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ======================================================================================
# DATA
# ======================================================================================
class Case:
    def __init__(self, cache: Path, case: str):
        self.case = case
        self.meta = json.loads((cache / f'{case}_meta.json').read_text())
        self.img = np.load(cache / f'{case}_img.npy', mmap_mode='r')
        self.p = np.load(cache / f'{case}_p.npy', mmap_mode='r')
        self.x0 = np.load(cache / f'{case}_x0.npy', mmap_mode='r')
        self.x1 = np.load(cache / f'{case}_x1.npy', mmap_mode='r')
        self.gt = np.load(cache / f'{case}_gt.npy', mmap_mode='r')
        idx = np.load(cache / f'{case}_idx.npz')
        self.fg, self.dis = idx['fg'], idx['dis']
        self.shape = self.img.shape
        self.clip = float(self.meta['clip_mm'])


def sample_patch(c: Case, patch, mode: str, rng: np.random.Generator):
    if mode == 'dis' and len(c.dis):
        ctr = c.dis[rng.integers(len(c.dis))]
    elif mode == 'fg' and len(c.fg):
        ctr = c.fg[rng.integers(len(c.fg))]
    else:
        ctr = np.array([rng.integers(s) for s in c.shape])

    sl, pad = [], []
    for k in range(3):
        lo = int(ctr[k]) - patch[k] // 2 + int(rng.integers(-patch[k]//4, patch[k]//4 + 1))
        lo = min(max(lo, -patch[k]//4), c.shape[k] - 1)
        a, b = max(lo, 0), min(lo + patch[k], c.shape[k])
        sl.append(slice(a, b)); pad.append((a-lo, lo+patch[k]-b))
    sl = tuple(sl)

    img = np.pad(np.asarray(c.img[sl], np.float32), pad)
    p = np.pad(np.asarray(c.p[(slice(None),)+sl], np.float32) / 255.0,
               [(0,0)] + pad, constant_values=0.0)
    x0 = np.pad(np.asarray(c.x0[(slice(None),)+sl], np.float32) / c.clip,
                [(0,0)] + pad, constant_values=1.0)
    x1 = np.pad(np.asarray(c.x1[(slice(None),)+sl], np.float32) / c.clip,
                [(0,0)] + pad, constant_values=1.0)
    return img[None], p, x0, x1


class CaseBalancedDataset(Dataset):
    def __init__(self, cache, cases, patch, epoch, seed, training=True):
        self.cache, self.cases = Path(cache), list(cases)
        self.patch, self.epoch, self.seed = tuple(patch), int(epoch), int(seed)
        self.training = bool(training)

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx):
        name = self.cases[idx]
        rng = np.random.default_rng(self.seed + self.epoch*1_000_003 + idx*10_007)
        c = Case(self.cache, name)
        aa = sample_patch(c, self.patch, 'dis', rng)
        bb = sample_patch(c, self.patch, 'fg', rng)
        imgs = []
        ps, x0s, x1s = [], [], []
        for img, p, x0, x1 in (aa, bb):
            if self.training:
                img = img * rng.uniform(0.9,1.1) + rng.uniform(-0.1,0.1)
            imgs.append(img); ps.append(p); x0s.append(x0); x1s.append(x1)
        return (torch.from_numpy(np.stack(imgs).astype(np.float32)),
                torch.from_numpy(np.stack(ps).astype(np.float32)),
                torch.from_numpy(np.stack(x0s).astype(np.float32)),
                torch.from_numpy(np.stack(x1s).astype(np.float32)),
                name)


def make_split(cache: Path, eval_fold=0, val_n=32, seed=42):
    metas = [json.loads(p.read_text()) for p in sorted(cache.glob('*_meta.json'))]
    ev = sorted(m['case'] for m in metas if m['fold'] == eval_fold)
    ext = sorted(m['case'] for m in metas if m['fold'] == -1)
    pf = sorted(m['case'] for m in metas if m['fold'] is not None and m['fold'] >= 0 and m['fold'] != eval_fold)
    assert len(ev) == 97, f'expected 97 eval, got {len(ev)}'
    assert len(ext) == 52, f'expected 52 external, got {len(ext)}'
    assert len(pf) == 383, f'expected 383 non-eval P/F, got {len(pf)}'
    r = random.Random(seed); q = list(pf); r.shuffle(q)
    va = sorted(q[:val_n]); tr = sorted(q[val_n:] + ext)
    assert len(tr)==403 and len(va)==32 and len(ev)==97
    assert not(set(tr)&set(va) or set(tr)&set(ev) or set(va)&set(ev))
    return tr, va, ev


# ======================================================================================
# TRUE RECTIFIED-FLOW VELOCITY NETWORK
# ======================================================================================
def _gn(c): return nn.GroupNorm(min(8, c), c)


class FiLMRes(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n1 = _gn(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.n2 = _gn(cout)
        self.film = nn.Linear(tdim, 2*cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, x, te):
        h = F.silu(self.n1(self.c1(x)))
        g,b = self.film(te).chunk(2,1)
        h = h * (1 + g[...,None,None,None]) + b[...,None,None,None]
        h = self.n2(self.c2(F.silu(h)))
        return F.silu(h + self.skip(x))


class VelocityUNet(nn.Module):
    """v_theta(x_t | image, x0, prior probs, t).  Output is dX/dt, 2 SDF channels."""
    def __init__(self, widths=(24,48,96,160), tdim=64):
        super().__init__()
        self.widths, self.levels, self.tdim = tuple(widths), len(widths), tdim
        # image(1) + xt(2) + x0(2) + prior probabilities(2) + uncertainty(1) = 8
        self.stem = nn.Conv3d(8, widths[0], 3, padding=1)
        self.tmlp = nn.Sequential(nn.Linear(tdim,tdim), nn.SiLU(), nn.Linear(tdim,tdim))
        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        for i,w in enumerate(widths):
            cin = widths[max(i-1,0)] if i else widths[0]
            self.enc.append(FiLMRes(cin,w,tdim))
            if i < len(widths)-1:
                self.down.append(nn.Conv3d(w,w,3,stride=2,padding=1))
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for i in range(len(widths)-1,0,-1):
            self.up.append(nn.ConvTranspose3d(widths[i], widths[i-1], 2, stride=2))
            self.dec.append(FiLMRes(2*widths[i-1], widths[i-1], tdim))
        self.head = nn.Conv3d(widths[0], 2, 1)
        # exact identity at initialization: v=0 => x stays x0
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def temb(self,t):
        half = self.tdim//2
        f = torch.exp(-math.log(1000.0)*torch.arange(half,device=t.device)/half)
        a = t[:,None]*1000.0*f[None]
        return self.tmlp(torch.cat([a.sin(),a.cos()],1))

    def forward(self, img, xt, x0, p, t):
        fg = p.sum(1,keepdim=True).clamp(0,1)
        unc = 1.0 - (2.0*fg - 1.0).abs()      # high near foreground decision boundary
        h = self.stem(torch.cat([img,xt,x0,p,unc],1))
        te = self.temb(t)
        skips=[]
        for i,blk in enumerate(self.enc):
            h = blk(h,te)
            if i < self.levels-1:
                skips.append(h); h = self.down[i](h)
        for up,blk in zip(self.up,self.dec):
            h = up(h); s = skips.pop()
            h = blk(torch.cat([h,s],1),te)
        return self.head(h)


# ======================================================================================
# LOSS: velocity + endpoint geometry.  No noise, no latent AE, no clDice-thickening term.
# ======================================================================================
def losses(v, xt, x0, x1, t, clip_mm, tau_mm=0.30):
    target_v = x1 - x0
    x1hat = (xt + (1.0 - t[:,None,None,None,None]) * v).clamp(-1,1)

    x1mm = x1 * clip_mm
    diffmm = (x1-x0).abs() * clip_mm
    w = 1.0 + 4.0*torch.exp(-x1mm.abs()/1.0) + 2.0*(diffmm/1.0).clamp(0,1)

    lf_map = F.smooth_l1_loss(v, target_v, reduction='none', beta=0.10)
    l_flow = (lf_map*w).sum()/w.sum()
    l_sdf = ((x1hat-x1).abs()*w).sum()/w.sum()

    logits = -(x1hat*clip_mm)/tau_mm
    prob = torch.sigmoid(logits)
    gt = (x1 < 0).float()
    dims=(0,2,3,4)
    inter=(prob*gt).sum(dims)
    l_dice = 1.0 - ((2*inter+1)/(prob.sum(dims)+gt.sum(dims)+1)).mean()
    l_bce = F.binary_cross_entropy_with_logits(logits, gt)

    # discourage gratuitous motion where prior is already geometrically close to GT
    stay_w = torch.exp(-diffmm/0.50)
    l_stay = (stay_w*v.abs()).sum()/(stay_w.sum()+1e-6)
    l_overlap = (prob[:,0]*prob[:,1]).mean()

    total = l_flow + 0.50*l_sdf + 0.75*l_dice + 0.10*l_bce + 0.05*l_stay + 0.05*l_overlap
    parts = dict(flow=float(l_flow.detach()), sdf=float(l_sdf.detach()), dice=float(l_dice.detach()),
                 bce=float(l_bce.detach()), stay=float(l_stay.detach()), overlap=float(l_overlap.detach()))
    return total, parts


# ======================================================================================
# FULL-VOLUME INFERENCE
# ======================================================================================
def window_starts(size, patch, overlap):
    if size <= patch: return [0]
    step=max(1,int(patch*(1-overlap)))
    st=list(range(0,size-patch+1,step))
    if st[-1] != size-patch: st.append(size-patch)
    return st


def gaussian_weight(patch, device):
    ws=[]
    for s in patch:
        x=torch.arange(s,device=device,dtype=torch.float32)-(s-1)/2
        ws.append(torch.exp(-0.5*(x/(s/8))**2))
    w=ws[0][:,None,None]*ws[1][None,:,None]*ws[2][None,None,:]
    return (w/w.max()).clamp_min(1e-3)


def ceil_mult(v,m): return int(math.ceil(v/m)*m)


@torch.no_grad()
def predict_velocity(model, img, x, x0, p, t_scalar, patch, device, overlap=0.5):
    # all tensors have batch=1; spatial ROI can be arbitrary size
    _,_,Z,Y,X=x.shape
    pz,py,px=[min(a,b) for a,b in zip(patch,(Z,Y,X))]
    mult=2**(model.levels-1)
    need=(ceil_mult(pz,mult),ceil_mult(py,mult),ceil_mult(px,mult))
    w=gaussian_weight(need,device)
    acc=torch.zeros_like(x,dtype=torch.float32)
    norm=torch.zeros((1,1,Z,Y,X),device=device)
    t=torch.full((1,),float(t_scalar),device=device)
    for z in window_starts(Z,pz,overlap):
      for y in window_starts(Y,py,overlap):
       for xx in window_starts(X,px,overlap):
        sl=(slice(z,z+pz),slice(y,y+py),slice(xx,xx+px))
        im=img[(slice(None),slice(None))+sl]
        xt=x[(slice(None),slice(None))+sl]
        x0s=x0[(slice(None),slice(None))+sl]
        ps=p[(slice(None),slice(None))+sl]
        pad=[0,need[2]-px,0,need[1]-py,0,need[0]-pz]
        im=F.pad(im,pad,value=0.0); xt=F.pad(xt,pad,value=1.0)
        x0s=F.pad(x0s,pad,value=1.0); ps=F.pad(ps,pad,value=0.0)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            vv=model(im,xt,x0s,ps,t).float()[...,:pz,:py,:px]
        ww=w[:pz,:py,:px]
        acc[(slice(None),slice(None))+sl] += vv*ww
        norm[(slice(None),slice(None))+sl] += ww
    return acc/norm


@torch.no_grad()
def integrate_heun(model, img, x0, p, steps, patch, device):
    """Deterministic Heun integration from t=0 to t=1.
    `steps=4` means 4 ODE steps / 8 velocity evaluations.
    """
    x = x0.clone().float()
    dt = 1.0 / float(steps)
    for k in range(steps):
        t0 = k / float(steps)
        t1 = (k + 1) / float(steps)
        v0 = predict_velocity(model, img, x, x0, p, t0, patch, device)
        xe = (x + dt * v0).clamp(-1, 1)
        v1 = predict_velocity(model, img, xe, x0, p, t1, patch, device)
        x = (x + 0.5 * dt * (v0 + v1)).clamp(-1, 1)
    return x


def decode_sdf(xn):
    # xn: (2,Z,Y,X)
    a=np.argmin(xn,axis=0)
    m=np.min(xn,axis=0)
    lab=np.zeros(m.shape,np.uint8)
    lab[m<0]=a[m<0].astype(np.uint8)+1
    return lab


@torch.no_grad()
def eval_full(model, cache: Path, cases, patch, device, steps, out_csv: Path|None=None, with_cldice=False):
    rows=[]
    done=set()
    if out_csv is not None and out_csv.exists():
        old=pd.read_csv(out_csv)
        if len(old):
            rows=old.to_dict('records'); done=set(old['case'].unique())
    for ii,case in enumerate(cases):
        if case in done: continue
        t0=time.time(); c=Case(cache,case)
        sp=tuple(c.meta['spacing']); clip=c.clip
        p_np=np.asarray(c.p,np.float32)/255.0
        x0_np=np.asarray(c.x0,np.float32)/clip
        gt=np.asarray(c.gt,np.uint8)
        prior2=prior_side_masks(p_np)

        roi=c.meta.get('roi_inference')
        if roi is None: continue
        sl=tuple(slice(int(a),int(b)) for a,b in roi)
        img=torch.from_numpy(np.asarray(c.img[sl],np.float32))[None,None].to(device)
        pp=torch.from_numpy(np.asarray(p_np[(slice(None),)+sl],np.float32))[None].to(device)
        xs=torch.from_numpy(np.asarray(x0_np[(slice(None),)+sl],np.float32))[None].to(device)
        xf=integrate_heun(model,img,xs,pp,steps,patch,device)[0].cpu().numpy()
        full=np.asarray(x0_np,np.float32).copy()
        full[(slice(None),)+sl]=xf
        pred=decode_sdf(full)

        for si,side in enumerate(('L','R')):
            g=(gt==(si+1)); pr=prior2[si]; fl=(pred==(si+1))
            rp=side_metrics(pr,g,sp,with_cldice=with_cldice)
            rf=side_metrics(fl,g,sp,with_cldice=with_cldice)
            rows.append(dict(case=case,side=side,method='prior',**rp))
            rows.append(dict(case=case,side=side,method='flow',**rf))
        if out_csv is not None:
            pd.DataFrame(rows).to_csv(out_csv,index=False)
        print(f'[eval {ii+1:03d}/{len(cases):03d}] {case} {time.time()-t0:.1f}s',flush=True)

    df=pd.DataFrame(rows)
    def summarize(m):
        d=df[df.method==m]
        return dict(dice_mean=float(d.dice.mean()), dice_median=float(d.dice.median()),
                    hd95_mean=float(d.hd95.mean()), hd95_median=float(d.hd95.median()),
                    beta0_err=float(d.beta0_err.mean()),
                    cldice=(float(d.cldice.mean()) if with_cldice and 'cldice' in d else None))
    return df, {'prior':summarize('prior'),'flow':summarize('flow')}


# ======================================================================================
# TRAIN
# ======================================================================================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--patch',default='96,128,128')
    ap.add_argument('--widths',default='24,48,96,160')
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--eval_fold',type=int,default=0)
    ap.add_argument('--val_n',type=int,default=32)
    ap.add_argument('--val_every',type=int,default=10)
    ap.add_argument('--val_steps',type=int,default=2)
    ap.add_argument('--final_steps',type=int,default=4)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=ap.parse_args(); seed_all(a.seed)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    cache,out=Path(a.cache),Path(a.out); out.mkdir(parents=True,exist_ok=True)
    dev=torch.device(a.device)
    assert dev.type=='cuda' and torch.cuda.is_available(), 'CUDA runtime required'
    patch=tuple(int(x) for x in a.patch.split(',')); widths=tuple(int(x) for x in a.widths.split(','))
    assert all(s%(2**(len(widths)-1))==0 for s in patch), 'patch must be divisible by 8'
    tr,va,ev=make_split(cache,a.eval_fold,a.val_n,a.seed)
    clip=float(json.loads((cache/f'{tr[0]}_meta.json').read_text())['clip_mm'])

    cfg=dict(vars(a),patch=patch,widths=widths,clip_mm=clip,n_train=len(tr),n_val=len(va),n_eval=len(ev),
             architecture='Rectified SDF residual flow: image+x_t+x0+prior probs+uncertainty -> velocity',
             path='x_t=(1-t)x0+t*x1, target_v=x1-x0, no noise, no latent AE')
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    (out/'split.json').write_text(json.dumps(dict(train=tr,val=va,eval=ev),indent=2))

    model=VelocityUNet(widths).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.lr*0.05)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    start=0; best_dice=-1.0; best_hd95=1e9; best_epoch=0
    last=out/'last.pt'; best=out/'best.pt'
    if last.exists():
        s=torch.load(last,map_location=dev)
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        start=int(s['epoch']); best_dice=float(s.get('best_dice',-1)); best_hd95=float(s.get('best_hd95',1e9)); best_epoch=int(s.get('best_epoch',0))
        print(f'[resume] epoch={start} best={best_dice:.5f}@{best_epoch}',flush=True)

    print(f'GPU={torch.cuda.get_device_name(0)} | train={len(tr)} val={len(va)} untouched={len(ev)} | params={sum(p.numel() for p in model.parameters())/1e6:.2f}M',flush=True)
    logp=out/'train.jsonl'

    for epoch in range(start,a.epochs):
        model.train(); ds=CaseBalancedDataset(cache,tr,patch,epoch,a.seed,True)
        loader=DataLoader(ds,batch_size=1,shuffle=True,num_workers=a.workers,pin_memory=True,persistent_workers=False)
        sums={k:0.0 for k in ['loss','flow','sdf','dice','bce','stay','overlap']}; n=0; te=time.time()
        for step,(img,p,x0,x1,names) in enumerate(loader,1):
            # one case -> two patches = actual batch 2
            img=img.squeeze(0).to(dev,non_blocking=True); p=p.squeeze(0).to(dev,non_blocking=True)
            x0=x0.squeeze(0).to(dev,non_blocking=True); x1=x1.squeeze(0).to(dev,non_blocking=True)
            B=img.shape[0]
            t=torch.rand(B,device=dev)
            # inference starts exactly at t=0, so expose that point deliberately
            t=torch.where(torch.rand(B,device=dev)<0.20,torch.zeros_like(t),t)
            tt=t[:,None,None,None,None]
            xt=(1-tt)*x0 + tt*x1
            with torch.autocast(device_type='cuda',enabled=True):
                v=model(img,xt,x0,p,t)
                loss,parts=losses(v,xt,x0,x1,t,clip)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
            scaler.step(opt); scaler.update()
            sums['loss']+=float(loss.detach());
            for k in parts: sums[k]+=parts[k]
            n+=1
            if step%100==0 or step==len(loader):
                print(f'[epoch {epoch+1:02d}/{a.epochs}] {step:03d}/{len(loader)} loss={sums["loss"]/n:.4f} flow={sums["flow"]/n:.4f} diceL={sums["dice"]/n:.4f} gn={gn:.3f}',flush=True)
        sched.step()
        rec=dict(epoch=epoch+1,sec=time.time()-te,lr=opt.param_groups[0]['lr'],**{k:v/n for k,v in sums.items()})
        with open(logp,'a') as f: f.write(json.dumps(rec)+'\n')
        print('[epoch done]',json.dumps(rec),flush=True)

        # meaningful checkpoint selection: full-volume validation, not patch loss
        if (epoch+1)%a.val_every==0 or epoch+1==a.epochs:
            val_csv=out/f'val_epoch_{epoch+1:03d}.csv'
            _,rep=eval_full(model,cache,va,patch,dev,a.val_steps,val_csv,with_cldice=False)
            vd=rep['flow']['dice_mean']; vh=rep['flow']['hd95_mean']
            pdice=rep['prior']['dice_mean']; phd=rep['prior']['hd95_mean']
            print(f'[FULL VAL] prior Dice={pdice:.5f} HD95={phd:.4f} | flow Dice={vd:.5f} HD95={vh:.4f}',flush=True)
            better=(vd>best_dice+1e-6) or (abs(vd-best_dice)<=1e-6 and vh<best_hd95)
            if better:
                best_dice,best_hd95,best_epoch=vd,vh,epoch+1
                torch.save(dict(model=model.state_dict(),epoch=epoch+1,widths=widths,patch=patch,clip_mm=clip,
                                best_dice=best_dice,best_hd95=best_hd95),best)
                print(f'[BEST] epoch={best_epoch} Dice={best_dice:.5f} HD95={best_hd95:.4f}',flush=True)

        tmp=out/'last.pt.tmp'
        torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),sched=sched.state_dict(),epoch=epoch+1,
                        widths=widths,patch=patch,clip_mm=clip,best_dice=best_dice,best_hd95=best_hd95,best_epoch=best_epoch),tmp)
        tmp.replace(last)

    # FINAL untouched fold-0, automatically. No 1-case smoke / no decoder zoo.
    assert best.exists(), 'best.pt missing; full-volume validation never completed'
    s=torch.load(best,map_location=dev); model.load_state_dict(s['model']); model.eval()
    final_csv=out/'fold0_97_per_side.csv'
    df,rep=eval_full(model,cache,ev,patch,dev,a.final_steps,final_csv,with_cldice=True)
    report=dict(best_epoch=int(s['epoch']),n_cases=len(ev),steps=a.final_steps,summary=rep)
    report['delta']=dict(dice_mean=rep['flow']['dice_mean']-rep['prior']['dice_mean'],
                         dice_median=rep['flow']['dice_median']-rep['prior']['dice_median'],
                         hd95_mean=rep['flow']['hd95_mean']-rep['prior']['hd95_mean'],
                         hd95_median=rep['flow']['hd95_median']-rep['prior']['hd95_median'],
                         cldice=(rep['flow']['cldice']-rep['prior']['cldice'] if rep['flow']['cldice'] is not None else None),
                         beta0_err=rep['flow']['beta0_err']-rep['prior']['beta0_err'])
    (out/'fold0_97_report.json').write_text(json.dumps(report,indent=2))
    print('\n'+'='*100+'\nFINAL UNTOUCHED FOLD-0 (97 CASES)\n'+'='*100)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()

#!/usr/bin/env python3
# ---------------------------------------------------------------
# Conditional CNF that learns p(x, y | N) with *isotropic*
# normalisation (one shared σ) so generated rings stay circular.
# Saves:
#   • cnf_condN_iso.pt   (weights + xy_mean + iso_std + logN stats)
#   • counts.npy         (empirical multiplicity PMF)
#   • progress/epXXX.png (training snapshots, equal aspect)
# ---------------------------------------------------------------
import math, random, re, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# ───────────────────── 1. LOAD & ORGANISE EVENTS ──────────────────
FNAME = "opticks_hits_output.txt"
hits, pat = defaultdict(list), re.compile(r"([\deE.+-]+)\s+[\deE.+-]+\s+\(([^)]+)\).*")

for ln in open(FNAME):
    if (m := pat.match(ln)):
        ev = int(float(m.group(1)) // 1000)
        xy = np.fromstring(m.group(2), sep=',', dtype=np.float32)[:2]
        if xy.size == 2:
            hits[ev].append(xy)

all_ev = np.array(sorted(hits))
np.random.default_rng(42).shuffle(all_ev)
cut1, cut2 = int(.7*len(all_ev)), int(.85*len(all_ev))
ev_train, ev_val = all_ev[:cut1], all_ev[cut1:cut2]

# multiplicities
counts = np.array([len(hits[e]) for e in all_ev], dtype=np.int32)
np.save("counts.npy", counts)
logN_mu, logN_std = np.mean(np.log(counts[:cut1])), np.std(np.log(counts[:cut1]))+1e-7

# isotropic mean+std
all_xy = np.vstack([np.asarray(hits[e], np.float32) for e in ev_train])
xy_mean = all_xy.mean(0)
iso_std = np.sqrt(((all_xy - xy_mean)**2).sum(1).mean() / 2)  # scalar

def to_tensor(ev_subset):
    rows = []
    for e in ev_subset:
        pts = (np.asarray(hits[e], np.float32) - xy_mean) / iso_std  # (N,2)
        n_feat = ((math.log(len(pts)) - logN_mu) / logN_std).astype(np.float32)
        n_col  = np.full((len(pts), 1), n_feat, dtype=np.float32)
        rows.append(np.hstack([pts, n_col]))                          # (N,3)
    return torch.from_numpy(np.vstack(rows))

train, val = map(to_tensor, (ev_train, ev_val))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train, val = train.to(device), val.to(device)
torch.manual_seed(0)

# ───────────────────── 2. MODEL DEFINITIONS ───────────────────────
HIDDEN = 128
class VF(nn.Module):
    def __init__(self, d=3, h=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d+1, h), nn.SiLU(),
            nn.Linear(h,h),    nn.SiLU(),
            nn.Linear(h,h//2), nn.SiLU(),
            nn.Linear(h//2, 2)
        )
    def forward(self, t, y):
        return self.net(torch.cat([y, t.expand(len(y),1)], 1))

class CNF_ODE(nn.Module):
    def __init__(self, vf): super().__init__(); self.vf=vf
    def forward(self, t, states):
        y, logp = states
        y = y.detach().requires_grad_(True)
        with torch.enable_grad():
            dy_xy = self.vf(t, y)
            dy    = torch.cat([dy_xy, torch.zeros_like(y[:,2:3])], -1)
            v     = torch.empty_like(dy).bernoulli_().mul_(2).sub_(1)
            vdy   = (dy*v).sum()
            div   = (torch.autograd.grad(vdy, y, create_graph=True)[0]*v)\
                    .sum(-1, keepdim=True)
        return dy, -div

vf      = VF().to(device)
odefunc = CNF_ODE(vf)
scaler  = GradScaler(enabled=(device.type=="cuda"))

# prior over (x',y')
class RingPrior(torch.distributions.Distribution):
    arg_constraints, has_rsample, EPS = {}, False, 1e-6
    def __init__(self, R=1.0, s=0.05, dev="cpu"):
        super().__init__(); self.R,self.s,self.dev=R,s,dev
    def sample(self, shape=torch.Size()):
        θ = torch.rand(shape,device=self.dev)*2*math.pi
        r = self.R + self.s*torch.randn(shape,device=self.dev)
        return torch.stack([r*torch.cos(θ), r*torch.sin(θ)], -1)
    def log_prob(self, x):
        r = torch.sqrt((x**2).sum(-1)+self.EPS)
        return -((r-self.R)**2)/(2*self.s**2) - torch.log(r+self.EPS) \
               - math.log(self.s*math.sqrt(2*math.pi))

ring_R = torch.median(torch.sqrt((train[:, :2]**2).sum(-1))).item()
ring   = RingPrior(ring_R, 0.05, device)
center = torch.distributions.MultivariateNormal(torch.zeros(2,device=device),
                                                torch.eye(2,device=device)*0.05**2)
noise  = torch.distributions.MultivariateNormal(torch.zeros(2,device=device),
                                                torch.eye(2,device=device)*3.0**2)
mix_w  = torch.tensor([.60,.25,.15], device=device)
cat    = torch.distributions.Categorical(mix_w)
class MixXY(torch.distributions.Distribution):
    arg_constraints, has_rsample = {}, False
    def __init__(self, comps, cat): super().__init__(); self.c,self.cat=comps,cat
    def sample(self, shape=torch.Size()):
        N=int(torch.tensor(shape).prod()) or 1
        idx=self.cat.sample((N,))
        out=torch.empty(N,2,device=idx.device)
        for i,comp in enumerate(self.c):
            m=idx==i
            if m.any(): out[m]=comp.sample((m.sum(),))
        return out.reshape(*shape,2)
    def log_prob(self, x):
        logw=torch.log(mix_w)
        lp=torch.stack([c.log_prob(x)+lw for c,lw in zip(self.c,logw)],0)
        return torch.logsumexp(lp,0)
xy_prior = MixXY([ring,center,noise],cat)
n_prior  = torch.distributions.Normal(0.0,1.0)
def prior_logp(z):
    xy, z_n = z[..., :2], z[..., 2]
    return xy_prior.log_prob(xy) + n_prior.log_prob(z_n)

def forward_logp(x):
    log0 = torch.zeros(x.size(0),1,device=x.device)
    z, logp = odeint(odefunc,(x,log0),
                     torch.tensor([0.,1.],device=x.device),
                     method="rk4", options={"step_size":0.05},
                     atol=3e-4, rtol=3e-4)
    return prior_logp(z) + logp.squeeze(-1)

# ───────────────────── 3. TRAINING LOOP ────────────────────────────
Path("progress").mkdir(exist_ok=True)
sample_evs = random.sample(list(ev_val), 2)

def plot_epoch(ep):
    fig = plt.figure(figsize=(12,5))
    for j,ev in enumerate(sample_evs):
        real_xy = (np.asarray(hits[ev],np.float32)-xy_mean)/iso_std
        N=len(real_xy)
        n_feat=((math.log(N)-logN_mu)/logN_std).astype(np.float32)
        z_xy=xy_prior.sample((N,)).to(device)
        z_n=torch.full((N,1), n_feat, device=device)
        z=torch.cat([z_xy,z_n],1)
        with torch.no_grad(), autocast(device_type='cuda') if device.type=='cuda' else torch.device("cpu"):
            x_inv,_ = odeint(odefunc,(z,torch.zeros_like(z[:,:1])),
                             torch.tensor([1.,0.],device=device),
                             method="rk4",options={"step_size":0.05})
        gen_xy = x_inv[-1][:,:2].cpu().numpy()
        ax=fig.add_subplot(1,2,j+1)
        ax.scatter(real_xy[:,0],real_xy[:,1],s=5,lw=0,label="Real")
        ax.scatter(gen_xy[:,0], gen_xy[:,1],s=5,lw=0,label="Gen")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Event {ev} (N={N})")
        if j==0: ax.legend()
    fig.tight_layout()
    fig.savefig(Path("progress")/f"ep{ep:03d}.png",dpi=150); plt.close(fig)

BATCH,EPOCHS,PLOT_EVERY = 512, 200, 20
opt = torch.optim.AdamW(vf.parameters(), lr=3e-4)
rand_id=lambda d: torch.randint(0,d.size(0),(BATCH,),device=device)
amp_ctx=autocast(device_type='cuda') if device.type=='cuda' else torch.device("cpu")

for ep in range(EPOCHS):
    vf.train(); opt.zero_grad(set_to_none=True)
    x=train[rand_id(train)]
    with amp_ctx:
        loss=-forward_logp(x).mean()
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    if ep%PLOT_EVERY==0:
        plot_epoch(ep)
        print(f"Epoch {ep:03d} | loss {loss.item():.4f}")

# ───────────────────── 4. SAVE ────────────────────────────────────
torch.save(dict(
    vf_state_dict=vf.state_dict(),
    xy_mean=xy_mean,
    iso_std=iso_std,
    logN_mu=logN_mu,
    logN_std=logN_std
), "cnf_condN_iso.pt")
print("✓ Saved cnf_condN_iso.pt and counts.npy")

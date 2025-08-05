#!/usr/bin/env python3
# train_cnf_amp_fast_compat.py
# Compact CNF + AMP + Hutchinson  (ring + center + noise prior)

# ────────────────────────────── Imports ──────────────────────────────
import math, random, re, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from contextlib import nullcontext
from torchdiffeq import odeint                       # pip install torchdiffeq

from torch.cuda.amp import GradScaler, autocast      # works on all torch versions

# ───────────────────────── 1. LOAD & SPLIT ──────────────────────────
FNAME = "opticks_hits_output.txt"
hits, pat = defaultdict(list), re.compile(r"([\deE.+-]+)\s+[\deE.+-]+\s+\(([^)]+)\).*")

for ln in open(FNAME):
    if (m := pat.match(ln)):
        ev  = int(float(m.group(1)) // 1000)
        xy  = np.fromstring(m.group(2), sep=',', dtype=np.float32)[:2]
        if xy.size == 2: hits[ev].append(xy)

all_ev = np.array(sorted(hits))
np.random.default_rng(42).shuffle(all_ev)
cut1, cut2 = int(.7*len(all_ev)), int(.85*len(all_ev))
ev_train, ev_val = all_ev[:cut1], all_ev[cut1:cut2]

stack = lambda evs: torch.from_numpy(np.vstack([hits[e] for e in evs]))
train, val = map(stack, (ev_train, ev_val))

# ───────────────────────── 2. NORMALISE ─────────────────────────────
mean, std  = train.mean(0), train.std(0)
train, val = (train-mean)/std, (val-mean)/std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train, val = train.to(device), val.to(device)
torch.manual_seed(0)

# ───────────────────────── 3. CNF MODEL ────────────────────────────
HIDDEN = 128

class VF(nn.Module):
    def __init__(self, d=2, h=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d+1, h), nn.SiLU(),
            nn.Linear(h, h),   nn.SiLU(),
            nn.Linear(h, h//2), nn.SiLU(),
            nn.Linear(h//2, d)
        )
    def forward(self, t, y):
        return self.net(torch.cat([y, t.expand(len(y),1)], 1))

vf = VF().to(device)

class CNF_ODE(nn.Module):
    def __init__(self, vf): super().__init__(); self.vf = vf
    def forward(self, t, states):
        y, logp = states
        y = y.detach().requires_grad_(True)
        with torch.enable_grad():
            dy  = self.vf(t, y)
            v   = torch.empty_like(dy).bernoulli_().mul_(2).sub_(1)
            vdy = (dy*v).sum()
            div = (torch.autograd.grad(vdy, y, create_graph=True)[0]*v).sum(-1, keepdim=True)
        return dy, -div

odefunc = CNF_ODE(vf)
scaler  = GradScaler(enabled=(device.type == "cuda"))

# ───────────── 4. PRIOR = ring + center + wide noise ───────────────
class RingPrior(torch.distributions.Distribution):
    arg_constraints, has_rsample, EPS = {}, False, 1e-6
    def __init__(self, radius=1.0, std=0.05, device="cpu"):
        super().__init__();  self.R, self.s, self.dev = radius, std, device
    def sample(self, shape=torch.Size()):
        theta = torch.rand(shape, device=self.dev) * 2 * math.pi
        r     = self.R + self.s*torch.randn(shape, device=self.dev)
        return torch.stack([r*torch.cos(theta), r*torch.sin(theta)], -1)
    def log_prob(self, x):
        r = torch.sqrt((x**2).sum(-1)+self.EPS)
        return -((r-self.R)**2)/(2*self.s**2) - torch.log(r+self.EPS) \
               - math.log(self.s*math.sqrt(2*math.pi))

with torch.no_grad():
    ring_R = torch.median(torch.sqrt((train**2).sum(-1))).item()

ring      = RingPrior(ring_R, 0.05, device)
center    = torch.distributions.MultivariateNormal(torch.zeros(2,device=device),
                                                   torch.eye(2,device=device)*0.05**2)
noise_big = torch.distributions.MultivariateNormal(torch.zeros(2,device=device),
                                                   torch.eye(2,device=device)*3.0**2)

mix_w = torch.tensor([0.60, 0.25, 0.15], device=device)
cat   = torch.distributions.Categorical(mix_w)

class MixturePrior(torch.distributions.Distribution):
    arg_constraints, has_rsample = {}, False
    def __init__(self, comps, cat):
        super().__init__(); self.comps, self.cat = comps, cat
    def sample(self, shape=torch.Size()):
        N   = int(torch.tensor(shape).prod()) or 1
        idx = self.cat.sample((N,))
        out = torch.empty(N,2,device=idx.device)
        for i,c in enumerate(self.comps):
            m = idx==i
            if m.any(): out[m] = c.sample((m.sum(),))
        return out.reshape(*shape,2)
    def log_prob(self, x):
        log_w = torch.log(mix_w)
        lp = torch.stack([c.log_prob(x)+lw for c,lw in zip(self.comps,log_w)], 0)
        return torch.logsumexp(lp, 0)

prior = MixturePrior([ring, center, noise_big], cat)

# ───────────────── 5. LOG-PROB VIA ODE (FAST) ──────────────────────
def forward_logp(x):
    logp0 = torch.zeros(x.size(0),1,device=x.device,dtype=x.dtype)
    z, logp = odeint(
        odefunc, (x,logp0),
        torch.tensor([0.,1.],device=x.device),
        method="rk4",
        options=dict(step_size=0.05),
        atol=3e-4, rtol=3e-4
    )
    return prior.log_prob(z) + logp.squeeze(-1)

# ───────────────────────── 6. VISUALISATION ────────────────────────
Path("progress").mkdir(exist_ok=True)
sample_evs = random.sample(list(ev_val), 2)

def plot_epoch(ep):
    fig = plt.figure(figsize=(12,5))
    for k,ev in enumerate(sample_evs):
        real = (np.vstack(hits[ev])-mean.cpu().numpy())/std.cpu().numpy()
        z    = prior.sample((real.shape[0],))
        with torch.no_grad(), \
             (autocast() if device.type=='cuda' else nullcontext()):
            x_inv,_ = odeint(odefunc, (z, torch.zeros_like(z[:, :1])),
                             torch.tensor([1.,0.],device=device),
                             method="rk4", options=dict(step_size=0.05))
        gen  = (x_inv[-1].cpu()*std + mean).numpy()
        real_den = real*std.cpu().numpy()+mean.cpu().numpy()
        ax = fig.add_subplot(1,2,k+1)
        ax.scatter(*real_den.T, s=5, c='steelblue', lw=0, label='Real')
        ax.scatter(*gen.T,     s=5, c='orange',    lw=0, label='Gen')
        ax.set_title(f"Event {ev}"); ax.set_xticks([]); ax.set_yticks([])
        if k==0: ax.legend()
    fig.tight_layout()
    fig.savefig(Path("progress")/f"ep{ep:03d}.png", dpi=150); plt.close(fig)

# ───────────────────────── 7. TRAIN ────────────────────────────────
BATCH, EPOCHS, PLOT_EVERY = 512, 200, 20
opt     = torch.optim.AdamW(vf.parameters(), lr=3e-4)
rand_id = lambda d: torch.randint(0, d.size(0), (BATCH,), device=device)
amp_ctx = autocast if device.type=='cuda' else nullcontext

for ep in range(EPOCHS):
    vf.train(); opt.zero_grad(set_to_none=True)
    x = train[rand_id(train)]
    with amp_ctx():
        loss = -forward_logp(x).mean()
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(vf.parameters(), 0.5)
    scaler.step(opt); scaler.update()

    if ep % PLOT_EVERY == 0:
        plot_epoch(ep)
        print(f"Epoch {ep:03d} | loss {loss.item():.4f}")

# ───────────────────────── 8. SAVE ────────────────────────────────
torch.save({"vf_state_dict": vf.state_dict(),
            "mean": mean.cpu(),
            "std":  std.cpu()},
           "cnf_ring_central_noise.pt")
print("✓ Saved model to cnf_ring_central_noise.pt")
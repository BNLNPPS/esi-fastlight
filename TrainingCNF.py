#!/usr/bin/env python3
# ---------------------------------------------------------------
# TrainingCNF.py  (prior weights FIXED)
# Conditional CNF that learns p(x, y | cond) with *isotropic*
# normalisation. Prior over (x',y') uses a fixed-weight mixture.
# Saves:
#   • cnf_condN_iso.pt   (vf weights + stats + fixed mix_weights)
#   • counts.npy         (per-event pairs: [momentum_last, n_hits])
#   • progress/epXXX.png
# ---------------------------------------------------------------
import math, random, re, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

# ───────────────────── 0. CONFIG ──────────────────────────────────
OPTICKS_FILE   = "opticks_hits_output.txt"
PRIMARIES_CSV  = "primaries.csv"
COND_COLS_1BASED = [2,3,4,5,6,7, -1]   # columns 2..7 + last
HIDDEN = 128
BATCH, EPOCHS, PLOT_EVERY = 512, 200, 20
SEED = 42

# ───────────────────── 1. LOAD OPTICKS HITS ───────────────────────
hits, pat = defaultdict(list), re.compile(r"([\deE.+-]+)\s+[\deE.+-]+\s+\(([^)]+)\).*")
for ln in open(OPTICKS_FILE):
    m = pat.match(ln)
    if not m:
        continue
    ev = int(float(m.group(1)) // 1000)
    xy = np.fromstring(m.group(2), sep=',', dtype=np.float32)[:2]
    if xy.size == 2:
        hits[ev].append(xy)

# ───────────────────── 2. LOAD PRIMARIES (CONDITIONERS) ───────────
def parse_primaries(path):
    prim = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue
            ev = int(vals[0])
            sel = []
            for c in COND_COLS_1BASED:
                if c == -1:
                    sel.append(vals[-1])
                else:
                    sel.append(vals[c-1])
            prim[ev] = np.asarray(sel, dtype=np.float32)
    return prim

prim_map = parse_primaries(PRIMARIES_CSV)

# Overlap & split
ev_all = sorted(set(hits.keys()) & set(prim_map.keys()))
if not ev_all:
    raise RuntimeError("No overlapping events between hits and primaries.csv")

rng = np.random.default_rng(SEED)
ev_all = np.array(ev_all); rng.shuffle(ev_all)
cut1, cut2 = int(.7*len(ev_all)), int(.85*len(ev_all))
ev_train, ev_val = ev_all[:cut1], ev_all[cut1:cut2]

# counts.npy: [momentum_last, n_hits] per event
mom_last = np.array([prim_map[e][-1] for e in ev_all], dtype=np.float32)
n_hits   = np.array([len(hits[e])    for e in ev_all], dtype=np.int32)
np.save("counts.npy", np.stack([mom_last, n_hits], axis=1))

# ───────────────────── 3. NORMALISATIONS ──────────────────────────
all_xy_train = np.vstack([np.asarray(hits[e], np.float32) for e in ev_train])
xy_mean = all_xy_train.mean(0).astype(np.float32)
iso_std = float(np.sqrt(((all_xy_train - xy_mean)**2).sum(1).mean() / 2))

param_train = np.stack([prim_map[e] for e in ev_train], axis=0)  # (Etr,7)
param_mean  = param_train.mean(axis=0).astype(np.float32)
param_std   = (param_train.std(axis=0) + 1e-7).astype(np.float32)

def to_tensor(ev_subset):
    rows = []
    for e in ev_subset:
        pts = (np.asarray(hits[e], np.float32) - xy_mean) / iso_std
        p   = (prim_map[e] - param_mean) / param_std
        p_rep = np.repeat(p[None, :], len(pts), axis=0)
        rows.append(np.hstack([pts, p_rep]))   # (Ni, 2+7)
    return torch.from_numpy(np.vstack(rows))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
train, val = map(lambda es: to_tensor(es).to(device), (ev_train, ev_val))

# ── Model: ~4,903 trainable params (COND_DIM=7 → STATE_DIM=9; +t appended) ──
HIDDEN   = 53
COND_DIM = 7
STATE_DIM = 2 + COND_DIM  # (x,y) + 7 conditioners

class VF(torch.nn.Module):
    def __init__(self, d=STATE_DIM, h=HIDDEN):
        super().__init__()
        # Input to the first layer is (d + 1) because we concat scalar time t
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d + 1, h), torch.nn.SiLU(),   # 10 → 53
            torch.nn.Linear(h, h),     torch.nn.SiLU(),   # 53 → 53
            torch.nn.Linear(h, h//2),  torch.nn.SiLU(),   # 53 → 26
            torch.nn.Linear(h//2, 2)                      # 26 → 2  (dx, dy)
        )
    def forward(self, t, y):
        return self.net(torch.cat([y, t.expand(len(y), 1)], 1))

class CNF_ODE(torch.nn.Module):
    def __init__(self, vf): 
        super().__init__(); 
        self.vf = vf
    def forward(self, t, states):
        y, logp = states                  # y: (N, 2 + COND_DIM), logp: (N,1)
        y = y.detach().requires_grad_(True)
        with torch.enable_grad():
            dy_xy = self.vf(t, y)         # (N,2)
            zeros = torch.zeros_like(y[:, 2:])  # keep conditioners static
            dy    = torch.cat([dy_xy, zeros], -1)
            # Hutchinson trace estimator
            v     = torch.empty_like(dy).bernoulli_().mul_(2).sub_(1)
            vdy   = (dy * v).sum()
            div   = (torch.autograd.grad(vdy, y, create_graph=True)[0] * v)\
                    .sum(-1, keepdim=True)
        return dy, -div
        
vf      = VF().to(device)
odefunc = CNF_ODE(vf)
scaler  = GradScaler('cuda', enabled=(device.type=="cuda"))

# ── Prior over (x', y') with FIXED mixture weights ────────────────
class RingPrior(torch.distributions.Distribution):
    arg_constraints, has_rsample, EPS = {}, False, 1e-6
    def __init__(self, R=1.0, s=0.05, dev="cpu"):
        super().__init__(); self.R,self.s,self.dev=R,s,dev
    def sample(self, shape=torch.Size()):
        θ = torch.rand(shape, device=self.dev)*2*math.pi
        r = self.R + self.s*torch.randn(shape, device=self.dev)
        return torch.stack([r*torch.cos(θ), r*torch.sin(θ)], -1)
    def log_prob(self, x):
        r = torch.sqrt((x**2).sum(-1) + self.EPS)
        return -((r-self.R)**2)/(2*self.s**2) - torch.log(r+self.EPS) \
               - math.log(self.s*math.sqrt(2*math.pi))

ring_R = torch.median(torch.sqrt((train[:, :2]**2).sum(-1))).item()
ring   = RingPrior(ring_R, 0.05, device)
center = torch.distributions.MultivariateNormal(torch.zeros(2,device=device),
                                                torch.eye(2,device=device)*0.05**2)
noise  = torch.distributions.MultivariateNormal(torch.zeros(2,device=device),
                                                torch.eye(2,device=device)*3.0**2)

# FIXED logits (no gradients; not in optimizer)
mix_logits = torch.log(torch.tensor([.60, .25, .15], device=device))

class MixXY(nn.Module):
    def __init__(self, comps, logits_tensor):
        super().__init__()
        self.c = comps
        self.register_buffer("logits", logits_tensor)  # fixed buffer
    def weights(self):
        return torch.softmax(self.logits, 0)
    def sample(self, shape=torch.Size()):
        N = int(torch.tensor(shape).prod()) or 1
        w = self.weights()
        cat = torch.distributions.Categorical(w)
        idx = cat.sample((N,))
        out = torch.empty(N,2, device=w.device)
        for i, comp in enumerate(self.c):
            m = (idx == i)
            if m.any():
                out[m] = comp.sample((m.sum(),))
        return out.reshape(*shape, 2)
    def log_prob(self, x):
        lp   = torch.stack([comp.log_prob(x) for comp in self.c], 0)
        logw = torch.log_softmax(self.logits, 0).view(-1, *([1]*(lp.dim()-1)))
        return torch.logsumexp(logw + lp, dim=0)

xy_prior = MixXY([ring, center, noise], mix_logits)

def prior_logp(z):
    return xy_prior.log_prob(z[..., :2])

def forward_logp(x):
    log0 = torch.zeros(x.size(0), 1, device=x.device)
    z_traj, logp_traj = odeint(
        odefunc, (x, log0),
        torch.tensor([0., 1.], device=x.device),
        method="rk4", options={"step_size": 0.05},
        atol=3e-4, rtol=3e-4
    )
    z_T    = z_traj[-1]
    logp_T = logp_traj[-1].squeeze(-1)
    return prior_logp(z_T) + logp_T

# ───────────────────── 5. TRAINING LOOP ────────────────────────────
Path("progress").mkdir(exist_ok=True)
sample_evs = random.sample(list(ev_val), k=min(2, len(ev_val)))
amp_ctx = (autocast(device_type='cuda') if device.type=='cuda' else nullcontext())

opt = torch.optim.AdamW([{"params": vf.parameters()}], lr=3e-4)  # ← no prior weights
rand_id = lambda d: torch.randint(0, d.size(0), (BATCH,), device=device)

def plot_epoch(ep):
    fig = plt.figure(figsize=(12,5))
    for j, ev in enumerate(sample_evs):
        real_xy = (np.asarray(hits[ev], np.float32) - xy_mean) / iso_std
        N = len(real_xy)
        if N == 0: 
            continue
        p = (prim_map[ev] - param_mean) / param_std
        p_t = torch.from_numpy(np.repeat(p[None,:], N, axis=0)).to(device)

        z_xy = xy_prior.sample((N,)).to(device)
        z    = torch.cat([z_xy, p_t], 1)
        with torch.no_grad(), (autocast(device_type='cuda') if device.type=='cuda' else nullcontext()):
            x_inv, _ = odeint(
                odefunc, (z, torch.zeros_like(z[:, :1])),
                torch.tensor([1., 0.], device=device),
                method="rk4", options={"step_size": 0.05},
                atol=3e-4, rtol=3e-4
            )
        gen_xy = x_inv[-1][:, :2].cpu().numpy()

        ax = fig.add_subplot(1, 2, j+1)
        ax.scatter(real_xy[:,0], real_xy[:,1], s=5, lw=0, label="Real")
        ax.scatter(gen_xy[:,0],  gen_xy[:,1],  s=5, lw=0, label="Gen")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Event {ev} (N={N})")
        if j == 0: ax.legend()
    fig.tight_layout()
    fig.savefig(Path("progress")/f"ep{ep:03d}.png", dpi=150)
    plt.close(fig)

for ep in range(EPOCHS):
    vf.train(); opt.zero_grad(set_to_none=True)
    x = train[rand_id(train)]
    with amp_ctx:
        loss = -forward_logp(x).mean()
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()

    if ep % PLOT_EVERY == 0:
        plot_epoch(ep)
        w = torch.softmax(mix_logits, 0).detach().cpu().numpy()
        print(f"Epoch {ep:03d} | loss {loss.item():.4f} | FIXED weights ring/center/noise = {w}")

# ───────────────────── 6. SAVE ────────────────────────────────────
torch.save(dict(
    vf_state_dict=vf.state_dict(),
    xy_mean=xy_mean, iso_std=iso_std,
    param_mean=param_mean, param_std=param_std,
    mix_weights=torch.softmax(mix_logits, 0).detach().cpu().numpy(),
    cond_cols_1based=COND_COLS_1BASED
), "cnf_condN_iso.pt")
print("Saved cnf_condN_iso.pt (fixed prior weights) and counts.npy")
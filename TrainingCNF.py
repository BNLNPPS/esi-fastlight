#!/usr/bin/env python3
# ---------------------------------------------------------------
# TrainingCNF.py
# Conditional CNF that learns p(x, y | N) with *isotropic*
# normalisation (one shared σ) so generated rings stay circular.
# Now with **learnable mixture weights** for the latent (x', y') prior.
#
# Saves:
#   • cnf_condN_iso.pt   (vf weights + xy_mean + iso_std + logN stats + mix_logits)
#   • counts.npy         (empirical multiplicity PMF)
#   • progress/epXXX.png (training snapshots, equal aspect)
# ---------------------------------------------------------------
import math, random, re, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

# ───────────────────── 1. LOAD & ORGANISE EVENTS ──────────────────
FNAME = "opticks_hits_output.txt"
hits, pat = defaultdict(list), re.compile(r"([\deE.+-]+)\s+[\deE.+-]+\s+\(([^)]+)\).*")

for ln in open(FNAME):
    m = pat.match(ln)
    if not m: 
        continue
    ev = int(float(m.group(1)) // 1000)
    xy = np.fromstring(m.group(2), sep=',', dtype=np.float32)[:2]
    if xy.size == 2:
        hits[ev].append(xy)

all_ev = np.array(sorted(hits))
rng = np.random.default_rng(42)
rng.shuffle(all_ev)
cut1, cut2 = int(.7*len(all_ev)), int(.85*len(all_ev))
ev_train, ev_val = all_ev[:cut1], all_ev[cut1:cut2]

# multiplicities
counts = np.array([len(hits[e]) for e in all_ev], dtype=np.int32)
np.save("counts.npy", counts)
logN_mu  = float(np.mean(np.log(counts[:cut1])))
logN_std = float(np.std(np.log(counts[:cut1])) + 1e-7)

# isotropic mean+std (on training only)
all_xy = np.vstack([np.asarray(hits[e], np.float32) for e in ev_train])
xy_mean = all_xy.mean(0).astype(np.float32)
iso_std = float(np.sqrt(((all_xy - xy_mean)**2).sum(1).mean() / 2))  # scalar

def to_tensor(ev_subset):
    rows = []
    for e in ev_subset:
        pts = (np.asarray(hits[e], np.float32) - xy_mean) / iso_std  # (N,2)
        n_feat = np.float32((np.log(len(pts)) - logN_mu) / logN_std)
        n_col  = np.full((len(pts), 1), n_feat, dtype=np.float32)
        rows.append(np.hstack([pts, n_col]))                          # (N,3)
    return torch.from_numpy(np.vstack(rows))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train, val = map(lambda es: to_tensor(es).to(device), (ev_train, ev_val))
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
    def __init__(self, vf): 
        super().__init__(); 
        self.vf = vf
    def forward(self, t, states):
        y, logp = states
        y = y.detach().requires_grad_(True)
        with torch.enable_grad():
            dy_xy = self.vf(t, y)
            dy    = torch.cat([dy_xy, torch.zeros_like(y[:,2:3])], -1)
            # Hutchinson trace estimator (Rademacher)
            v     = torch.empty_like(dy).bernoulli_().mul_(2).sub_(1)
            vdy   = (dy*v).sum()
            div   = (torch.autograd.grad(vdy, y, create_graph=True)[0]*v)\
                    .sum(-1, keepdim=True)
        return dy, -div

vf      = VF().to(device)
odefunc = CNF_ODE(vf)
scaler  = GradScaler('cuda', enabled=(device.type=="cuda"))  # deprecation-safe

# ── Latent prior over (x', y') with **learnable mixture weights** ─
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

# Initialise ring radius from data in latent space
ring_R = torch.median(torch.sqrt((train[:, :2]**2).sum(-1))).item()
ring   = RingPrior(ring_R, 0.05, device)
center = torch.distributions.MultivariateNormal(torch.zeros(2,device=device),
                                                torch.eye(2,device=device)*0.05**2)
noise  = torch.distributions.MultivariateNormal(torch.zeros(2,device=device),
                                                torch.eye(2,device=device)*3.0**2)

# Learnable logits (init from 0.60/0.25/0.15)
mix_logits = nn.Parameter(torch.log(torch.tensor([.60, .25, .15], device=device)))

class MixXY(nn.Module):
    def __init__(self, comps, logits_param):
        super().__init__()
        self.c = comps
        self.logits = logits_param  # nn.Parameter shared with optimiser
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
        # lp: (K, *batch_shape); broadcast logw across batch dims
        lp   = torch.stack([comp.log_prob(x) for comp in self.c], 0)
        logw = torch.log_softmax(self.logits, 0).view(-1, *([1]*(lp.dim()-1)))
        return torch.logsumexp(logw + lp, dim=0)

xy_prior = MixXY([ring, center, noise], mix_logits)
n_prior  = torch.distributions.Normal(0.0,1.0)

def prior_logp(z):
    xy, z_n = z[..., :2], z[..., 2]
    return xy_prior.log_prob(xy) + n_prior.log_prob(z_n)

def forward_logp(x):
    """Return log p(x) via CNF: log p(z_T) + log|det J|, using final state only."""
    log0 = torch.zeros(x.size(0), 1, device=x.device)
    z_traj, logp_traj = odeint(
        odefunc, (x, log0),
        torch.tensor([0., 1.], device=x.device),
        method="rk4", options={"step_size": 0.05},
        atol=3e-4, rtol=3e-4
    )
    z_T    = z_traj[-1]              # (N, 3)
    logp_T = logp_traj[-1].squeeze(-1)  # (N,)
    return prior_logp(z_T) + logp_T

# ───────────────────── 3. TRAINING LOOP ────────────────────────────
Path("progress").mkdir(exist_ok=True)
sample_evs = random.sample(list(ev_val), k=min(2, len(ev_val)))

def plot_epoch(ep):
    fig = plt.figure(figsize=(12,5))
    for j, ev in enumerate(sample_evs):
        real_xy = (np.asarray(hits[ev], np.float32) - xy_mean) / iso_std
        N = len(real_xy)
        if N == 0:
            continue
        n_feat = np.float32((np.log(N) - logN_mu) / logN_std)   # ← use N, not pts

        # sample z from current learned prior, then invert through the flow
        z_xy = xy_prior.sample((N,)).to(device)
        z_n  = torch.full((N, 1), float(n_feat), device=device)
        z    = torch.cat([z_xy, z_n], 1)

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

BATCH, EPOCHS, PLOT_EVERY = 512, 200, 20

# Include mix_logits in the optimiser (no weight decay on logits)
opt = torch.optim.AdamW(
    [
        {"params": vf.parameters()},
        {"params": [mix_logits], "weight_decay": 0.0}
    ],
    lr=3e-4
)

rand_id = lambda d: torch.randint(0, d.size(0), (BATCH,), device=device)
amp_ctx = (autocast(device_type='cuda') if device.type=='cuda' else nullcontext())

for ep in range(EPOCHS):
    vf.train(); opt.zero_grad(set_to_none=True)
    x = train[rand_id(train)]
    with amp_ctx:
        loss = -forward_logp(x).mean()
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()

    if ep % PLOT_EVERY == 0:
        plot_epoch(ep)
        with torch.no_grad():
            w = torch.softmax(mix_logits, 0).detach().cpu().numpy()
        print(f"Epoch {ep:03d} | loss {loss.item():.4f} | weights ring/center/noise = {w}")

# ───────────────────── 4. SAVE ────────────────────────────────────
torch.save(dict(
    vf_state_dict=vf.state_dict(),
    xy_mean=xy_mean,
    iso_std=iso_std,
    logN_mu=logN_mu,
    logN_std=logN_std,
    mix_logits=mix_logits.detach().cpu().numpy()
), "cnf_condN_iso.pt")
print("✓ Saved cnf_condN_iso.pt (incl. mix_logits) and counts.npy")

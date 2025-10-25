#!/usr/bin/env python3
# ---------------------------------------------------------------
# CNF with translation-equivariant VF and isotropy regularization
# (KE->Cherenkov ring-proportional transform on last conditioner)
# ---------------------------------------------------------------

import math, random, re, os
from pathlib import Path
from collections import defaultdict
from matplotlib.colors import LogNorm
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchdiffeq import odeint

# ───────────────────── 0. CONFIG ──────────────────────────────────
OPTICKS_FILE = "opticks_hits_output.txt"
PRIMARIES_CSV = "primaries.csv"
# Take columns 2..7 plus the last one (1-based) -> 7-D conditioner
# The last one (c == -1) will be transformed from KE[GeV] -> ring-proportional scale.
COND_COLS_1BASED = [2, 3, 4, 5, 6, 7, -1]

# Cherenkov transform settings
CERENKOV_N = 1.33          # refractive index (e.g., water)
CERENKOV_SCALE = 1.0       # overall multiplicative scale for tan(theta_c)
DEFAULT_PID = 11           # assume electron if we can't infer a PDG from the row

# PDG -> mass [GeV] (common cases; extend if needed)
PDG_MASS_GEV = {
    11: 0.00051099895,  -11: 0.00051099895,  # e-/e+
    13: 0.1056583755,   -13: 0.1056583755,   # mu-/mu+
    211: 0.13957039,    -211: 0.13957039,    # pi+/pi-
    321: 0.493677,      -321: 0.493677,      # K+/K-
    2212: 0.9382720813, 2112: 0.9395654133,  # p, n
}

HIDDEN = 20
BATCH, EPOCHS, PLOT_EVERY = 512, 200, 20
SEED = 42

# Loss / training knobs
RING_S = 0.01
CEN_SIGMA = 0.01
RLOSS_LAMBDA = 0.05
CLOSS_LAMBDA = 0.10
ANCHOR_LAMBDA = 0.10
VF_PEN_LAMBDA = 1e-4

# Mixture prior + warmup
MIX_TARGET = torch.tensor([0.60, 0.25, 0.15], dtype=torch.float32)
MIX_PRIOR_LAMBDA = 0.05
LOGITS_WARMUP_EPOCHS = 50

# Warm-up to lock μ(cond) as translation early
CENTER_WARMUP_EPOCHS = 10

# μ(cond) supervised pre-train (centroids)
MU_PRETRAIN_STEPS = 2000
MU_PRETRAIN_BATCH = 1024
MU_PRETRAIN_LR    = 5e-3

# Isotropy penalty (keeps circles circular after inverse)
ISO_LAMBDA = 1e-2
ISO_EPS    = 0.25
ISO_K      = 24

# Robust extent for plotting / sampler noise clamp
EXTENT_Q_LOW  = 0.005
EXTENT_Q_HIGH = 0.995
EXTENT_PAD_FRAC = 0.02

# Optims
VF_LR = 3e-4
R_LR  = 1e-3
MU_LR = 3e-3
LOGITS_LR = 2e-3
WEIGHT_DECAY = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ───────────────────── Cherenkov helper ───────────────────────────
def _infer_pdg(vals):
    """Try to infer PDG code from second column (1-based col=2). If missing/invalid -> None."""
    if len(vals) < 2: 
        return None
    try:
        cand = int(round(vals[1]))
        return cand if cand in PDG_MASS_GEV else None
    except Exception:
        return None

def ke_to_ring_scale(ke_gev, pid=None, n=CERENKOV_N, scale=CERENKOV_SCALE):
    """
    Map kinetic energy [GeV] -> number proportional to Cherenkov ring radius.
      r ∝ tan(theta_c),   cos(theta_c) = 1/(n * beta), beta = sqrt(1 - 1/gamma^2), gamma = 1 + T/m.
    Below threshold (n*beta <= 1) => 0.
    """
    # choose mass from PDG if available; otherwise default to electron
    m = PDG_MASS_GEV.get(pid if pid is not None else DEFAULT_PID, PDG_MASS_GEV[DEFAULT_PID])
    # guard against non-physical inputs
    T = max(float(ke_gev), 0.0)
    gamma = 1.0 + (T / max(m, 1e-12))
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    if beta2 <= 0.0:
        return 0.0
    beta = math.sqrt(beta2)
    nb = n * beta
    if nb <= 1.0:
        return 0.0
    # Cherenkov angle
    c = 1.0 / nb
    # numerical guard (acos domain)
    c = min(1.0, max(-1.0, c))
    theta = math.acos(c)
    return float(scale * math.tan(theta))

# ───────────────────── 1. LOAD OPTICKS HITS ───────────────────────
hits = defaultdict(list)
pat = re.compile(r"([\deE.+-]+)\s+[\deE.+-]+\s+\(([^)]+)\).*")
if not os.path.exists(OPTICKS_FILE):
    raise FileNotFoundError(f"Missing {OPTICKS_FILE}")

for ln in open(OPTICKS_FILE, "r"):
    m = pat.match(ln)
    if not m: continue
    ev = int(float(m.group(1)) // 1000)
    xy = np.fromstring(m.group(2), sep=',', dtype=np.float32)[:2]
    if xy.size == 2: hits[ev].append(xy)

# ───────────────────── 2. LOAD PRIMARIES (CONDITIONERS) ───────────
def parse_primaries(path):
    """
    Returns:
      prim_map[e] -> np.array shape (7,) with last dim = transformed KE -> ring-proportional number
      ke_raw_map[e] -> original KE [GeV] from last column
    """
    prim, ke_raw_map = {}, {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            parts = s.split(",")
            try: 
                vals = [float(x) for x in parts]
            except ValueError: 
                continue
            ev = int(vals[0])

            # try to infer PDG from col 2 (1-based) if present/valid; else None -> default mass
            pid = _infer_pdg(vals)

            sel = []
            for c in COND_COLS_1BASED:
                if c == -1:
                    ke_gev = vals[-1]
                    ring_scale = ke_to_ring_scale(ke_gev, pid=pid)
                    sel.append(ring_scale)
                    ke_raw_map[ev] = float(ke_gev)
                else:
                    sel.append(vals[c-1])
            prim[ev] = np.asarray(sel, dtype=np.float32)
    return prim, ke_raw_map

prim_map, ke_raw_map = parse_primaries(PRIMARIES_CSV)

# overlap & split
ev_all = sorted(set(hits.keys()) & set(prim_map.keys()))
if not ev_all:
    raise RuntimeError("No overlapping events between hits and primaries.csv")

rng = np.random.default_rng(SEED)
ev_all = np.array(ev_all); rng.shuffle(ev_all)
cut1, cut2 = int(.7 * len(ev_all)), int(.85 * len(ev_all))
ev_train, ev_val = ev_all[:cut1], ev_all[cut1:cut2]

# counts.npy: [KE_raw_in_GeV, n_hits]  (still save the original KE here)
ke_last_raw = np.array([ke_raw_map[e] for e in ev_all], dtype=np.float32)
n_hits = np.array([len(hits[e]) for e in ev_all], dtype=np.int32)
np.save("counts.npy", np.stack([ke_last_raw, n_hits], axis=1))

# ───────────────────── 3. NORMALISATIONS / EXTENT ─────────────────
all_xy_train = np.vstack([np.asarray(hits[e], np.float32) for e in ev_train])
xy_mean = all_xy_train.mean(0).astype(np.float32)
iso_std = float(np.sqrt(((all_xy_train - xy_mean) ** 2).sum(1).mean() / 2))

param_train = np.stack([prim_map[e] for e in ev_train], axis=0)  # (Etr,7)
param_mean = param_train.mean(axis=0).astype(np.float32)
param_std  = (param_train.std(axis=0) + 1e-7).astype(np.float32)

def to_tensor(ev_subset):
    rows = []; ev_ids = []
    for e in ev_subset:
        pts = (np.asarray(hits[e], np.float32) - xy_mean) / iso_std
        if len(pts) == 0: continue
        p = (prim_map[e] - param_mean) / param_std
        p_rep = np.repeat(p[None, :], len(pts), axis=0)
        rows.append(np.hstack([pts, p_rep]))        # (Ni, 2+7)
        ev_ids.append(np.full(len(pts), e, dtype=np.int64))
    data = np.vstack(rows)
    ev_ids = np.concatenate(ev_ids)
    return torch.from_numpy(data), ev_ids

train, train_ev_ids = to_tensor(ev_train)
val,   val_ev_ids   = to_tensor(ev_val)
train = train.to(device); val = val.to(device)
train_ev_ids = np.asarray(train_ev_ids)

# ---- Base ring radius: per-event centroids in normalized coords
def per_event_radii(ev_subset):
    radii = []
    for e in ev_subset:
        pts = (np.asarray(hits[e], np.float32) - xy_mean) / iso_std
        if len(pts) == 0: continue
        c = pts.mean(0)
        r = np.sqrt(((pts - c)**2).sum(axis=1))
        radii.append(r)
    if not radii: return 1.0
    return float(np.median(np.concatenate(radii)))

ring_R_init = per_event_radii(ev_train)

# robust data-space extent
def compute_dataset_extent(ev_ids, pad_frac=EXTENT_PAD_FRAC, qlo=EXTENT_Q_LOW, qhi=EXTENT_Q_HIGH):
    all_xy = np.vstack([np.asarray(hits[e], np.float32) for e in ev_ids if len(hits[e]) > 0])
    xlo, xhi = np.quantile(all_xy[:,0], [qlo, qhi])
    ylo, yhi = np.quantile(all_xy[:,1], [qlo, qhi])
    dx, dy = (xhi - xlo), (yhi - ylo)
    xpad, ypad = dx*pad_frac, dy*pad_frac
    return [float(xlo - xpad), float(xhi + xpad), float(ylo - ypad), float(yhi + ypad)]

DATA_EXTENT = compute_dataset_extent(ev_all)
print(f"[extent] Using robust dataset extent (q={EXTENT_Q_LOW:.3f}..{EXTENT_Q_HIGH:.3f}): "
      f"x=[{DATA_EXTENT[0]:.1f},{DATA_EXTENT[1]:.1f}], y=[{DATA_EXTENT[2]:.1f},{DATA_EXTENT[3]:.1f}]")

# ───────────────────── 3b. CENTROIDS FOR μ PRE-TRAIN ──────────────
centroid_pairs = []
for e in ev_train:
    pts = (np.asarray(hits[e], np.float32) - xy_mean) / iso_std
    if len(pts) == 0: continue
    c_xy = pts.mean(0).astype(np.float32)
    p = ((prim_map[e] - param_mean) / param_std).astype(np.float32)
    centroid_pairs.append((p, c_xy))
if not centroid_pairs:
    raise RuntimeError("No centroids could be computed from training events.")
cent_cond = torch.from_numpy(np.stack([p for p,_ in centroid_pairs], 0)).to(device)
cent_xy   = torch.from_numpy(np.stack([c for _,c in centroid_pairs], 0)).to(device)

# ───────────────────── 4. MODEL ───────────────────────────────────
COND_DIM  = 7
STATE_DIM = 2 + COND_DIM

class VF(nn.Module):
    """Vector field on centered coords. Input = [(xy-μ(cond)), cond, t]; output = d(xy)/dt."""
    def __init__(self, d=STATE_DIM, h=HIDDEN):
        super().__init__()
        # d+1 == 2 + COND_DIM + 1 (time)
        self.net = nn.Sequential(
            nn.Linear(d + 1, h), nn.SiLU(),
            nn.Linear(h, h),     nn.SiLU(),
            nn.Linear(h, h//2),  nn.SiLU(),
            nn.Linear(h//2, 2)
        )
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    def forward(self, inp):
        return self.net(inp)

class CNF_ODE(nn.Module):
    """ODE using centered inputs for VF; enables-grad internally for divergence."""
    def __init__(self, vf, mu_fn):
        super().__init__(); self.vf = vf; self.mu_fn = mu_fn
        self.allow_grad_wrt_y = False
        self.compute_divergence = True  # can be disabled during sampling/plotting
    def forward(self, t, states):
        y, logp = states
        if self.allow_grad_wrt_y:
            y = y.requires_grad_(True)
        else:
            y = y.detach().requires_grad_(True)

        xy, cond = y[:, :2], y[:, 2:]
        mu = self.mu_fn(cond)
        inp = torch.cat([xy - mu, cond, t.expand(len(y), 1)], 1)

        if not self.compute_divergence:
            dy_xy = self.vf(inp)
            zeros = torch.zeros_like(cond)
            dy = torch.cat([dy_xy, zeros], -1)
            div = torch.zeros(len(y), 1, device=y.device, dtype=y.dtype)
            return dy, -div

        with torch.enable_grad():
            dy_xy = self.vf(inp)
            zeros = torch.zeros_like(cond)
            dy = torch.cat([dy_xy, zeros], -1)
            v   = torch.empty_like(dy).bernoulli_().mul_(2).sub_(1)
            vdy = (dy * v).sum()
            div = (torch.autograd.grad(vdy, y, create_graph=True)[0] * v).sum(-1, keepdim=True)
            return dy, -div

class CondRingPrior(nn.Module):
    EPS = 1e-6
    def __init__(self, cond_dim, base_R=1.0, s=0.05, hidden=32):
        super().__init__()
        self.s = float(s); self.base_R = float(base_R)
        self.R_net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        self.mu_net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
            nn.Linear(hidden, 2)
        )
        # IMPORTANT: zero ONLY last layers so learning isn't dead
        with torch.no_grad():
            last_R  = [m for m in self.R_net.modules() if isinstance(m, nn.Linear)][-1]
            last_mu = [m for m in self.mu_net.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last_R.weight);  nn.init.zeros_(last_R.bias)
            nn.init.zeros_(last_mu.weight); nn.init.zeros_(last_mu.bias)

    def R(self, cond):
        delta = 0.1 * self.R_net(cond).squeeze(-1)
        return torch.clamp(self.base_R * torch.exp(delta), min=1e-3)
    def mu(self, cond):
        return self.mu_net(cond)
    def sample(self, cond):
        N = cond.size(0)
        theta = torch.rand(N, device=cond.device) * (2 * math.pi)
        r = self.R(cond) + self.s * torch.randn(N, device=cond.device)
        xy = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
        return xy + self.mu(cond)
    def log_prob(self, x, cond):
        xc = x - self.mu(cond)
        r = torch.sqrt((xc ** 2).sum(-1) + self.EPS)
        R = self.R(cond)
        return -((r - R) ** 2) / (2 * self.s ** 2) - torch.log(r + self.EPS) \
               - math.log(self.s * math.sqrt(2 * math.pi))

class CondCenterGaussian(nn.Module):
    def __init__(self, mu_fn, sigma=0.01):
        super().__init__()
        self.mu_fn = mu_fn
        self.sigma = float(sigma)
        self._log_norm = -math.log(2 * math.pi * (self.sigma ** 2))
    def sample(self, cond):
        return self.mu_fn(cond) + self.sigma * torch.randn(cond.size(0), 2, device=cond.device)
    def log_prob(self, x, cond):
        diff2 = ((x - self.mu_fn(cond)) ** 2).sum(-1)
        return -0.5 * (diff2 / (self.sigma ** 2)) + self._log_norm

class CondMixXY_LearnedWeights(nn.Module):
    def __init__(self, ring_prior_cond, center_cond, noise_dist, cond_dim, hidden=32,
                 init_probs=(0.60, 0.25, 0.15)):
        super().__init__()
        self.ring, self.center, self.noise = ring_prior_cond, center_cond, noise_dist
        self.logits_net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
            nn.Linear(hidden, 3)
        )
        with torch.no_grad():
            for m in self.logits_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
            self.logits_net[-1].bias.copy_(torch.log(torch.tensor(init_probs)))
    def weights(self, cond): return torch.softmax(self.logits_net(cond), dim=-1)
    def log_weights(self, cond): return torch.log_softmax(self.logits_net(cond), dim=-1)
    def sample(self, cond, return_comp=False):
        N = cond.size(0); w = self.weights(cond)
        idx = torch.distributions.Categorical(probs=w).sample()
        out = torch.empty(N, 2, device=cond.device)
        m = (idx == 0)
        if m.any(): out[m] = self.ring.sample(cond[m])
        m = (idx == 1)
        if m.any(): out[m] = self.center.sample(cond[m])
        m = (idx == 2)
        if m.any(): out[m] = self.noise.sample((int(m.sum()),)).to(cond.device)
        return (out, idx) if return_comp else out
    def log_prob(self, x, cond):
        lp0 = self.ring.log_prob(x, cond)
        lp1 = self.center.log_prob(x, cond)
        lp2 = self.noise.log_prob(x)
        LP  = torch.stack([lp0, lp1, lp2], dim=-1)
        logw = self.log_weights(cond)
        return torch.logsumexp(LP + logw, dim=-1)

# Assemble
vf = VF().to(device)
ring   = CondRingPrior(cond_dim=COND_DIM, base_R=ring_R_init, s=RING_S, hidden=32).to(device)
center = CondCenterGaussian(ring.mu, sigma=CEN_SIGMA)
noise  = torch.distributions.MultivariateNormal(torch.zeros(2, device=device),
                                                torch.eye(2, device=device) * (3.0 ** 2))
xy_prior = CondMixXY_LearnedWeights(ring, center, noise, COND_DIM, hidden=32).to(device)
odefunc = CNF_ODE(vf, ring.mu).to(device)

scaler = GradScaler(enabled=(device.type == "cuda"))

def prior_logp(z):
    x2, cond = z[..., :2], z[..., 2:]
    return xy_prior.log_prob(x2, cond)

def forward_logp(x):
    log0 = torch.zeros(x.size(0), 1, device=x.device)
    odefunc.compute_divergence = True
    z_traj, logp_traj = odeint(
        odefunc, (x, log0), torch.tensor([0., 1.], device=x.device),
        method="rk4", options={"step_size": 0.05}, atol=3e-4, rtol=3e-4
    )
    z_T    = z_traj[-1]
    logp_T = logp_traj[-1].squeeze(-1)
    return prior_logp(z_T) + logp_T, z_T

# ───────────────────── 4b. PRE-TRAIN μ(cond) ON CENTROIDS ─────────
print(f"[mu-pretrain] {len(centroid_pairs)} event centroids")
mu_opt = torch.optim.AdamW(ring.mu_net.parameters(), lr=MU_PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
perm = torch.randperm(len(centroid_pairs))
cent_cond_shuf = cent_cond[perm]
cent_xy_shuf   = cent_xy[perm]
steps = MU_PRETRAIN_STEPS; bs = MU_PRETRAIN_BATCH
for step in range(steps):
    i0 = (step * bs) % len(centroid_pairs); i1 = i0 + bs
    cc = cent_cond_shuf[i0:i1]; cx = cent_xy_shuf[i0:i1]
    if cc.size(0) == 0: continue
    mu_pred = ring.mu(cc)
    loss_mu = ((mu_pred - cx)**2).mean()
    mu_opt.zero_grad(set_to_none=True); loss_mu.backward(); mu_opt.step()
    if (step % 200) == 0:
        with torch.no_grad():
            err = ((ring.mu(cent_cond) - cent_xy)**2).mean().sqrt().item()
        print(f"[mu-pretrain] step {step:04d} | batch_loss {loss_mu.item():.4f} | RMS err vs centroids {err:.4f}")
print("[mu-pretrain] done.")

# ───────────────────── 5. TRAINING LOOP ───────────────────────────
Path("progress").mkdir(exist_ok=True)
sample_evs = random.sample(list(ev_val), k=min(2, len(ev_val)))

amp_ctx = (autocast(device_type='cuda') if device.type == 'cuda' else nullcontext())
opt = torch.optim.AdamW([
    {"params": vf.parameters(),                    "lr": VF_LR,    "name": "vf"},
    {"params": ring.R_net.parameters(),            "lr": R_LR,     "name": "R"},
    {"params": ring.mu_net.parameters(),           "lr": MU_LR,    "name": "mu"},
    {"params": xy_prior.logits_net.parameters(),   "lr": LOGITS_LR,"name": "logits"},
], weight_decay=WEIGHT_DECAY)

vf_pg     = next(pg for pg in opt.param_groups if pg.get("name","") == "vf")
logits_pg = next(pg for pg in opt.param_groups if pg.get("name","") == "logits")
vf_pg["base_lr"] = vf_pg["lr"]; logits_pg["base_lr"] = logits_pg["lr"]

def rand_id(d): return torch.randint(0, d.size(0), (BATCH,), device=device)

def plot_epoch(ep):
    fig = plt.figure(figsize=(12, 5))
    for j, ev in enumerate(sample_evs):
        real_xy = np.asarray(hits[ev], np.float32)
        N = len(real_xy); 
        if N == 0: 
            continue
        p  = (prim_map[ev] - param_mean) / param_std
        p_t = torch.from_numpy(np.repeat(p[None, :], N, axis=0)).to(device)
        z_xy = xy_prior.sample(p_t); z = torch.cat([z_xy, p_t], 1)
        odefunc.compute_divergence = False  # fast inverse map
        x_inv, _ = odeint(odefunc, (z, torch.zeros_like(z[:, :1])),
                           torch.tensor([1., 0.], device=device),
                           method="rk4", options={"step_size": 0.05}, atol=3e-4, rtol=3e-4)
        odefunc.compute_divergence = True
        gen_xy = (x_inv[-1][:, :2].detach().cpu().numpy() * iso_std + xy_mean)

        ax = fig.add_subplot(1, 2, j+1)
        extent = DATA_EXTENT
        H1, _, _ = np.histogram2d(real_xy[:,0], real_xy[:,1], bins=100,
                                  range=[[extent[0], extent[1]], [extent[2], extent[3]]])
        H2, _, _ = np.histogram2d(gen_xy[:,0], gen_xy[:,1], bins=100,
                                  range=[[extent[0], extent[1]], [extent[2], extent[3]]])
        ax.imshow(H1.T, origin="lower", extent=extent, aspect="equal")
        ax.imshow(H2.T, origin="lower", extent=extent, aspect="equal", alpha=0.6)
        ax.set_title(f"Event {ev} (N={N})"); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(Path("progress")/f"ep{ep:03d}.png", dpi=150); plt.close(fig)

seen_train_events = set()

def sym_kl(p, q):
    p = torch.clamp(p, 1e-8, 1.0); q = torch.clamp(q, 1e-8, 1.0)
    kl_pq = (p * (p.log() - q.log())).sum(-1)
    kl_qp = (q * (q.log() - p.log())).sum(-1)
    return 0.5 * (kl_pq + kl_qp)

theta_iso = torch.linspace(0, 2*math.pi, ISO_K+1, device=device)[:-1]
unit_iso = torch.stack([theta_iso.cos(), theta_iso.sin()], -1)  # (K,2)

for ep in range(EPOCHS):
    vf_pg["lr"]     = 0.0 if ep < CENTER_WARMUP_EPOCHS else vf_pg["base_lr"]
    logits_pg["lr"] = 0.0 if ep < LOGITS_WARMUP_EPOCHS else logits_pg["base_lr"]

    vf.train(); ring.train(); xy_prior.train()
    opt.zero_grad(set_to_none=True)

    idx = rand_id(train); x = train[idx]
    seen_train_events.update(train_ev_ids[idx.detach().cpu().numpy()].tolist())

    with amp_ctx:
        logp, zT = forward_logp(x)
        nll = -logp.mean()

    with autocast(device_type='cuda', enabled=False):
        zxy   = zT[:, :2].float()
        condT = zT[:, 2:].float()

        mu_i  = ring.mu(condT)
        r_lat = torch.sqrt(((zxy - mu_i)**2).sum(-1) + 1e-6)
        R_pred = ring.R(condT)
        rloss = RLOSS_LAMBDA * ((r_lat - R_pred)**2).mean()

        condX = x[:, 2:].float()
        keys, inv = torch.unique(condX, dim=0, return_inverse=True)
        G = keys.size(0)
        ones = torch.ones(inv.size(0), 1, device=x.device, dtype=torch.float32)

        sum_z  = torch.zeros(G, 2, device=x.device, dtype=torch.float32)
        sum_mu = torch.zeros(G, 2, device=x.device, dtype=torch.float32)
        sum_x  = torch.zeros(G, 2, device=x.device, dtype=torch.float32)
        cnts   = torch.zeros(G, 1, device=x.device, dtype=torch.float32)

        sum_z.index_add_(0, inv, zxy)
        sum_mu.index_add_(0, inv, mu_i)
        sum_x.index_add_(0, inv, x[:, :2].float())
        cnts.index_add_(0, inv, ones)

        mean_z  = sum_z / cnts.clamp_min(1.0)
        mean_mu = sum_mu / cnts.clamp_min(1.0)
        x_ctr   = sum_x / cnts.clamp_min(1.0)

        closs = CLOSS_LAMBDA * ((mean_z - mean_mu)**2).mean()

        # VF penalty at t=0 on centered inputs
        with torch.no_grad():
            mu_x = ring.mu(x[:, 2:].float())
        t0 = torch.zeros(len(x), 1, device=x.device, dtype=torch.float32)
        vf_inp0 = torch.cat([x[:, :2].float() - mu_x, x[:, 2:].float(), t0], 1)
        v0 = vf(vf_inp0)
        vpen = VF_PEN_LAMBDA * (v0**2).mean()

        with torch.no_grad():
            target = MIX_TARGET.to(device)
        w_keys = torch.softmax(xy_prior.logits_net(keys), dim=-1)
        mix_kl = sym_kl(w_keys, target.expand_as(w_keys)).mean()
        mix_term = MIX_PRIOR_LAMBDA * mix_kl

        # inverse-anchor
        mu_keys = ring.mu(keys.float())
        z_mu    = torch.cat([mu_keys, keys.float()], 1)
        odefunc.allow_grad_wrt_y = True
        odefunc.compute_divergence = True
        x_from_mu, _ = odeint(odefunc, (z_mu, torch.zeros(G,1,device=x.device)),
                              torch.tensor([1.,0.], device=x.device),
                              method="rk4", options={"step_size":0.05}, atol=3e-4, rtol=3e-4)
        odefunc.allow_grad_wrt_y = False
        x_mu = x_from_mu[-1][:,:2]
        anchor = ANCHOR_LAMBDA * ((x_mu - x_ctr)**2).mean()

        # isotropy penalty around center
        z_ring = (mu_keys[:, None, :] + ISO_EPS * unit_iso[None, :, :]).reshape(-1, 2)
        z_ring_full = torch.cat([z_ring, keys[:, None, :].expand(-1, ISO_K, -1).reshape(-1, COND_DIM)], 1)
        odefunc.compute_divergence = False
        x_ring_traj, _ = odeint(odefunc, (z_ring_full, torch.zeros(G*ISO_K,1, device=x.device)),
                                torch.tensor([1.,0.], device=x.device),
                                method="rk4", options={"step_size":0.05}, atol=3e-4, rtol=3e-4)
        odefunc.compute_divergence = True
        x_ring = x_ring_traj[-1][:,:2].reshape(G, ISO_K, 2)
        rad = ((x_ring - x_ctr[:, None, :])**2).sum(-1).sqrt()      # (G,K)
        iso_pen = ISO_LAMBDA * rad.std(dim=1).mean()

        loss = nll.float() + rloss + closs + vpen + mix_term + anchor + iso_pen

    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()

    with torch.no_grad():
        med_r = r_lat.median().item()
        mean_R = R_pred.mean().item()
        w_mean = xy_prior.weights(condT).mean(0).detach().cpu().numpy()
        c_err  = ((mean_z - mean_mu)**2).mean().sqrt().item()
        vnorm0 = (v0**2).mean().sqrt().item()
        mix_kl_val = mix_kl.item()
        anchor_val = ((x_mu - x_ctr)**2).mean().item()
        iso_val = rad.std(dim=1).mean().item()

    if ep % PLOT_EVERY == 0: plot_epoch(ep)

    print(f"Epoch {ep:03d} | loss {loss.item():.4f} | nll {nll.item():.4f} "
          f"| rloss {rloss.item():.4f} | closs {closs.item():.4f} | anchor {anchor_val:.4f} "
          f"| iso {iso_val:.4f} | mixKL {mix_kl_val:.4f} | median r_lat {med_r:.3f} vs mean R(cond) {mean_R:.3f} "
          f"| center_err {c_err:.3f} | ||v0||_rms {vnorm0:.4f} "
          f"| mean weights [ring,center,noise] = {w_mean}")

# ───────────────────── 6. SAVE ────────────────────────────────────
torch.save(dict(
    vf_state_dict=vf.state_dict(),
    ring_state_dict=ring.state_dict(),
    logits_state_dict=xy_prior.logits_net.state_dict(),
    xy_mean=xy_mean,
    iso_std=iso_std,
    param_mean=param_mean,
    param_std=param_std,
    cond_cols_1based=COND_COLS_1BASED,
    ring_base_R=ring_R_init,
    ring_s=RING_S,
    center_sigma=CEN_SIGMA,
    prior_kind="cond_ring_mixture_equivariant_iso",
    mix_target=MIX_TARGET.numpy(),
    mix_prior_lambda=MIX_PRIOR_LAMBDA,
    logits_warmup_epochs=LOGITS_WARMUP_EPOCHS,
    anchor_lambda=ANCHOR_LAMBDA,
    data_extent=DATA_EXTENT,
    extent_quantiles=(EXTENT_Q_LOW, EXTENT_Q_HIGH),
    center_warmup_epochs=CENTER_WARMUP_EPOCHS,
    mu_pretrain_steps=MU_PRETRAIN_STEPS,
    iso_lambda=ISO_LAMBDA,
    iso_eps=ISO_EPS,
    iso_k=ISO_K,
    cherenkov_n=CERENKOV_N,
    cherenkov_scale=CERENKOV_SCALE,
    ke_transform_note="Last conditioner column is KE[GeV] transformed to ring-proportional scale via tan(theta_c)."
), "cnf_condN_condRing.pt")

print("Saved cnf_condN_condRing.pt and counts.npy")

# ───────────────────── 7. (SAMPLER) ───────────────────────────────
@torch.no_grad()
def sample_event(cond_raw, N=200, ckpt_path="cnf_condN_condRing.pt", dev=device, max_resample=5):
    ckpt = torch.load(ckpt_path, map_location=dev if dev.type=="cuda" else "cpu", weights_only=False)
    vf_s = VF().to(dev); vf_s.load_state_dict(ckpt["vf_state_dict"])
    ring_s = CondRingPrior(COND_DIM, base_R=ckpt["ring_base_R"], s=ckpt["ring_s"], hidden=32).to(dev)
    ring_s.load_state_dict(ckpt["ring_state_dict"])
    center_s = CondCenterGaussian(ring_s.mu, sigma=ckpt.get("center_sigma", 0.05))
    noise_s  = torch.distributions.MultivariateNormal(torch.zeros(2, device=dev),
                                                      torch.eye(2, device=dev) * (3.0 ** 2))
    xy_prior_s = CondMixXY_LearnedWeights(ring_s, center_s, noise_s, COND_DIM, hidden=32).to(dev)
    xy_prior_s.logits_net.load_state_dict(ckpt["logits_state_dict"])
    odefunc_s = CNF_ODE(vf_s, ring_s.mu).to(dev)

    xy_mean_s = torch.tensor(ckpt["xy_mean"], device=dev)
    iso_std_s = torch.tensor(ckpt["iso_std"], device=dev)
    p_mean_s  = torch.tensor(ckpt["param_mean"], device=dev)
    p_std_s   = torch.tensor(ckpt["param_std"],  device=dev)
    extent    = ckpt.get("data_extent", None)

    p = (torch.tensor(cond_raw, dtype=torch.float32, device=dev) - p_mean_s) / p_std_s
    p = p.repeat(N, 1)

    z_xy, comp_idx = xy_prior_s.sample(p, return_comp=True)
    z = torch.cat([z_xy, p], dim=1)

    odefunc_s.compute_divergence = False
    x_inv, _ = odeint(odefunc_s, (z, torch.zeros(N,1, device=dev)),
                      torch.tensor([1., 0.], device=dev),
                      method="rk4", options={"step_size": 0.05}, atol=3e-4, rtol=3e-4)
    odefunc_s.compute_divergence = True
    xy_norm = x_inv[-1][:, :2]
    xy = xy_norm * iso_std_s + xy_mean_s

    if extent is not None:
        xmin, xmax, ymin, ymax = extent
        for _ in range(max_resample):
            noise_mask = (comp_idx == 2)
            if noise_mask.sum().item() == 0: break
            oob = (xy[:,0] < xmin) | (xy[:,0] > xmax) | (xy[:,1] < ymin) | (xy[:,1] > ymax)
            bad = noise_mask & oob
            if bad.sum().item() == 0: break
            z_xy_new = xy_prior_s.noise.sample((int(bad.sum().item()),)).to(dev)
            z_bad = torch.cat([z_xy_new, p[bad]], 1)
            odefunc_s.compute_divergence = False
            x_inv_bad, _ = odeint(odefunc_s, (z_bad, torch.zeros(int(bad.sum().item()),1, device=dev)),
                                  torch.tensor([1., 0.], device=dev),
                                  method="rk4", options={"step_size": 0.05}, atol=3e-4, rtol=3e-4)
            odefunc_s.compute_divergence = True
            xy_bad = x_inv_bad[-1][:,:2] * iso_std_s + xy_mean_s
            xy[bad] = xy_bad

        xy[:,0] = torch.clamp(xy[:,0], xmin, xmax)
        xy[:,1] = torch.clamp(xy[:,1], ymin, ymax)

    return xy.detach().cpu().numpy()

# ───────────────────── 8. HITMAPS ─────────────────────────────────
def _first_n_evs_from_primaries(n=10):
    evs = []
    with open(PRIMARIES_CSV, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            parts = s.split(",")
            try: vals = [float(x) for x in parts]
            except ValueError: continue
            ev = int(vals[0])
            if ev in prim_map and ev in hits and len(hits[ev]) > 0:
                evs.append(ev)
                if len(evs) >= n: break
    return evs

def _draw_hitmap(ax, xy, extent, bins=100, vmax=None, title=""):
    if len(xy) == 0:
        ax.set_title(title + " (no hits)")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        return

    H, _, _ = np.histogram2d(
        xy[:,0], xy[:,1], bins=bins,
        range=[[extent[0], extent[1]], [extent[2], extent[3]]]
    )

    # Transpose for imshow and mask zeros so LogNorm is happy
    H = H.T
    H_masked = np.where(H > 0, H, np.nan)

    # Choose color scale; keep optional vmax override
    v_max = (np.nanmax(H_masked) if vmax is None else float(vmax))
    if not np.isfinite(v_max) or v_max <= 0:
        v_max = 1.0  # fallback

    # Integer counts => set vmin=1 so we’re effectively log(1..v_max)
    im = ax.imshow(
        H_masked,
        origin="lower",
        extent=extent,
        aspect="equal",
        norm=LogNorm(vmin=1.0, vmax=v_max)
    )

    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # Add a small colorbar per panel
    ax.get_figure().colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Hits (log scale)")


Path("hitmap").mkdir(exist_ok=True)
evs_to_plot = _first_n_evs_from_primaries(10)
print(f"[hitmap] Will plot {len(evs_to_plot)} events:", evs_to_plot)
_ckpt_path = "cnf_condN_condRing.pt"

for k, ev in enumerate(evs_to_plot, start=1):
    real_xy = np.asarray(hits[ev], dtype=np.float32)
    if real_xy.size == 0: continue
    cond_raw = prim_map[ev]
    gen_xy = sample_event(cond_raw, N=len(real_xy), ckpt_path=_ckpt_path, dev=device)
    extent = DATA_EXTENT

    fig = plt.figure(figsize=(10, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    _draw_hitmap(ax1, real_xy, extent, bins=100, vmax=None, title=f"Real event {ev} (N={len(real_xy)})")
    _draw_hitmap(ax2, gen_xy,  extent, bins=100, vmax=None, title=f"Generated (same cond)")
    fig.tight_layout(); out_path = Path("hitmap")/f"hitmap_pair_{k:02d}_ev{ev}.png"
    fig.savefig(out_path, dpi=160); plt.close(fig)
    print(f"[hitmap] saved {out_path}")

# ───────────────────── 9. DIAGNOSTIC ─────────────────────────────
@torch.no_grad()
def ring_responsibility(zxy, cond):
    lp0 = ring.log_prob(zxy, cond)
    lp1 = center.log_prob(zxy, cond)
    lp2 = noise.log_prob(zxy)
    logw = xy_prior.log_weights(cond)
    num = logw[:,0] + lp0
    den = torch.logsumexp(torch.stack([logw[:,0]+lp0, logw[:,1]+lp1, logw[:,2]+lp2], dim=-1), dim=-1)
    return torch.exp(num - den)

@torch.no_grad()
def diagnose_event(ev):
    if ev not in hits or len(hits[ev]) == 0 or ev not in prim_map:
        print(f"[diag] skip ev {ev}"); return
    pts = (np.asarray(hits[ev], np.float32) - xy_mean) / iso_std
    p = (prim_map[ev] - param_mean) / param_std
    N = len(pts)
    y0 = torch.from_numpy(np.hstack([pts, np.repeat(p[None,:], N, axis=0)])).to(device).float()
    log0 = torch.zeros(N,1, device=device)
    odefunc.compute_divergence = False
    z_traj, _ = odeint(odefunc, (y0, log0), torch.tensor([0.,1.], device=device),
                       method="rk4", options={"step_size":0.05}, atol=3e-4, rtol=3e-4)
    odefunc.compute_divergence = True
    zT = z_traj[-1]; zxy = zT[:, :2].float(); cond = zT[:, 2:].float()
    R_pred = ring.R(cond).mean().item()
    r_all  = torch.sqrt(((zxy - ring.mu(cond)) ** 2).sum(-1))
    med_all = r_all.median().item()
    gamma = ring_responsibility(zxy, cond); mask = gamma > 0.5
    if mask.any():
        med_ring = r_all[mask].median().item()
        print(f"[diag] ev {ev}: median r(all) {med_all:.3f}, median r(ring>0.5) {med_ring:.3f}, "
              f"mean R(cond) {R_pred:.3f}, Δ(ring)={med_ring-R_pred:.3f}")
    else:
        print(f"[diag] ev {ev}: median r(all) {med_all:.3f}, mean R(cond) {R_pred:.3f} "
              f"(no strong ring responsibility)")

for ev in evs_to_plot:
    diagnose_event(ev)

# ───────────────────── 10. TRAINING-EVENT COVERAGE ────────────────
print("\n──────────────── Training event coverage ────────────────")
print(f"Unique training events available: {len(ev_train)}")
print(f"Unique training events seen in batches: {len(seen_train_events)} "
      f"({len(seen_train_events)/max(1,len(ev_train)):.1%})")
print("─────────────────────────────────────────────────────────")


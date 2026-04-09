#!/usr/bin/env python3
"""
Dual-ring conditional CNF for Cherenkov detector hits.
K=2 ring slots: ring 1 (primary, always present) + ring 2 (secondary, optional).
Center blob + broad background Gaussian shared at ring-1 center.
Based on CNF_single_ring.py with all 4 bug fixes preserved.
"""

import math, os, random, re, time, sys, shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# AMP disabled — exact divergence is fragile under fp16

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

from torchdiffeq import odeint

# ────────────────────────────────────────────────────────────────────────────────
# 0) CONFIG
# ────────────────────────────────────────────────────────────────────────────────
OPTICKS_FILE = "/home/ggalgoczi/surrogate/esi-fastlight/opticks_hits_output.txt"
PRIMARIES_CSV = "primaries.csv"
COND_COLS_1BASED = [2, 3, 4, 5, 6, 7, -1]

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Dimensions
COND_DIM   = 7
H_DIM      = 16
MAX_RINGS  = 2         # primary ring + optional secondary ring
VF_HIDDEN  = 128

# Training — overnight run (~12h, no AMP)
BATCH_EVENTS = 8
EPOCHS       = 5
KL_BETA      = 1.0
LR           = 3e-4
WEIGHT_DECAY = 1e-4
KL_WARMUP    = 1
PLOT_EVERY   = 1
ENTROPY_ALPHA = 0.15   # slightly stronger entropy to prevent ring-2 collapse
MAX_TRAIN    = 70000
MAX_VAL      = 500

# Prior
MAX_HITS  = 128        # subsample hits per event for speed
RING_S    = 0.02       # radial spread of ring prior component (~5mm, matches true FWHM)
BG_SIGMA  = 0.94       # background Gaussian sigma in normalized units (~250mm / 267mm)

# ODE
ATOL = 3e-4; RTOL = 3e-4; STEP = 0.1

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD HITS & CONDITIONERS  (no single-ring filter)
# ────────────────────────────────────────────────────────────────────────────────
pat = re.compile(r"([\deE.+-]+)\s+[\deE.+-]+\s+\(([^)]+)\).*")
if not os.path.exists(OPTICKS_FILE):
    raise FileNotFoundError(f"Missing {OPTICKS_FILE}")

hits = defaultdict(list)
for ln in open(OPTICKS_FILE, "r"):
    m = pat.match(ln)
    if not m: continue
    ev = int(float(m.group(1)) // 1000)
    xy = np.fromstring(m.group(2), sep=',', dtype=np.float32)[:2]
    if xy.size == 2:
        hits[ev].append(xy)

if not os.path.exists(PRIMARIES_CSV):
    raise FileNotFoundError(f"Missing {PRIMARIES_CSV}")

def parse_primaries(path):
    prim = {}
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            parts = s.split(",")
            try: vals = [float(x) for x in parts]
            except ValueError: continue
            ev = int(vals[0])
            sel = []
            for c in COND_COLS_1BASED:
                sel.append(vals[-1] if c == -1 else vals[c - 1])
            prim[ev] = np.asarray(sel, dtype=np.float32)
    return prim

prim_map = parse_primaries(PRIMARIES_CSV)

# Overlap & split — use ALL events (no ring filter)
all_evs = sorted(set(hits.keys()) & set(prim_map.keys()))
if not all_evs:
    raise RuntimeError("No overlapping events between hits and primaries.csv")
print(f"Total events with hits & primaries: {len(all_evs)}")

rng = np.random.default_rng(SEED)
all_evs = np.array(all_evs); rng.shuffle(all_evs)
cut1, cut2 = int(.70 * len(all_evs)), int(.85 * len(all_evs))
ev_train, ev_val, ev_test = all_evs[:cut1], all_evs[cut1:cut2], all_evs[cut2:]
ev_train = ev_train[:MAX_TRAIN]
ev_val   = ev_val[:MAX_VAL]
print(f"  train {len(ev_train)} / val {len(ev_val)} / test {len(ev_test)}")

# ────────────────────────────────────────────────────────────────────────────────
# 2) NORMALIZATION & EXTENT
# ────────────────────────────────────────────────────────────────────────────────
all_xy_train = np.vstack([np.asarray(hits[e], np.float32) for e in ev_train])
xy_mean = all_xy_train.mean(0).astype(np.float32)
iso_std = float(np.sqrt(((all_xy_train - xy_mean) ** 2).sum(1).mean() / 2))

param_train = np.stack([prim_map[e] for e in ev_train], axis=0)
param_mean = param_train.mean(axis=0).astype(np.float32)
param_std  = (param_train.std(axis=0) + 1e-7).astype(np.float32)

def compute_extent(ev_ids, pad_frac=0.02, qlo=0.005, qhi=0.995):
    all_xy = np.vstack([np.asarray(hits[e], np.float32) for e in ev_ids if len(hits[e]) > 0])
    xlo, xhi = np.quantile(all_xy[:, 0], [qlo, qhi])
    ylo, yhi = np.quantile(all_xy[:, 1], [qlo, qhi])
    dx, dy = (xhi - xlo), (yhi - ylo)
    return [float(xlo - dx * pad_frac), float(xhi + dx * pad_frac),
            float(ylo - dy * pad_frac), float(yhi + dy * pad_frac)]

DATA_EXTENT = compute_extent(all_evs)

# ────────────────────────────────────────────────────────────────────────────────
# 3) DATASET
# ────────────────────────────────────────────────────────────────────────────────
class EventDataset(Dataset):
    def __init__(self, ev_ids):
        self.items = []
        for e in ev_ids:
            pts = (np.asarray(hits[e], np.float32) - xy_mean) / iso_std
            if len(pts) == 0: continue
            c = (prim_map[e] - param_mean) / param_std
            self.items.append((e, pts.astype(np.float32), c.astype(np.float32)))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        ev, xy, c = self.items[idx]
        xy = torch.from_numpy(xy)
        return {'ev': ev, 'xy': xy, 'cond': torch.from_numpy(c)}

train_ds = EventDataset(ev_train)
val_ds   = EventDataset(ev_val)

def collate_events(batch): return batch

train_loader = DataLoader(train_ds, batch_size=BATCH_EVENTS, shuffle=True,
                          drop_last=False, collate_fn=collate_events)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_EVENTS, shuffle=False,
                          drop_last=False, collate_fn=collate_events)

# ────────────────────────────────────────────────────────────────────────────────
# 4) MODEL COMPONENTS
# ────────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, dims, act=nn.SiLU):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), act()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class PriorH(nn.Module):
    """p(h | c) — prior on event latent."""
    def __init__(self, cond_dim, h_dim):
        super().__init__()
        self.mu     = MLP([cond_dim, 128, 128, h_dim])
        self.logstd = MLP([cond_dim, 128, 128, h_dim])
        nn.init.constant_(self.logstd.net[-1].bias, -0.5)
    def forward(self, c):
        return self.mu(c), self.logstd(c)


class InferenceH(nn.Module):
    """q(h | X, c) — posterior, Deep Sets aggregation over hits."""
    def __init__(self, cond_dim, h_dim):
        super().__init__()
        self.phi       = MLP([2 + cond_dim, 128, 128, 128])
        self.rho_mu    = MLP([128, 128, h_dim])
        self.rho_logstd = MLP([128, 128, h_dim])
    def forward(self, xy, c):
        c_rep = c[None, :].expand(xy.size(0), -1)
        h = self.phi(torch.cat([xy, c_rep], dim=1))
        s = h.mean(0)
        return self.rho_mu(s), self.rho_logstd(s)


class MultiCenterRadius(nn.Module):
    """Predict K ring centers and radii from (c, h)."""
    def __init__(self, cond_dim, h_dim, max_rings=MAX_RINGS):
        super().__init__()
        self.K = max_rings
        dh = cond_dim + h_dim
        self.centers = MLP([dh, 128, 128, 2 * max_rings])
        self.radii   = MLP([dh, 128, 128, max_rings])
        # Skip connection: ring center ≈ linear(pos_x, pos_y) from conditioner
        self.center_skip = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.centers.net[-1].weight)
            nn.init.zeros_(self.centers.net[-1].bias)
            nn.init.zeros_(self.radii.net[-1].weight)
            # Init so Rs = exp(0.1 * bias) ≈ 0.95 ≈ 254mm ring radius
            nn.init.constant_(self.radii.net[-1].bias, -8.7)
            # Init skip: center_norm ≈ 0.39*cn[0], 0.40*cn[1]
            self.center_skip.weight.copy_(torch.diag(torch.tensor([0.39, 0.40])))
            self.center_skip.bias.copy_(torch.tensor([0.05, 0.00]))

    def forward(self, c, h):
        ch = torch.cat([c, h], dim=-1)
        mus = self.centers(ch).view(*ch.shape[:-1], self.K, 2)   # (..., K, 2)
        Rs  = torch.exp(0.1 * self.radii(ch)).clamp_min(1e-3)    # (..., K)
        # Add skip from conditioner pos_x, pos_y (first 2 dims) to all ring centers
        skip = self.center_skip(c[..., :2]).unsqueeze(-2)         # (..., 1, 2)
        mus = mus + skip
        return mus, Rs


class VF(nn.Module):
    """Velocity field.  Input = (xy, c, h, rel_pos_to_K_centers, t)."""
    def __init__(self, cond_dim, h_dim, max_rings=MAX_RINGS, hidden=VF_HIDDEN):
        super().__init__()
        d_in = 2 + cond_dim + h_dim + 2 * max_rings + 1
        self.net = MLP([d_in, hidden, hidden, hidden, hidden // 2, 2])
        with torch.no_grad():
            nn.init.zeros_(self.net.net[-1].weight)
            nn.init.zeros_(self.net.net[-1].bias)
    def forward(self, inp):
        return self.net(inp)


class CNF_ODE(nn.Module):
    def __init__(self, vf, center_radius):
        super().__init__()
        self.vf = vf
        self.cr = center_radius
        self.K  = center_radius.K
        self.compute_divergence = True

    def forward(self, t, states):
        y, logp = states
        y = y.requires_grad_(True)

        if not self.compute_divergence:
            xy = y[:, :2]
            c  = y[:, 2:2 + COND_DIM]
            h  = y[:, 2 + COND_DIM:]
            mus, _Rs = self.cr(c, h)
            rel = (xy.unsqueeze(1) - mus).reshape(len(y), -1)
            inp = torch.cat([xy, c, h, rel, t.expand(len(y), 1)], dim=1)
            dy_xy = self.vf(inp)
            zeros = torch.zeros_like(y[:, 2:])
            dy  = torch.cat([dy_xy, zeros], dim=1)
            div = torch.zeros(len(y), 1, device=y.device, dtype=y.dtype)
            return dy, -div

        # enable_grad ensures divergence works even under outer no_grad (validation)
        with torch.enable_grad():
            xy = y[:, :2]
            c  = y[:, 2:2 + COND_DIM]
            h  = y[:, 2 + COND_DIM:]
            mus, _Rs = self.cr(c, h)
            rel = (xy.unsqueeze(1) - mus).reshape(len(y), -1)
            inp = torch.cat([xy, c, h, rel, t.expand(len(y), 1)], dim=1)
            dy_xy = self.vf(inp)
            dvx_dx = torch.autograd.grad(dy_xy[:, 0].sum(), xy, create_graph=True)[0][:, 0]
            dvy_dy = torch.autograd.grad(dy_xy[:, 1].sum(), xy, create_graph=True)[0][:, 1]
            div = (dvx_dx + dvy_dy).unsqueeze(-1)
            zeros = torch.zeros_like(y[:, 2:])
            dy = torch.cat([dy_xy, zeros], dim=1)
            return dy, -div


class MultiRingPrior(nn.Module):
    """Mixture prior: K rings + 1 center Gaussian + 1 broad background Gaussian.
    Components: [ring_0, ring_1, center, background]  (K+2 total)
    Center blob + background share ring-0 center (primary particle).
    Ring-1 (secondary) has its own center from MultiCenterRadius.
    Measured single-ring fractions: ring 64%, center 26%, background 11%.
    """
    NOISE_CAP = 0.15
    CENTER_SIGMA = 0.01  # tight Gaussian at ring center (~2.7mm)

    def __init__(self, center_radius, cond_dim, h_dim, max_rings=MAX_RINGS):
        super().__init__()
        self.cr  = center_radius
        self.K   = max_rings
        self.ring_s     = float(RING_S)
        self.bg_sigma2  = float(BG_SIGMA ** 2)
        self.center_sigma2 = float(self.CENTER_SIGMA ** 2)
        # K rings + 1 center + 1 background = K+2 components
        self.n_comp = max_rings + 2
        self.logits = MLP([cond_dim + h_dim, 128, 128, self.n_comp])
        with torch.no_grad():
            nn.init.zeros_(self.logits.net[-1].weight)
            # Init: ring-0 dominant (64%), ring-1 small (5%), center (26%), bg (5%)
            # Most events are single-ring, so ring-1 starts with low weight
            bias = torch.zeros(self.n_comp)
            bias[0] = 1.0            # ring-0 (primary) — dominant
            bias[1] = -1.5           # ring-1 (secondary) — starts small
            bias[-2] = 0.35          # center
            bias[-1] = -1.2          # background
            self.logits.net[-1].bias.copy_(bias)

    def _capped_weights(self, c, h):
        """Softmax then clamp background, renormalize."""
        x = torch.cat([c, h], -1).float()
        w = torch.softmax(self.logits(x), dim=-1)
        w_bg = w[..., -1:].clamp(max=self.NOISE_CAP)
        w_rest = w[..., :-1]
        rest_sum = w_rest.sum(-1, keepdim=True).clamp(min=1e-8)
        w_rest = w_rest * (1.0 - w_bg) / rest_sum
        return torch.cat([w_rest, w_bg], dim=-1) + 1e-8

    def log_weights(self, c, h):
        return torch.log(self._capped_weights(c, h))

    def weights(self, c, h):
        return self._capped_weights(c, h)

    def log_prob(self, zxy, c, h):
        zxy = zxy.float(); c = c.float(); h = h.float()
        mus, Rs = self.cr(c, h)            # (B,K,2), (B,K)

        # Ring log-probs — radial Gaussian around each ring
        xc = zxy.unsqueeze(1) - mus        # (B, K, 2)
        r  = torch.sqrt((xc ** 2).sum(-1) + 1e-6)  # (B, K)
        lp_rings = (
            -((r - Rs) ** 2) / (2 * self.ring_s ** 2)
            - torch.log(r + 1e-6)
            - math.log(self.ring_s * math.sqrt(2 * math.pi))
            - math.log(2 * math.pi)         # angular normalization (uniform on circle)
        )                                   # (B, K)

        # Center log-prob — tight 2D Gaussian at ring center
        center_mu = mus[:, 0, :]            # (B, 2)
        d2_ctr = ((zxy - center_mu) ** 2).sum(-1)
        lp_center = -0.5 * (
            d2_ctr / self.center_sigma2
            + 2.0 * math.log(2 * math.pi * self.center_sigma2)
        )                                   # (B,)

        # Background log-prob — broad 2D Gaussian at ring center
        lp_bg = -0.5 * (
            d2_ctr / self.bg_sigma2
            + 2.0 * math.log(2 * math.pi * self.bg_sigma2)
        )                                   # (B,)

        LP = torch.cat([lp_rings, lp_center.unsqueeze(-1), lp_bg.unsqueeze(-1)], dim=-1)
        logw = self.log_weights(c, h)
        return torch.logsumexp(LP + logw, dim=-1)

    def sample(self, c, h, N):
        """Sample N points from the mixture prior."""
        w = self.weights(c, h).squeeze(0)
        idx = torch.distributions.Categorical(w).sample((N,))
        mus, Rs = self.cr(c.float(), h.float())
        mus = mus.squeeze(0); Rs = Rs.squeeze(0)  # (K, 2), (K,)

        out = []
        for k in range(N):
            j = int(idx[k].item())
            if j < self.K:
                # Ring j — each ring uses its own center mus[j]
                theta = torch.rand((), device=c.device) * 2 * math.pi
                r = Rs[j] + self.ring_s * torch.randn((), device=c.device)
                xy = torch.stack([r * torch.cos(theta), r * torch.sin(theta)]) + mus[j]
            elif j == self.K:
                # Center — tight Gaussian at primary ring center (mus[0])
                xy = mus[0] + self.CENTER_SIGMA * torch.randn(2, device=c.device)
            else:
                # Background — broad Gaussian at primary ring center (mus[0])
                xy = mus[0] + float(BG_SIGMA) * torch.randn(2, device=c.device)
            out.append(xy)
        return torch.stack(out, dim=0), idx


def kl_diag_normals(mu_q, logstd_q, mu_p, logstd_p):
    var_q = torch.exp(2 * logstd_q); var_p = torch.exp(2 * logstd_p)
    return ((var_q + (mu_q - mu_p) ** 2) / var_p
            - 1.0 + 2 * (logstd_p - logstd_q)).sum(-1) * 0.5


class CountHead(nn.Module):
    """Predict hit count distribution from (c, h). NegBin(mu, alpha)."""
    def __init__(self, cond_dim, h_dim, init_mean=250.0):
        super().__init__()
        self.net = MLP([cond_dim + h_dim, 64, 64, 2])  # outputs: log_mu, log_alpha
        with torch.no_grad():
            nn.init.zeros_(self.net.net[-1].weight)
            self.net.net[-1].bias.copy_(torch.tensor([math.log(init_mean), math.log(0.01)]))

    def forward(self, c, h):
        out = self.net(torch.cat([c, h], -1))
        log_mu, log_alpha = out[..., 0], out[..., 1]
        mu = torch.exp(log_mu).clamp(min=1.0, max=2000.0)
        alpha = torch.exp(log_alpha).clamp(min=1e-4, max=10.0)
        return mu, alpha

    def log_prob(self, n, c, h):
        mu, alpha = self(c, h)
        r = 1.0 / alpha
        p = 1.0 / (1.0 + alpha * mu)
        return (torch.lgamma(n + r) - torch.lgamma(n + 1) - torch.lgamma(r)
                + r * torch.log(p) + n * torch.log(1 - p + 1e-8))

    def sample_count(self, c, h):
        mu, alpha = self(c, h)
        r = 1.0 / alpha
        p = 1.0 / (1.0 + alpha * mu)
        # log_prob uses "failures before r successes" convention where p = success prob.
        # PyTorch NegativeBinomial counts "successes before total_count failures",
        # so we pass probs=1-p to match our log_prob convention.
        return int(torch.distributions.NegativeBinomial(r, probs=1.0 - p).sample().clamp(min=10).item())


# ────────────────────────────────────────────────────────────────────────────────
# 5) BUILD MODEL
# ────────────────────────────────────────────────────────────────────────────────
train_mean_count = float(np.mean([len(hits[e]) for e in ev_train]))

cr      = MultiCenterRadius(COND_DIM, H_DIM).to(device)
vf      = VF(COND_DIM, H_DIM).to(device)
odef    = CNF_ODE(vf, cr).to(device)
prior_h = PriorH(COND_DIM, H_DIM).to(device)
post_h  = InferenceH(COND_DIM, H_DIM).to(device)
prior_z = MultiRingPrior(cr, COND_DIM, H_DIM).to(device)
count_head = CountHead(COND_DIM, H_DIM, init_mean=train_mean_count).to(device)

# cr is owned by prior_z, include it once
params = (list(vf.parameters()) + list(prior_h.parameters())
          + list(post_h.parameters()) + list(prior_z.parameters())
          + list(count_head.parameters()))
opt    = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

n_params = sum(p.numel() for p in params)
print(f"Model parameters: {n_params:,}")

# ────────────────────────────────────────────────────────────────────────────────
# 6) LOG-LIKELIHOOD THROUGH CNF
# ────────────────────────────────────────────────────────────────────────────────
def prior_logp_z(z):
    zxy = z[:, :2]
    c   = z[:, 2:2 + COND_DIM]
    h   = z[:, 2 + COND_DIM:]
    return prior_z.log_prob(zxy, c, h)

def flow_forward_logp(xyh):
    log0 = torch.zeros(xyh.size(0), 1, device=xyh.device)
    z_traj, logp_traj = odeint(
        odef, (xyh, log0),
        torch.tensor([0., 1.], device=xyh.device),
        method="rk4", options={"step_size": STEP}, atol=ATOL, rtol=RTOL
    )
    zT    = z_traj[-1]
    logpT = logp_traj[-1].squeeze(-1)
    return prior_logp_z(zT) - logpT, zT

# ────────────────────────────────────────────────────────────────────────────────
# 7) TRAINING / EVALUATION
# ────────────────────────────────────────────────────────────────────────────────
def fmt_time(s):
    if s < 60: return f"{s:4.1f}s"
    m, s = divmod(int(s), 60)
    if m < 60: return f"{m:02d}:{s:02d}"
    h, m = divmod(m, 60); return f"{h:d}:{m:02d}:{s:02d}"

def print_bar(prefix, i, n, elapsed, extra=""):
    cols = shutil.get_terminal_size((100, 20)).columns
    pct = i / max(1, n)
    bar_len = max(10, min(40, cols - 60))
    filled = int(bar_len * pct)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    avg = elapsed / max(1, i)
    eta = (n - i) * avg
    line = f"{prefix} [{bar}] {100*pct:5.1f}% | ETA {fmt_time(eta)} | {extra}"
    print("\r" + line[:cols - 1], end="", flush=True)

def event_elbo(batch_events, kl_beta=1.0):
    elbos, stats = [], []
    for item in batch_events:
        xy = item['xy'].to(device).float()
        c  = item['cond'].to(device).float()

        # q(h | X, c)
        mu_q, logstd_q = post_h(xy, c)
        eps = torch.randn_like(mu_q)
        h   = mu_q + torch.exp(logstd_q) * eps

        # p(h | c)
        mu_p, logstd_p = prior_h(c)
        kl = kl_diag_normals(mu_q, logstd_q, mu_p, logstd_p)

        # log p(X | c, h) via CNF
        c_rep = c[None, :].expand(xy.size(0), -1)
        h_rep = h[None, :].expand(xy.size(0), -1)
        xyh   = torch.cat([xy, c_rep, h_rep], dim=1)
        logp, _ = flow_forward_logp(xyh)
        recon = logp.sum() / 250.0  # joint set likelihood, scaled by typical N for stability

        # Count log-likelihood
        n_hits = torch.tensor(float(xy.size(0)), device=device)
        count_ll = count_head.log_prob(n_hits, c, h)

        elbo = recon - kl_beta * kl + count_ll
        # Entropy bonus on mixture weights
        w = prior_z.weights(c, h)
        entropy = -(w * torch.log(w + 1e-8)).sum()
        elbo = elbo + ENTROPY_ALPHA * entropy
        elbos.append(elbo)
        stats.append({'recon': recon.detach(), 'kl': kl.detach(),
                      'count_ll': count_ll.detach(), 'Ni': xy.size(0)})
    return torch.stack(elbos).mean(), stats

Path("progress_dual").mkdir(exist_ok=True)

@torch.no_grad()
def sample_event(cond_raw, N=None):
    """Generate hits for a given primary conditioner. N=None uses count head."""
    c = (torch.tensor(cond_raw, dtype=torch.float32, device=device)
         - torch.tensor(param_mean, device=device)) / torch.tensor(param_std, device=device)
    c = c.unsqueeze(0)

    mu_p, logstd_p = prior_h(c.squeeze(0))
    h = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
    h = h.unsqueeze(0)  # (1, H_DIM) to match c

    if N is None:
        N = count_head.sample_count(c.squeeze(0), h.squeeze(0))

    zxy, _ = prior_z.sample(c, h, N)
    c_rep = c.expand(N, -1)
    h_rep = h.expand(N, -1)
    z = torch.cat([zxy, c_rep, h_rep], dim=1)

    prev_div = odef.compute_divergence
    odef.compute_divergence = False
    x_inv, _ = odeint(
        odef, (z, torch.zeros(N, 1, device=device)),
        torch.tensor([1., 0.], device=device),
        method="rk4", options={"step_size": STEP}, atol=ATOL, rtol=RTOL
    )
    odef.compute_divergence = prev_div
    xy_norm = x_inv[-1][:, :2]
    xy = xy_norm * torch.tensor(iso_std, device=device) + torch.tensor(xy_mean, device=device)
    return xy.detach().cpu().numpy()

@torch.no_grad()
def get_ring_params(cond_raw):
    """Return predicted ring centers, radii, and weights (in data coords)."""
    c = (torch.tensor(cond_raw, dtype=torch.float32, device=device)
         - torch.tensor(param_mean, device=device)) / torch.tensor(param_std, device=device)
    c = c.unsqueeze(0)
    mu_p, logstd_p = prior_h(c.squeeze(0))
    h = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
    mus_z, Rs_z = cr(c.squeeze(0), h)      # (K,2), (K,)
    w = prior_z.weights(c.squeeze(0), h)    # (K+1,)
    # Convert z-space ring params to data coords
    mus_data = mus_z.cpu().numpy() * iso_std + xy_mean
    Rs_data  = Rs_z.cpu().numpy() * iso_std
    return mus_data, Rs_data, w.cpu().numpy()

def plot_compare(ev, k=0):
    real_xy = np.asarray(hits[ev], np.float32)
    if real_xy.size == 0: return
    cond_raw = prim_map[ev]
    gen_xy = sample_event(cond_raw)
    extent = DATA_EXTENT

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, xy, title in [(ax1, real_xy, f"Real ev {ev} (N={len(real_xy)})"),
                          (ax2, gen_xy,  "Generated")]:
        H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=100,
                                 range=[[extent[0], extent[1]],
                                        [extent[2], extent[3]]])
        ax.imshow(H.T, origin="lower", extent=extent, aspect="equal")
        ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')

    # Overlay predicted ring circles on generated plot
    try:
        mus_d, Rs_d, w = get_ring_params(cond_raw)
        for j in range(MAX_RINGS):
            if w[j] > 0.05:
                circ = plt.Circle(mus_d[j], Rs_d[j], fill=False,
                                  color='r', lw=1, ls='--', alpha=float(w[j]))
                ax2.add_patch(circ)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(Path("progress_dual") / f"cmp_{k:03d}_ev{ev}.png", dpi=150)
    plt.close(fig)

# ────────────────────────────────────────────────────────────────────────────────
# 8) TRAINING LOOP
# ────────────────────────────────────────────────────────────────────────────────
best_val = None
global_start = time.time()

for ep in range(1, EPOCHS + 1):
    # ── Train ──
    beta = min(1.0, ep / KL_WARMUP)
    odef.compute_divergence = True
    for m in [vf, cr, prior_h, post_h, prior_z, count_head]: m.train()
    total_batches = len(train_loader)
    epoch_start = time.time()
    train_elbo_acc = 0.0

    for bi, batch in enumerate(train_loader, 1):
        opt.zero_grad(set_to_none=True)
        # No AMP for ODE/divergence path — exact divergence is fragile under fp16
        elbo, stats = event_elbo(batch, kl_beta=beta)
        loss = -elbo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()

        train_elbo_acc += elbo.item()
        Ni = sum(s['Ni'] for s in stats)
        recon_mean = float(torch.stack([s['recon'] for s in stats]).mean())
        kl_mean    = float(torch.stack([s['kl']    for s in stats]).mean())
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed = time.time() - epoch_start
        extra = (f"ELBO {elbo.item():.4f} | recon {recon_mean:.4f} "
                 f"| KL {kl_mean:.4f} | {Ni:5d} hits")
        print_bar(f"Ep {ep:02d}/{EPOCHS} Tr", bi, total_batches, elapsed, extra)
    print()

    # ── Validate ──
    for m in [vf, cr, prior_h, post_h, prior_z, count_head]: m.eval()
    val_elbos = []
    val_start = time.time()
    with torch.no_grad():
        for vi, batch in enumerate(val_loader, 1):
            elbo, _ = event_elbo(batch, kl_beta=1.0)
            val_elbos.append(elbo.item())
            if device.type == "cuda": torch.cuda.synchronize()
            elapsed = time.time() - val_start
            print_bar(f"Ep {ep:02d}/{EPOCHS} Va", vi, len(val_loader), elapsed,
                      f"ELBO {elbo.item():.4f}")
    if val_loader: print()

    val_mean = float(np.mean(val_elbos)) if val_elbos else float('nan')
    train_mean = train_elbo_acc / max(1, total_batches)
    epoch_time = time.time() - epoch_start
    print(f"[Ep {ep:02d}] train {train_mean:.4f} | val {val_mean:.4f} "
          f"| beta {beta:.2f} | {fmt_time(epoch_time)} | total {fmt_time(time.time()-global_start)}")

    # ── Checkpoint best ──
    if best_val is None or val_mean > best_val:
        best_val = val_mean
        torch.save({
            'vf':      vf.state_dict(),
            'cr':      cr.state_dict(),
            'prior_h': prior_h.state_dict(),
            'post_h':  post_h.state_dict(),
            'prior_z': prior_z.state_dict(),
            'count_head': count_head.state_dict(),
            'xy_mean': xy_mean, 'iso_std': iso_std,
            'param_mean': param_mean, 'param_std': param_std,
            'train_mean_count': train_mean_count,
            'config': dict(COND_DIM=COND_DIM, H_DIM=H_DIM, MAX_RINGS=MAX_RINGS,
                           VF_HIDDEN=VF_HIDDEN, RING_S=RING_S, BG_SIGMA=BG_SIGMA,
                           ATOL=ATOL, RTOL=RTOL, STEP=STEP),
        }, 'dual_cnf_best.pt')
        print(f"  -> saved best checkpoint (val {val_mean:.4f})")

    # ── Plots ──
    if ep % PLOT_EVERY == 0 and len(ev_val) > 0:
        plot_compare(int(ev_val[0]), k=ep)
        # Also plot a known multi-ring event (ev 94152 has 2 clear rings)
        if 94152 in prim_map and 94152 in hits:
            plot_compare(94152, k=ep + 100)

# ────────────────────────────────────────────────────────────────────────────────
# 9) FINAL SAVE & PLOTS
# ────────────────────────────────────────────────────────────────────────────────
torch.save({
    'vf':      vf.state_dict(),
    'cr':      cr.state_dict(),
    'prior_h': prior_h.state_dict(),
    'post_h':  post_h.state_dict(),
    'prior_z': prior_z.state_dict(),
    'count_head': count_head.state_dict(),
    'xy_mean': xy_mean, 'iso_std': iso_std,
    'param_mean': param_mean, 'param_std': param_std,
    'train_mean_count': train_mean_count,
    'config': dict(COND_DIM=COND_DIM, H_DIM=H_DIM, MAX_RINGS=MAX_RINGS,
                   VF_HIDDEN=VF_HIDDEN, RING_S=RING_S, BG_SIGMA=BG_SIGMA,
                   ATOL=ATOL, RTOL=RTOL, STEP=STEP),
}, 'dual_cnf_final.pt')
print("Saved dual_cnf_final.pt")

for i, ev in enumerate(ev_val[:min(5, len(ev_val))]):
    plot_compare(int(ev), k=1000 + i)
print("Done.")

#!/usr/bin/env python3
"""
Generate synthetic events from a trained T3_single checkpoint.

Usage:
  /usr/bin/python3.9 generate_CNF_single_ring.py                          # 1000 events, learned NegBin counts
  /usr/bin/python3.9 generate_CNF_single_ring.py -n 5000 --hits 300       # 5000 events, fixed 300 hits
  /usr/bin/python3.9 generate_CNF_single_ring.py -n 50000 -o events.npy   # 50k events → events.npy
  /usr/bin/python3.9 generate_CNF_single_ring.py --save-plots 20          # save 20 hitmap PNGs
"""

import argparse, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1) MODEL DEFINITIONS (must match T3_single.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
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
    def __init__(self, cond_dim, h_dim):
        super().__init__()
        self.mu     = MLP([cond_dim, 128, 128, h_dim])
        self.logstd = MLP([cond_dim, 128, 128, h_dim])
    def forward(self, c):
        return self.mu(c), self.logstd(c)


class MultiCenterRadius(nn.Module):
    def __init__(self, cond_dim, h_dim, max_rings=1):
        super().__init__()
        self.K = max_rings
        dh = cond_dim + h_dim
        self.centers = MLP([dh, 128, 128, 2 * max_rings])
        self.radii   = MLP([dh, 128, 128, max_rings])
        self.center_skip = nn.Linear(2, 2, bias=True)

    def forward(self, c, h):
        ch = torch.cat([c, h], dim=-1)
        mus = self.centers(ch).view(*ch.shape[:-1], self.K, 2)
        Rs  = torch.exp(0.1 * self.radii(ch)).clamp_min(1e-3)
        skip = self.center_skip(c[..., :2]).unsqueeze(-2)
        mus = mus + skip
        return mus, Rs


class VF(nn.Module):
    def __init__(self, cond_dim, h_dim, max_rings=1, hidden=128):
        super().__init__()
        d_in = 2 + cond_dim + h_dim + 2 * max_rings + 1
        self.net = MLP([d_in, hidden, hidden, hidden, hidden // 2, 2])
    def forward(self, inp):
        return self.net(inp)


class CNF_ODE(nn.Module):
    def __init__(self, vf, center_radius, cond_dim, h_dim):
        super().__init__()
        self.vf = vf
        self.cr = center_radius
        self.K  = center_radius.K
        self.cond_dim = cond_dim
        self.h_dim = h_dim

    def forward(self, t, states):
        y, logp = states
        xy = y[:, :2]
        c  = y[:, 2:2 + self.cond_dim]
        h  = y[:, 2 + self.cond_dim:]
        mus, _Rs = self.cr(c, h)
        rel = (xy.unsqueeze(1) - mus).reshape(len(y), -1)
        inp = torch.cat([xy, c, h, rel, t.expand(len(y), 1)], dim=1)
        dy_xy = self.vf(inp)
        zeros = torch.zeros_like(y[:, 2:])
        dy  = torch.cat([dy_xy, zeros], dim=1)
        div = torch.zeros(len(y), 1, device=y.device, dtype=y.dtype)
        return dy, -div


class MultiRingPrior(nn.Module):
    def __init__(self, center_radius, cond_dim, h_dim, max_rings=1,
                 ring_s=0.02, bg_sigma=0.94, center_sigma=0.01):
        super().__init__()
        self.cr  = center_radius
        self.K   = max_rings
        self.ring_s = ring_s
        self.bg_sigma = bg_sigma
        self.center_sigma = center_sigma
        self.n_comp = max_rings + 2
        self.logits = MLP([cond_dim + h_dim, 128, 128, self.n_comp])
        self.NOISE_CAP = 0.15

    def _capped_weights(self, c, h):
        x = torch.cat([c, h], -1).float()
        w = torch.softmax(self.logits(x), dim=-1)
        w_bg = w[..., -1:].clamp(max=self.NOISE_CAP)
        w_rest = w[..., :-1]
        rest_sum = w_rest.sum(-1, keepdim=True).clamp(min=1e-8)
        w_rest = w_rest * (1.0 - w_bg) / rest_sum
        return torch.cat([w_rest, w_bg], dim=-1) + 1e-8

    def weights(self, c, h):
        return self._capped_weights(c, h)

    def sample(self, c, h, N):
        w = self.weights(c, h).squeeze(0)
        idx = torch.distributions.Categorical(w).sample((N,))
        mus, Rs = self.cr(c.float(), h.float())
        mus = mus.squeeze(0); Rs = Rs.squeeze(0)

        out = []
        for k in range(N):
            j = int(idx[k].item())
            if j < self.K:
                theta = torch.rand((), device=c.device) * 2 * math.pi
                r = Rs[j] + self.ring_s * torch.randn((), device=c.device)
                xy = torch.stack([r * torch.cos(theta), r * torch.sin(theta)]) + mus[j]
            elif j == self.K:
                xy = mus[0] + self.center_sigma * torch.randn(2, device=c.device)
            else:
                xy = mus[0] + self.bg_sigma * torch.randn(2, device=c.device)
            out.append(xy)
        return torch.stack(out, dim=0), idx


class CountHead(nn.Module):
    """Predict hit count distribution from (c, h). NegBin(mu, alpha)."""
    def __init__(self, cond_dim, h_dim, init_mean=250.0):
        super().__init__()
        self.net = MLP([cond_dim + h_dim, 64, 64, 2])

    def forward(self, c, h):
        out = self.net(torch.cat([c, h], -1))
        log_mu, log_alpha = out[..., 0], out[..., 1]
        mu = torch.exp(log_mu).clamp(min=1.0, max=2000.0)
        alpha = torch.exp(log_alpha).clamp(min=1e-4, max=10.0)
        return mu, alpha

    def sample_count(self, c, h):
        mu, alpha = self(c, h)
        r = 1.0 / alpha
        p = 1.0 / (1.0 + alpha * mu)
        # PyTorch NegBin counts successes before failures; our log_prob uses
        # failures-before-successes convention, so pass probs=1-p
        return int(torch.distributions.NegativeBinomial(r, probs=1.0 - p).sample().clamp(min=10).item())


# ─────────────────────────────────────────────────────────────────────────────
# 2) LOAD CHECKPOINT & BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']

    cond_dim  = cfg['COND_DIM']
    h_dim     = cfg['H_DIM']
    max_rings = cfg['MAX_RINGS']
    vf_hidden = cfg['VF_HIDDEN']
    ring_s    = cfg['RING_S']
    bg_sigma  = cfg['BG_SIGMA']
    step      = cfg['STEP']
    atol      = cfg['ATOL']
    rtol      = cfg['RTOL']
    train_mean_count = ckpt.get('train_mean_count', 250.0)

    cr      = MultiCenterRadius(cond_dim, h_dim, max_rings).to(device)
    vf      = VF(cond_dim, h_dim, max_rings, vf_hidden).to(device)
    odef    = CNF_ODE(vf, cr, cond_dim, h_dim).to(device)
    prior_h = PriorH(cond_dim, h_dim).to(device)
    prior_z = MultiRingPrior(cr, cond_dim, h_dim, max_rings, ring_s, bg_sigma).to(device)
    count_h = CountHead(cond_dim, h_dim, init_mean=train_mean_count).to(device)

    vf.load_state_dict(ckpt['vf'])
    cr.load_state_dict(ckpt['cr'])
    prior_h.load_state_dict(ckpt['prior_h'])
    prior_z.load_state_dict(ckpt['prior_z'])
    if 'count_head' in ckpt:
        count_h.load_state_dict(ckpt['count_head'])

    for m in [vf, cr, prior_h, prior_z, count_h]:
        m.eval()

    norm = {
        'xy_mean':    ckpt['xy_mean'],
        'iso_std':    ckpt['iso_std'],
        'param_mean': ckpt['param_mean'],
        'param_std':  ckpt['param_std'],
    }
    ode_cfg = {'step': step, 'atol': atol, 'rtol': rtol}

    return odef, prior_h, prior_z, count_h, norm, ode_cfg, cond_dim


# ─────────────────────────────────────────────────────────────────────────────
# 3) PARSE PRIMARIES
# ─────────────────────────────────────────────────────────────────────────────
COND_COLS_1BASED = [2, 3, 4, 5, 6, 7, -1]

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
            sel = [vals[-1] if c == -1 else vals[c - 1] for c in COND_COLS_1BASED]
            prim[ev] = np.asarray(sel, dtype=np.float32)
    return prim


# ─────────────────────────────────────────────────────────────────────────────
# 5) DETECTOR MASK & GENERATION
# ─────────────────────────────────────────────────────────────────────────────
# Dead zone geometry (measured from 25M hits):
#   - Central cross extends ~183mm from origin along each axis
#   - Horizontal arm: |y| < 60mm AND |x| < 183mm
#   - Vertical arm:   |x| < 60mm AND |y| < 183mm
#   - Outside detector: r > 553mm
CROSS_ARM_W   = 60.0   # mm, half-width of each arm
CROSS_ARM_LEN = 183.0  # mm, half-length of each arm (from origin)
DET_RADIUS    = 553.0  # mm

def detector_mask(xy_mm):
    """Return boolean mask: True = valid hit (keep), False = in dead zone or outside."""
    x, y = xy_mm[:, 0], xy_mm[:, 1]
    ax, ay = np.abs(x), np.abs(y)
    # Cross: two arms intersecting at center, each ~183mm long, ~60mm wide
    horiz_arm = (ay < CROSS_ARM_W) & (ax < CROSS_ARM_LEN)
    vert_arm  = (ax < CROSS_ARM_W) & (ay < CROSS_ARM_LEN)
    in_cross  = horiz_arm | vert_arm
    outside   = (x**2 + y**2) > DET_RADIUS**2
    return ~in_cross & ~outside

@torch.no_grad()
def generate_event(odef, prior_h, prior_z, count_h, cond_raw, norm, ode_cfg, N, device,
                   apply_mask=True, oversample=1.3, max_retries=5):
    """Generate N hits for one conditioner vector. Returns (N, 2) numpy array in mm.
    If N is None, uses count head to predict hit count.
    If apply_mask, rejects hits in dead zone / outside detector and resamples."""
    param_mean = torch.tensor(norm['param_mean'], device=device)
    param_std  = torch.tensor(norm['param_std'], device=device)
    xy_mean    = torch.tensor(norm['xy_mean'], device=device)
    iso_std    = norm['iso_std']

    c = (torch.tensor(cond_raw, dtype=torch.float32, device=device) - param_mean) / param_std
    c = c.unsqueeze(0)

    mu_p, logstd_p = prior_h(c.squeeze(0))
    h = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
    h = h.unsqueeze(0)

    if N is None:
        N = count_h.sample_count(c.squeeze(0), h.squeeze(0))

    if not apply_mask:
        # Original path: generate exactly N, no filtering
        zxy, comp_idx = prior_z.sample(c, h, N)
        c_rep = c.expand(N, -1)
        h_rep = h.expand(N, -1)
        z = torch.cat([zxy, c_rep, h_rep], dim=1)
        x_inv, _ = odeint(
            odef, (z, torch.zeros(N, 1, device=device)),
            torch.tensor([1., 0.], device=device),
            method="rk4", options={"step_size": ode_cfg['step']},
            atol=ode_cfg['atol'], rtol=ode_cfg['rtol']
        )
        xy = x_inv[-1][:, :2] * iso_std + xy_mean
        return xy.cpu().numpy(), comp_idx.cpu().numpy()

    # Generate with oversampling + rejection loop
    target = N
    collected = []
    for _ in range(max_retries):
        n_gen = max(int(target * oversample), target + 10)
        zxy, comp_idx = prior_z.sample(c, h, n_gen)
        c_rep = c.expand(n_gen, -1)
        h_rep = h.expand(n_gen, -1)
        z = torch.cat([zxy, c_rep, h_rep], dim=1)
        x_inv, _ = odeint(
            odef, (z, torch.zeros(n_gen, 1, device=device)),
            torch.tensor([1., 0.], device=device),
            method="rk4", options={"step_size": ode_cfg['step']},
            atol=ode_cfg['atol'], rtol=ode_cfg['rtol']
        )
        xy_mm = (x_inv[-1][:, :2] * iso_std + xy_mean).cpu().numpy()
        mask = detector_mask(xy_mm)
        collected.append(xy_mm[mask])
        total = sum(len(c) for c in collected)
        if total >= target:
            break
        target -= total  # still need this many
        oversample = 1.5  # increase for next retry

    all_xy = np.concatenate(collected, axis=0)[:N]
    return all_xy, np.zeros(len(all_xy), dtype=np.int64)


def save_hitmap(xy, ev_id, out_dir, extent, bins=100):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins,
                              range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    H_log = np.log10(H + 1)
    ax.imshow(H_log.T, origin="lower", extent=extent, aspect="equal", cmap="inferno")
    ax.set_title(f"Generated ev {ev_id} (N={len(xy)})")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    fig.tight_layout()
    fig.savefig(out_dir / f"gen_ev{ev_id}.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6) MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Generate events from T3_single checkpoint")
    p.add_argument("-m", "--model", default="single_cnf_best.pt", help="Checkpoint path")
    p.add_argument("-p", "--primaries", default="primaries.csv")
    p.add_argument("-n", "--num-events", type=int, default=1000)
    p.add_argument("-o", "--output", default="generated_events.npy",
                   help="Output .npy file (ragged array of (N_i, 2) per event)")
    p.add_argument("--hits", type=int, default=0,
                   help="Fixed hit count per event (0 = sample from empirical distribution)")
    p.add_argument("--save-plots", type=int, default=10,
                   help="Number of hitmap PNGs to save (0 = none)")
    p.add_argument("--plot-dir", default="generated_hitmaps")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model} ...")
    odef, prior_h_net, prior_z_net, count_h_net, norm, ode_cfg, cond_dim = load_model(args.model, device)

    print(f"Loading primaries from {args.primaries} ...")
    prim_map = parse_primaries(args.primaries)
    ev_ids = sorted(prim_map.keys())
    print(f"  {len(ev_ids)} primaries available")

    # Hit count mode
    use_count_head = (args.hits == 0)
    if args.hits > 0:
        print(f"Using fixed hit count: {args.hits}")
    else:
        print("Using learned count head (NegBin)")

    # Detector extent for plots
    extent = [-1150, 1150, -1150, 1150]

    # Output
    plot_dir = Path(args.plot_dir)
    if args.save_plots > 0:
        plot_dir.mkdir(exist_ok=True, parents=True)

    # Select which primaries to use
    n_events = min(args.num_events, len(ev_ids))
    selected = np.random.choice(ev_ids, size=n_events, replace=(n_events > len(ev_ids)))

    print(f"\nGenerating {n_events} events on {device} ...")
    all_events = []
    all_conds  = []
    t0 = time.time()

    for i, ev in enumerate(selected):
        N = args.hits if args.hits > 0 else None
        xy, comp = generate_event(odef, prior_h_net, prior_z_net, count_h_net,
                                  prim_map[ev], norm, ode_cfg, N, device)
        all_events.append(xy.astype(np.float32))
        all_conds.append(prim_map[ev])

        if i < args.save_plots:
            save_hitmap(xy, ev, plot_dir, extent)

        if (i + 1) % 100 == 0 or i == n_events - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_events - i - 1) / rate
            print(f"  [{i+1:5d}/{n_events}] {rate:.1f} ev/s | ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {n_events} events in {elapsed:.1f}s ({n_events/elapsed:.1f} ev/s)")

    # Save as object array (ragged — different N per event)
    np.save(args.output, np.array(all_events, dtype=object), allow_pickle=True)
    np.save(args.output.replace('.npy', '_conds.npy'),
            np.array(all_conds, dtype=np.float32))
    print(f"Saved {args.output} and {args.output.replace('.npy', '_conds.npy')}")

    if args.save_plots > 0:
        print(f"Saved {min(args.save_plots, n_events)} hitmaps to {plot_dir}/")


if __name__ == "__main__":
    main()

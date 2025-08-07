#!/usr/bin/env python3
# ---------------------------------------------------------------
# gen_masked_symmetric_best.py
#
# Super-fast CNF sampler for one GPU (RTX-2000 Ada tuned)
#   – Midpoint integrator, one global step (h = 1 / n_steps)
#   – Half-precision weights, FP32 activations
#   – No autocast, huge micro-batch
#
# Accuracy: n_steps = 1 works for most flows; bump to 2-3 if needed.
# Speed   : 13.7 M hits in ~7 s on an Ada (≈3× baseline).
# ---------------------------------------------------------------
import math, time, argparse, numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path

torch.set_float32_matmul_precision("high")

# ─────────────── CLI ────────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-c","--ckpt",default="cnf_condN_iso.pt")
ap.add_argument("-n","--num-events",type=int,default=50_000)
ap.add_argument("-o","--output",default="events.npz")
ap.add_argument("--chunks",type=int,default=2)
ap.add_argument("--hits-per-batch",type=int,default=4_000_000)
ap.add_argument("--n-steps",type=int,default=1,help="# Midpoint steps (≥1)")
ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args();  dev = torch.device(args.device)

if args.n_steps < 1:
    raise SystemExit("  --n-steps must be ≥ 1")

# ─────────────── 1. LOAD FLOW (FP16 weights) ───────────────────
ck       = torch.load(args.ckpt, map_location=dev, weights_only=False)
xy_mean  = torch.as_tensor(ck["xy_mean"], device=dev)
iso_std  = float(ck["iso_std"])
iso_std_t, xy_mean_t = torch.tensor(iso_std,device=dev), xy_mean

HIDDEN = 128
class VF(torch.nn.Module):
    def __init__(self,d=3,h=HIDDEN):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d+1,h), torch.nn.SiLU(),
            torch.nn.Linear(h,h),   torch.nn.SiLU(),
            torch.nn.Linear(h,h//2),torch.nn.SiLU(),
            torch.nn.Linear(h//2,2))
    def forward(self,t,y):
        return self.net(torch.cat([y, t.expand(len(y),1)],1))

vf = VF().to(dev).eval(); vf.load_state_dict(ck["vf_state_dict"])
for m in vf.modules():                      # weights → FP16
    if isinstance(m, torch.nn.Linear):
        m.weight.data = m.weight.half()
        m.bias.data   = m.bias.half()

class Sampler(torch.nn.Module):             # 2-D vf → 3-D with zeros
    def __init__(self,vf): super().__init__(); self.vf=vf
    def forward(self,t,y):
        y16 = y.half(); t16 = t.half()
        out = self.vf(t16,y16).float()
        return torch.cat([out, torch.zeros_like(y[:,2:3])], -1)

odef = Sampler(vf)

# ─────────────── 2. FAST MIDPOINT INTEGRATOR ───────────────────
@torch.no_grad()

def cnf_inverse(z0):
    h = 1.0 / args.n_steps
    t = 0.0
    z = z0
    for _ in range(args.n_steps):
        t0 = torch.tensor(t,  device=dev)          # ← add
        k1 = odef(t0, z)

        z_mid = z + 0.5 * h * k1
        tmid = torch.tensor(t + 0.5*h, device=dev) # ← add
        k2  = odef(tmid, z_mid)

        z   = z + h * k2
        t  += h
    return z

# ─────────────── 3. PRIOR IN LATENT Z ──────────────────────────
class RingPrior(torch.distributions.Distribution):
    arg_constraints, has_rsample = {}, False
    def __init__(self,R=1,s=.05): super().__init__(); self.R,self.s = R,s
    def sample(self,shape=torch.Size(), device="cpu"):
        θ = torch.rand(shape, device=device)*2*math.pi
        r = self.R + self.s*torch.randn(shape, device=device)
        return torch.stack([r*torch.cos(θ), r*torch.sin(θ)], -1)

devc = dev.type
ring   = RingPrior(1.0,0.05)
center = torch.distributions.MultivariateNormal(
            torch.zeros(2,device=dev), torch.eye(2,device=dev)*0.05**2)
noise  = torch.distributions.MultivariateNormal(
            torch.zeros(2,device=dev), torch.eye(2,device=dev)*3.0**2)
mix_w  = torch.tensor([.60,.25,.15],device=dev)
cat    = torch.distributions.Categorical(mix_w)

def sample_xy_prior(n:int)->torch.Tensor:
    idx = cat.sample((n,))
    out = torch.empty(n,2,device=dev)
    m=idx==0; out[m]=ring.sample((m.sum(),),device=dev)
    m=idx==1; out[m]=center.sample((m.sum(),))
    m=idx==2; out[m]=noise.sample((m.sum(),))
    return out

# ─────────────── 4. HARD MASK (unchanged logic) ────────────────
STEP = 123.0
Y_MAX = 4*STEP + STEP/2.
X_MAX = 1_000.0
Y_L1, Y_L2 = 2.5*STEP, 3.5*STEP
X_L1, X_L2, X_L3 = 4.5*STEP, 3.5*STEP, 2.5*STEP
BH, BF = 0.5*STEP, 1.5*STEP
@torch.no_grad()
def valid_mask_mm(x_mm:torch.Tensor, y_mm:torch.Tensor)->torch.Tensor:
    abs_x, abs_y = x_mm.abs(), y_mm.abs()
    outer = (
        ((abs_y >  Y_L2) & (abs_x <= X_L3)) |
        ((abs_y >  Y_L1) & (abs_y <= Y_L2) & (abs_x <= X_L2)) |
        ((abs_y <= Y_L1) & (abs_x <= X_L1))
    )
    centre = (abs_x <= BH) & (abs_y <= BH)
    up     = (abs_x <= BH) & (y_mm >=  BH) & (y_mm <=  BF)
    down   = (abs_x <= BH) & (y_mm <= -BH) & (y_mm >= -BF)
    right  = (abs_y <= BH) & (x_mm >=  BH) & (x_mm <=  BF)
    left   = (abs_y <= BH) & (x_mm <= -BH) & (x_mm >= -BF)
    inner  = centre | up | down | left | right
    return outer & ~inner & (abs_x <= X_MAX) & (abs_y <= Y_MAX)

# ─────────────── 5. DRAW WITH HUGE MICRO-BATCH ────────────────
@torch.no_grad()
def draw_inside_mask(n:int)->torch.Tensor:
    out, done, B = torch.empty(n,2,device=dev), 0, args.hits_per_batch
    while done < n:
        z_xy = sample_xy_prior(B)
        z    = torch.cat([z_xy, torch.zeros(B,1,device=dev)],1)
        x    = cnf_inverse(z)[:,:2]
        mm   = x*iso_std_t + xy_mean_t
        m    = valid_mask_mm(mm[:,0], mm[:,1])
        k    = min(int(m.sum().item()), n-done)
        if k:
            out[done:done+k] = x[m][:k]
            done += k
    return out

# ─────────────── 6. MULTIPLICITIES & FILE I/O ─────────────────
counts = np.load("counts.npy")
rng    = np.random.default_rng(123)
N_ev   = args.num_events
ev_N   = rng.choice(counts, size=N_ev, replace=True).astype(np.int32)
idx    = np.zeros(N_ev+1,np.int64); np.cumsum(ev_N, out=idx[1:])
tot_hits = int(idx[-1])
print(f"Total hits to generate: {tot_hits:,}")

binfile = "events_points.bin"
pts_mm  = np.lib.format.open_memmap(binfile,"w+",dtype="float32",shape=(tot_hits,2))

# ─────────────── 7. GENERATION LOOP ───────────────────────────
torch.inference_mode().__enter__()
t0 = time.perf_counter()

for c,ev_slice in enumerate(np.array_split(np.arange(N_ev),args.chunks),1):
    if ev_slice.size==0: continue
    s,e   = int(idx[ev_slice[0]]), int(idx[ev_slice[-1]+1])
    this_hits = e - s
    print(f"chunk {c}/{args.chunks}: {len(ev_slice)} ev, {this_hits:,} hits")

    xy_norm = draw_inside_mask(this_hits)
    host    = torch.empty_like(xy_norm, device='cpu', pin_memory=True)
    host.copy_(xy_norm*iso_std + xy_mean, non_blocking=True)
    pts_mm[s:e] = host.numpy().astype("float32")

torch.cuda.synchronize() if dev.type=="cuda" else None
dt = time.perf_counter()-t0
pts_mm.flush(); np.savez_compressed(args.output, points_file=binfile, idx=idx)
print(f"✓ wrote {args.output}   ({dt:0.1f} s, {tot_hits/dt:,.0f} hits/s)")

# ─────────────── 8. QUICK-LOOK PNGs ───────────────────────────
Path("synth").mkdir(exist_ok=True)
for k in range(10):
    ev  = rng.integers(0, N_ev)
    pts = pts_mm[idx[ev]:idx[ev+1]]
    fig,ax = plt.subplots(figsize=(4,4))
    ax.scatter(pts[:,0],pts[:,1],s=4,lw=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Synth ev {ev}  N={len(pts)}")
    fig.tight_layout()
    fig.savefig(Path("synth")/f"synth_sample_{k}.png",dpi=150)
    plt.close(fig)

print("quick-look PNGs → synth/   (mask symmetric + central cross empty)")

#!/usr/bin/env python3
# ---------------------------------------------------------------
# Generate events with the isotropically-normalised conditional CNF
# – integrates with lightweight SamplerODE (no log-p)
# – keeps GPU RAM low via --hits-per-batch
# – saves events.npz  +  synth/synth_sample_*.png
# ---------------------------------------------------------------
import math, time, argparse, numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path
from torchdiffeq import odeint

# ───────────── CLI ────────────────────────────────────────────────
p=argparse.ArgumentParser()
p.add_argument("-c","--ckpt",default="cnf_condN_iso.pt")
p.add_argument("-n","--num-events",type=int,default=50_000)
p.add_argument("-o","--output",default="events.npz")
p.add_argument("--chunks",type=int,default=3)
p.add_argument("--hits-per-batch",type=int,default=250_000)
p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
args=p.parse_args(); device=torch.device(args.device)

# ───────────── LOAD MODEL & STATS ────────────────────────────────
ckpt=torch.load(args.ckpt,map_location=device,weights_only=False)
xy_mean=torch.as_tensor(ckpt["xy_mean"],device=device)
iso_std=float(ckpt["iso_std"])
logN_mu,logN_std=ckpt["logN_mu"],ckpt["logN_std"]

HIDDEN=128
class VF(torch.nn.Module):
    def __init__(self,d=3,h=HIDDEN):
        super().__init__()
        self.net=torch.nn.Sequential(
            torch.nn.Linear(d+1,h),torch.nn.SiLU(),
            torch.nn.Linear(h,h),  torch.nn.SiLU(),
            torch.nn.Linear(h,h//2),torch.nn.SiLU(),
            torch.nn.Linear(h//2,2))
    def forward(self,t,y):
        return self.net(torch.cat([y,t.expand(len(y),1)],1))

class SamplerODE(torch.nn.Module):
    def __init__(self,vf): super().__init__(); self.vf=vf
    def forward(self,t,y):
        dy_xy=self.vf(t,y)
        return torch.cat([dy_xy, torch.zeros_like(y[:,2:3])], -1)

vf=VF().to(device).eval(); vf.load_state_dict(ckpt["vf_state_dict"])
odef=SamplerODE(vf)

# prior (same as training, isotropic space)
class RingPrior(torch.distributions.Distribution):
    arg_constraints,has_rsample,EPS={},False,1e-6
    def __init__(self,R=1,s=.05,dev="cpu"):
        super().__init__(); self.R,self.s,self.dev=R,s,dev
    def sample(self,shape=torch.Size()):
        θ=torch.rand(shape,device=self.dev)*2*math.pi
        r=self.R+self.s*torch.randn(shape,device=self.dev)
        return torch.stack([r*torch.cos(θ),r*torch.sin(θ)],-1)
ring_R=1.0                                   # radius ~1 after iso norm
ring=RingPrior(ring_R,.05,device)
center=torch.distributions.MultivariateNormal(torch.zeros(2,device=device),
           torch.eye(2,device=device)*0.05**2)
noise=torch.distributions.MultivariateNormal(torch.zeros(2,device=device),
           torch.eye(2,device=device)*3.0**2)
mix_w=torch.tensor([.60,.25,.15],device=device)
cat   =torch.distributions.Categorical(mix_w)
class MixXY(torch.distributions.Distribution):
    arg_constraints,has_rsample={},False
    def __init__(self,comps,cat): super().__init__(); self.c,self.cat=comps,cat
    def sample(self,shape=torch.Size()):
        N=int(torch.tensor(shape).prod()) or 1
        idx=self.cat.sample((N,))
        out=torch.empty(N,2,device=idx.device)
        for i,comp in enumerate(self.c):
            m=idx==i
            if m.any(): out[m]=comp.sample((m.sum(),))
        return out.reshape(*shape,2)
xy_prior=MixXY([ring,center,noise],cat)

# ───────────── MULTIPLICITIES & STORAGE ───────────────────────────
counts=np.load("counts.npy")
rng=np.random.default_rng(123)
N_ev=args.num_events
ev_N=rng.choice(counts,size=N_ev,replace=True).astype(np.int32)
idx=np.zeros(N_ev+1,dtype=np.int64); np.cumsum(ev_N,out=idx[1:])
tot_hits=int(idx[-1]); print(f"Total hits: {tot_hits:,}")
bin_file="events_points.bin"
out_mm=np.lib.format.open_memmap(bin_file,"w+",dtype="float32",shape=(tot_hits,2))

# ───────────── GENERATION LOOP ────────────────────────────────────
torch.set_grad_enabled(False); t0=time.perf_counter()
for c,ev_slice in enumerate(np.array_split(np.arange(N_ev),args.chunks),1):
    if not len(ev_slice): continue
    start,end=int(idx[ev_slice[0]]),int(idx[ev_slice[-1]+1])
    off=start
    print(f"chunk {c}/{args.chunks}: events {ev_slice[0]}–{ev_slice[-1]}")
    while off<end:
        budget=min(args.hits_per_batch,end-off)
        ev= np.searchsorted(idx,off,side="right")-1
        collected=0; z_parts=[]
        while collected<budget and ev<=ev_slice[-1]:
            N=int(ev_N[ev]); take=min(N, budget-collected)
            n_feat=(math.log(N)-logN_mu)/logN_std
            z_xy=xy_prior.sample((take,)).to(device)
            z_n=torch.full((take,1),n_feat,device=device)
            z_parts.append(torch.cat([z_xy,z_n],1))
            collected+=take; ev+=1
        z_batch=torch.cat(z_parts,0)
        x_inv=odeint(odef,z_batch,
                     torch.tensor([1.,0.],device=device),
                     method="rk4",options={"step_size":0.05})
        pts=(x_inv[-1][:,:2]*iso_std+xy_mean)\
               .detach().cpu().numpy().astype("float32")
        out_mm[off:off+collected]=pts
        off+=collected
        del z_batch,x_inv,pts
        torch.cuda.empty_cache()
dt=time.perf_counter()-t0
out_mm.flush(); np.savez_compressed(args.output,points_file=bin_file,idx=idx)
print(f"Saved {args.output}  |  {dt:0.2f} s   ({tot_hits/dt:,.0f} pts/s)")

# ───────────── QUICK-LOOK PNGs (equal aspect) ─────────────────────
Path("synth").mkdir(exist_ok=True)
for k in range(10):
    ev=rng.integers(0,N_ev)
    pts=out_mm[idx[ev]:idx[ev+1]]
    fig,ax=plt.subplots(figsize=(4,4))
    ax.scatter(pts[:,0],pts[:,1],s=4,lw=0)
    ax.set_aspect("equal","box")
    ax.set_title(f"Synth ev {ev}  N={len(pts)}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(Path("synth")/f"synth_sample_{k}.png",dpi=150)
    plt.close(fig)
print("✓ quick-look PNGs saved to synth/")

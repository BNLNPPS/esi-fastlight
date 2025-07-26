import re
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt
from torchdyn.models import CNF
from torchdiffeq import odeint
import os

# ---- 1. Load and parse data ----
FILENAME = "opticks_hits_output.txt"
event_positions = defaultdict(list)
with open(FILENAME) as f:
    for line in f:
        match = re.match(
            r"([\deE\+\.\-]+) ([\deE\+\.\-]+)\s+\(([\deE\+\.\-,\s]+)\)\s+\(([\deE\+\.\-,\s]+)\)\s+\(([\deE\+\.\-,\s]+)\)\s+CreationProcessID=(\d+)",
            line.strip()
        )
        if match:
            time = float(match.group(1))
            event_number = int(time // 1000)
            pos_str = match.group(3)
            pos = np.array([float(x) for x in pos_str.split(',')])
            event_positions[event_number].append(pos)

# ---- 2. Event indexing and splits ----
all_event_numbers = sorted(list(event_positions.keys()))
event_to_idx = {ev: i for i, ev in enumerate(all_event_numbers)}
num_events = len(all_event_numbers)

event_numbers = np.array(all_event_numbers)
np.random.seed(42)
np.random.shuffle(event_numbers)

n_total = len(event_numbers)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_events = set(event_numbers[:n_train])
val_events = set(event_numbers[n_train:n_train + n_val])
test_events = set(event_numbers[n_train + n_val:])

def events_to_tensor_with_eventid(event_subset):
    data, event_ids = [], []
    for ev in event_subset:
        for pos in event_positions[ev]:
            data.append(pos[:2])  # ONLY X, Y coordinates
            event_ids.append(event_to_idx[ev])
    return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(event_ids, dtype=torch.long)

train_data, train_ev_ids = events_to_tensor_with_eventid(train_events)
val_data, val_ev_ids = events_to_tensor_with_eventid(val_events)
test_data, test_ev_ids = events_to_tensor_with_eventid(test_events)

# ---- 3. Normalization ----
mean, std = train_data.mean(0), train_data.std(0)
train_data = (train_data - mean) / std
val_data = (val_data - mean) / std
test_data = (test_data - mean) / std

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dim = 2
embed_dim = 12

# ---- Updated Improved Model ----
class VectorField(nn.Module):
    def __init__(self, num_events, embed_dim, data_dim):
        super().__init__()
        self.event_emb = nn.Embedding(num_events, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + embed_dim + 1, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, data_dim)
        )

    def forward(self, t, y_event):
        y, event_id = y_event
        emb = self.event_emb(event_id)
        t_exp = t.expand(y.shape[0], 1)
        inp = torch.cat([y, emb, t_exp], dim=1)
        return self.net(inp)

vector_field = VectorField(num_events, embed_dim, data_dim).to(device)
optimizer = torch.optim.AdamW(vector_field.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.3)

# Gradient clipping to stabilize training
clip_value = 1.0

# Prior
torch.manual_seed(42) # ;)
prior = torch.distributions.MultivariateNormal(torch.zeros(data_dim, device=device), torch.eye(data_dim, device=device))

train_data, train_ev_ids = train_data.to(device), train_ev_ids.to(device)
val_data, val_ev_ids = val_data.to(device), val_ev_ids.to(device)
test_data, test_ev_ids = test_data.to(device), test_ev_ids.to(device)

# Training Loop
num_epochs = 50
batch_size = 1024

def get_log_likelihood(x, event_ids):
    t_span = torch.tensor([0., 1.], device=device)
    y = x.to(device)
    e = event_ids.to(device)
    z_t = odeint(lambda t, y: vector_field(t, (y, e)), y, t_span, atol=1e-5, rtol=1e-5, method='dopri5')
    z = z_t[-1]
    log_pz = prior.log_prob(z)
    return log_pz

for epoch in range(num_epochs):
    vector_field.train()
    idx = torch.randperm(len(train_data))[:batch_size]
    batch_x, batch_event = train_data[idx], train_ev_ids[idx]
    ll = get_log_likelihood(batch_x, batch_event)
    loss = -ll.mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(vector_field.parameters(), clip_value)
    optimizer.step()

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        vector_field.eval()
        with torch.no_grad():
            idx = torch.randperm(len(val_data))[:4096]
            val_x, val_event = val_data[idx], val_ev_ids[idx]
            val_ll = get_log_likelihood(val_x, val_event)
            val_nll = -val_ll.mean().item()
            print(f"Epoch {epoch:03d} | Train NLL: {loss.item():.3f} | Val NLL: {val_nll:.3f}")
            scheduler.step(val_nll)


# Create output directory for quiver frames
output_dir = "quiver_frames"
os.makedirs(output_dir, exist_ok=True)
# Parameters for quiver grid
grid_size = 20
xv, yv = np.meshgrid(np.linspace(-2, 2, grid_size), np.linspace(-2, 2, grid_size))
points = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)   # <-- fixed

# Pick a representative event (first from validation set)
event_idx = list(val_events)[0]
event_id_tensor = torch.full((points.shape[0],), event_to_idx[event_idx], dtype=torch.long, device=device)

# Quiver plots at multiple time slices
timesteps = np.linspace(0, 1, 20, dtype=np.float32)  # Now float32
for i, t in enumerate(timesteps):
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
    with torch.no_grad():
        vf = vector_field(t_tensor, (torch.tensor(points, dtype=torch.float32, device=device), event_id_tensor))
    u, v = vf[:, 0].cpu().numpy(), vf[:, 1].cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.quiver(points[:, 0], points[:, 1], u, v, angles='xy')
    plt.xlim(-2, 2); plt.ylim(-2, 2)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"Vector Field, t={t:.2f}")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"quiver_{i:03d}.png")
    plt.savefig(fname)
    plt.close()
print(f"Saved {len(timesteps)} quiver frames in {output_dir}")

# ---- 6. Plot 10 random real and generated events ----
import random
fig = plt.figure(figsize=(22, 9))
chosen_events = np.random.choice(list(val_events), size=10, replace=False)

for i, event_number in enumerate(chosen_events):
    event_idx = event_to_idx[event_number]
    # Extract real hits for this event (denormalized)
    real_hits = np.array([pos for ev, pos in zip(val_ev_ids.cpu().numpy(), val_data.cpu().numpy()) if ev == event_idx])
    real_hits = real_hits * std.cpu().numpy() + mean.cpu().numpy()
    num_points = len(real_hits)
    z_samples = torch.randn(num_points, data_dim, device=device)
    event_id_tensor = torch.full((num_points,), event_idx, dtype=torch.long, device=device)
    t_span = torch.tensor([1., 0.], device=device)
    with torch.no_grad():
        x_t = odeint(lambda t, y: vector_field(t, (y, event_id_tensor)), z_samples, t_span, atol=1e-5, rtol=1e-5, method='dopri5')
        x_generated = x_t[-1].cpu().numpy()
        x_generated = x_generated * std.cpu().numpy() + mean.cpu().numpy()

    ax = fig.add_subplot(2, 5, i+1)
    ax.scatter(real_hits[:, 0], real_hits[:, 1], label='Real', alpha=0.6)
    ax.scatter(x_generated[:, 0], x_generated[:, 1], label='CNF Gen', alpha=0.6)
    ax.set_title(f"Event {event_number}")
    ax.set_xticks([]); ax.set_yticks([])
    if i == 0:
        ax.legend()

plt.suptitle("10 Random Validation Events: Real (blue) vs CNF Generated (orange)")
plt.tight_layout()
plt.show()

"""Minimal Koopman: a 4-D latent where z = (hour-hand circle, minute-hand circle) and K = two rotations.

A 4-D Koopman VAE trained on its own gets stuck on harmonics (see the experiment log).
But the trained 16-D model already contains both fundamentals as Koopman eigenfunctions:
    hour:   phi_{0.5 deg}                      (learned directly)
    minute: phi_{18 deg} * conj(phi_{12 deg})  (product of eigenfunctions = eigenfunction, 6 deg)
So we
  1. distill: train a fresh 4-D Koopman VAE whose latent mean matches (Re, Im) of these two,
  2. initialise K by least squares on those coordinates (frozen during step 1),
  3. fine-tune with the plain Koopman loss only (teacher dropped), on longer runs (H up to 36).

Saves runs/koopman_min.pt and runs/minimal_snapshots.npz.
"""
import argparse, time
import numpy as np
import torch
import animate as A                 # the 16-D model, its eigen-decomposition and hand-mode recipes
from models import KoopmanVAE, get_device, load_data, to_tensor, recon_loss, kl_loss

p = argparse.ArgumentParser()
p.add_argument("--distill_iters", type=int, default=3000)
p.add_argument("--iters", type=int, default=4000)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--H_max", type=int, default=36)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--w_teach", type=float, default=200.0)
p.add_argument("--w_lat", type=float, default=500.0)
args = p.parse_args()
torch.manual_seed(0); np.random.seed(0)
dev = get_device()
data = load_data()
train_f, val_f, val_a = data["train_frames"], data["val_frames"], data["val_angles"]
T = train_f.shape[1]

# ---- teacher coordinates from the 16-D model ------------------------------------------
teacher = A.kvae.to(dev).eval()


@torch.no_grad()
def teach(frames_u8):
    out = []
    for i in range(0, len(frames_u8), 2000):
        mu = teacher.encode(to_tensor(frames_u8[i:i + 2000], dev))[0].cpu().numpy()
        h, m = A.modes(mu)
        out.append(np.c_[h.real, h.imag, m.real, m.imag])
    return np.concatenate(out).astype(np.float32)


t0 = time.time()
targets = teach(train_f.reshape(-1, 64, 64)).reshape(len(train_f), T, 4)
print(f"teacher targets {targets.shape}  ({time.time() - t0:.0f}s)")
# K from least squares on the teacher's coordinates: should be two rotation blocks
Z0, Z1 = targets[:, :-1].reshape(-1, 4), targets[:, 1:].reshape(-1, 4)
K0 = np.linalg.lstsq(Z0, Z1, rcond=None)[0].T
ev = np.linalg.eigvals(K0)
print("K from teacher coordinates: angles", np.round(np.degrees(np.angle(ev)), 3), "|lambda|", np.round(np.abs(ev), 4))

model = KoopmanVAE(4).to(dev)
with torch.no_grad():
    model.K.copy_(torch.as_tensor(K0, dtype=torch.float32))
# K stays frozen at the teacher fit while the fresh encoder/decoder are distilled
opt = torch.optim.Adam([p_ for n, p_ in model.named_parameters() if n != "K"], lr=args.lr)
eval_x = to_tensor(val_f[:, ::3].reshape(-1, 64, 64)[:2000], dev)
log = {"it": [], "phase": [], "rec": [], "lat": [], "pred": [], "angles": []}


def step_batch(H):
    idx = np.random.randint(0, len(train_f), args.bs)
    st = np.random.randint(0, T - H, args.bs)
    seq = np.stack([train_f[i, s:s + H + 1] for i, s in zip(idx, st)])
    tgt = np.stack([targets[i, s:s + H + 1] for i, s in zip(idx, st)])
    return to_tensor(seq.reshape(-1, 64, 64), dev), torch.as_tensor(tgt, device=dev)


snaps = {"it": [], "phase": [], "mu": [], "K": [], "rec": [], "pred": []}


@torch.no_grad()
def snapshot(it, phase, rec, pred):
    model.eval()
    mu_ = torch.cat([model.encode(eval_x[i:i + 500])[0] for i in range(0, 2000, 500)]).cpu().numpy()
    model.train()
    snaps["it"].append(it); snaps["phase"].append(phase); snaps["mu"].append(mu_)
    snaps["K"].append(model.K.detach().cpu().numpy()); snaps["rec"].append(rec); snaps["pred"].append(pred)


total = args.distill_iters + args.iters
snapshot(0, 1, float("nan"), float("nan"))
t0 = time.time()
for it in range(1, total + 1):
    distill = it <= args.distill_iters
    H = 8 if distill else int(min(args.H_max, 8 + (args.H_max - 8) * (it - args.distill_iters) / (0.4 * args.iters)))
    x, tgt = step_batch(H)
    logits, mu, logvar = model(x)
    l_rec = recon_loss(logits, x) + kl_loss(mu, logvar)
    mu = mu.view(args.bs, H + 1, 4)
    z_pred = model.rollout(mu[:, 0], H)[:, 1:]
    l_lat = ((z_pred - mu[:, 1:]) ** 2).sum(-1).mean() / mu.var((0, 1)).sum().detach()
    x_next = x.view(args.bs, H + 1, 1, 64, 64)[:, 1:].reshape(-1, 1, 64, 64)
    l_pred = recon_loss(model.decode(z_pred.reshape(-1, 4)), x_next)
    l_teach = ((mu - tgt) ** 2).sum(-1).mean()
    if distill:                                   # phase 1: only VAE loss + match the teacher's eigen-coordinates
        loss = l_rec + args.w_teach * l_teach
    else:                                         # phase 2: plain Koopman VAE loss, teacher dropped
        loss = l_rec + args.w_lat * l_lat + l_pred
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    opt.step()
    if it == args.distill_iters:                  # phase 2: K joins, teacher is dropped
        opt = torch.optim.Adam(model.parameters(), lr=args.lr / 2)
    if it % 50 == 0:
        snapshot(it, int(distill), l_rec.item(), l_pred.item())
    if it % 250 == 0:
        ang = np.degrees(np.angle(np.linalg.eigvals(model.K.detach().cpu().numpy())))
        mag = np.abs(np.linalg.eigvals(model.K.detach().cpu().numpy()))
        log["it"].append(it); log["phase"].append(int(distill)); log["rec"].append(l_rec.item())
        log["lat"].append(l_lat.item()); log["pred"].append(l_pred.item()); log["angles"].append(ang)
        print(f"it {it:5d} {'distill' if distill else 'koopman'} H {H:2d}  rec {l_rec.item():6.1f}  "
              f"teach {l_teach.item():.4f}  lat {l_lat.item():.4f}  pred {l_pred.item():6.1f}  angles {np.round(ang, 3)}  |lambda| {np.round(mag, 4)}  "
              f"({time.time() - t0:.0f}s)", flush=True)

model.eval()
with torch.no_grad():
    mu_eval = torch.cat([model.encode(eval_x[i:i + 500])[0] for i in range(0, 2000, 500)]).cpu().numpy()
torch.save({"state": model.state_dict(), "d": 4}, "runs/koopman_min.pt")
np.savez_compressed("runs/minimal_snapshots.npz", mu=mu_eval, eval_angles=val_a[:, ::3].reshape(-1, 2)[:2000],
                    **{k: np.array(v) for k, v in log.items()},
                    **{"snap_" + k: np.array(v) for k, v in snaps.items()})
print("saved runs/koopman_min.pt, runs/minimal_snapshots.npz")

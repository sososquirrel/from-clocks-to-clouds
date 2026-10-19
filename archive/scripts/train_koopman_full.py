"""Stage 2: add the Koopman loss.

Start from the stage-1 VAE and add a learnable d x d matrix K. On short trajectories
x_0, x_1, ..., x_H we ask that

    z_k  ~=  K^k z_0                (latent linearity:   ||mu(x_k) - K^k mu(x_0)||^2)
    x_k  ~=  decode(K^k z_0)        (prediction:         BCE(decode(K^k mu(x_0)), x_k))

on top of the usual VAE loss. The encoder is forced to find coordinates in which
"one tick of the clock" is a single matrix multiplication.

Also fits the baseline "VAE + best linear map" (least squares on stage-1 latents) so
the animations can compare the two.

Saves runs/koopman.pt and runs/koopman_snapshots.npz.
"""
import argparse, time
import numpy as np
import torch
from models import VAE, KoopmanVAE, get_device, load_data, to_tensor, recon_loss, kl_loss, downsample

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=6000)
p.add_argument("--bs", type=int, default=32, help="trajectories per batch")
p.add_argument("--H_max", type=int, default=24, help="max rollout horizon (curriculum from 2)")
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--beta", type=float, default=1.0)
p.add_argument("--w_lat", type=float, default=500.0)
p.add_argument("--w_pred", type=float, default=1.0)
p.add_argument("--koopman_start", type=int, default=0,
               help="iterations of VAE + slowness only before the Koopman losses switch on")
p.add_argument("--w_slow", type=float, default=0.0,
               help="slowness prior (slow feature analysis): penalise ||mu_t+1 - mu_t||^2 / Var(mu)")
p.add_argument("--snap_every", type=int, default=50)
p.add_argument("--rollout_len", type=int, default=360, help="steps in the long probe rollout")
p.add_argument("--rollout_stride", type=int, default=10, help="store every n-th rollout frame")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="")
p.add_argument("--vae", default="runs/vae.pt")
p.add_argument("--data", default="data/clocks.npz")
p.add_argument("--K_mode", choices=["param", "lstsq"], default="param",
               help="param: K is a learned matrix; lstsq: K is the least-squares fit on each batch (DMD in the loop)")
p.add_argument("--K_init", choices=["identity", "spread"], default="identity")
p.add_argument("--K_init_deg", default="", help="comma-separated rotation angles (deg/step) for 2x2 blocks of the initial K, e.g. 3,10")
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = get_device()
data = load_data(args.data)
train_f, train_a = data["train_frames"], data["train_angles"]
val_frames, val_angles = data["val_frames"], data["val_angles"]
T = train_f.shape[1]
eval_angles = val_angles[:, ::3].reshape(-1, 2)[:2000]

ck = torch.load(args.vae, map_location=dev)
d, S, WID = ck["d"], ck.get("size", 64), ck.get("width", 1.0)
model = KoopmanVAE(d, S, WID).to(dev)
train_f, val_frames = downsample(train_f, S), downsample(val_frames, S)
eval_x = to_tensor(val_frames[:, ::3].reshape(-1, S, S)[:2000], dev)
probe_x0 = to_tensor(val_frames[:4, 0], dev)          # 4 clocks we roll out far into the future
model.load_state_dict(ck["state"], strict=False)       # K starts at the identity
if args.K_init_deg:                                     # rotations at the given angles
    with torch.no_grad():
        for j, a in enumerate(float(v) for v in args.K_init_deg.split(",")):
            w = np.deg2rad(a)
            model.K[2 * j:2 * j + 2, 2 * j:2 * j + 2] = torch.tensor([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
elif args.K_init == "spread":                             # rotations at evenly spread frequencies
    with torch.no_grad():
        for j in range(d // 2):
            w = np.pi / 2 * (j + 1) / (d // 2 + 1)
            model.K[2 * j:2 * j + 2, 2 * j:2 * j + 2] = torch.tensor([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])


@torch.no_grad()
def encode_all(frames_u8):
    out = []
    for i in range(0, len(frames_u8), 1000):
        out.append(model.encode(to_tensor(frames_u8[i:i + 1000], dev))[0])
    return torch.cat(out)


# ---- baseline: best linear map in the *stage-1* latent (no retraining) ----------
model.eval()
mu_tr = encode_all(train_f.reshape(-1, S, S)).view(len(train_f), T, d)
Z0, Z1 = mu_tr[:, :-1].reshape(-1, d), mu_tr[:, 1:].reshape(-1, d)
A_base = torch.linalg.lstsq(Z0.cpu(), Z1.cpu()).solution.T.to(dev)      # z1 ~= A z0
with torch.no_grad():
    mu0 = model.encode(probe_x0)[0]
    zs = [mu0]
    for _ in range(args.rollout_len):
        zs.append(zs[-1] @ A_base.T)
    base_rollout = torch.sigmoid(model.decode(torch.cat(zs))).view(len(zs), 4, S, S).transpose(0, 1)
    base_rollout = base_rollout.cpu().numpy()
    vae_mu_eval = encode_all(val_frames[:, ::3].reshape(-1, S, S)[:2000]).cpu().numpy()
print("baseline A eigenvalues:", np.round(np.linalg.eigvals(A_base.cpu().numpy()), 3))
model.train()

# ---- training ---------------------------------------------------------------------
opt = torch.optim.Adam([{"params": [p_ for n, p_ in model.named_parameters() if n != "K"], "lr": args.lr},
                        {"params": [model.K], "lr": 2 * args.lr}])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
snaps = {k: [] for k in ["it", "mu", "K", "rollout", "loss_rec", "loss_lat", "loss_pred", "H"]}


@torch.no_grad()
def snapshot(it, H, losses):
    model.eval()
    mu = encode_all(val_frames[:, ::3].reshape(-1, S, S)[:2000])
    zs = model.rollout(model.encode(probe_x0)[0], args.rollout_len)          # (4, L+1, d)
    zs = zs[:, ::args.rollout_stride]
    roll = torch.sigmoid(model.decode(zs.reshape(-1, d))).view(4, -1, S, S)
    snaps["it"].append(it); snaps["H"].append(H)
    snaps["mu"].append(mu.cpu().numpy()); snaps["K"].append(model.K.detach().cpu().numpy())
    snaps["rollout"].append((roll.cpu().numpy() * 255).astype(np.uint8))
    for k, v in zip(["loss_rec", "loss_lat", "loss_pred"], losses):
        snaps[k].append(v)
    model.train()


t0 = time.time()
snapshot(0, 2, (np.nan,) * 3)
for it in range(1, args.iters + 1):
    H0 = min(2, args.H_max)                                                  # horizon curriculum H0 -> H_max
    kit = max(0, it - args.koopman_start)                                    # iterations since Koopman switched on
    H = int(min(args.H_max, H0 + (args.H_max - H0) * kit / (0.4 * (args.iters - args.koopman_start))))
    idx = np.random.randint(0, len(train_f), args.bs)
    start = np.random.randint(0, T - H, args.bs)
    seq = np.stack([train_f[i, s:s + H + 1] for i, s in zip(idx, start)])   # (B, H+1, 64, 64)
    x = to_tensor(seq.reshape(-1, S, S), dev)                              # (B*(H+1), 1, 64, 64)

    logits, mu, logvar = model(x)
    l_rec = recon_loss(logits, x) + args.beta * kl_loss(mu, logvar)

    mu = mu.view(args.bs, H + 1, d)
    if args.K_mode == "lstsq":
        Kb = torch.linalg.lstsq(mu[:, :-1].reshape(-1, d).cpu(), mu[:, 1:].reshape(-1, d).cpu()).solution.T.to(dev)
        with torch.no_grad():
            model.K.lerp_(Kb, 0.05)                                          # EMA, used at test time
        zs = [mu[:, 0]]
        for _ in range(H):
            zs.append(zs[-1] @ Kb.T)
        z_pred = torch.stack(zs[1:], 1)
    else:
        z_pred = model.rollout(mu[:, 0], H)[:, 1:]                           # K^k z0, k=1..H
    # normalised by the latent spread so the encoder cannot cheat by shrinking z
    l_lat = ((z_pred - mu[:, 1:]) ** 2).sum(-1).mean() / mu.var((0, 1)).sum().detach()
    x_next = x.view(args.bs, H + 1, 1, S, S)[:, 1:].reshape(-1, 1, S, S)
    l_pred = recon_loss(model.decode(z_pred.reshape(-1, d)), x_next)

    koopman_on = it > args.koopman_start
    loss = l_rec + (args.w_lat * l_lat + args.w_pred * l_pred if koopman_on else 0.0)
    l_slow = ((mu[:, 1:] - mu[:, :-1]) ** 2).sum(-1).mean() / mu.var((0, 1)).sum().detach()
    if args.w_slow:                       # prefer the slowest coordinates that still reconstruct the clock
        loss = loss + args.w_slow * l_slow
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    opt.step(); sched.step()

    if it % args.snap_every == 0:
        snapshot(it, H, (l_rec.item(), l_lat.item(), l_pred.item()))
    if it % 250 == 0:
        ev = np.linalg.eigvals(model.K.detach().cpu().numpy())
        print(f"it {it:5d} {'K ' if koopman_on else 'SF'} H {H:2d}  rec {l_rec.item():7.1f}  lat {l_lat.item():.4f}  pred {l_pred.item():7.1f}  slow {l_slow.item():.4f}  "
              f"|lambda| {np.round(np.abs(ev), 3)}  angle(deg) {np.round(np.degrees(np.angle(ev)), 1)}  "
              f"({time.time() - t0:.0f}s)", flush=True)

torch.save({"state": model.state_dict(), "d": d, "size": S, "width": WID, "A_base": A_base.cpu()}, f"runs/koopman{args.tag}.pt")
np.savez_compressed(f"runs/koopman_snapshots{args.tag}.npz",
                    **{k: np.array(v) for k, v in snaps.items()},
                    eval_angles=eval_angles, vae_mu=vae_mu_eval, A_base=A_base.cpu().numpy(),
                    base_rollout=(base_rollout * 255).astype(np.uint8),
                    probe_angles=val_angles[:4, 0], rollout_stride=args.rollout_stride, dt_hour=data["dt_hour"], dt_min=data["dt_min"])
print(f"saved runs/koopman{args.tag}.pt, runs/koopman_snapshots{args.tag}.npz")

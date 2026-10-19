"""Stage 2: switch on the Koopman loss.

Start from the stage-1 VAE and add a learnable d x d matrix K (initialised to the identity).
On clips x_0, ..., x_H of consecutive minutes, with mu_k the encoder mean of x_k:

    L = BCE(decode(z), x) + beta * KL                           VAE loss, every frame
      + w_lat  * mean_k ||K^k mu_0 - mu_k||^2 / Var(mu)          latent evolves linearly (scale-free)
      + w_pred * mean_k BCE(decode(K^k mu_0), x_k)               predicted future looks right

The horizon H grows from 2 to H_max during the first 40% of training. Defaults are the
minimal setup (d = 4 from the VAE, H_max = 36, 12,000 iterations, ~3-4 min on Apple silicon).

    python train_koopman.py                     -> runs/minimal/koopman.pt (+ _snapshots.npz)
"""
import argparse, os, time
import numpy as np
import torch
from models import KoopmanVAE, get_device, load_data, to_tensor, recon_loss, kl_loss, downsample

p = argparse.ArgumentParser()
p.add_argument("--vae", default="runs/minimal/vae.pt", help="stage-1 checkpoint to start from")
p.add_argument("--out", default="runs/minimal/koopman.pt")
p.add_argument("--data", default="data/clocks.npz")
p.add_argument("--iters", type=int, default=12000)
p.add_argument("--bs", type=int, default=16, help="clips per batch")
p.add_argument("--H_max", type=int, default=36, help="max prediction horizon in steps (curriculum from 2)")
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--beta", type=float, default=1.0, help="weight of the KL term")
p.add_argument("--w_lat", type=float, default=125.0, help="latent-linearity weight (125 at 32 px, 500 at 64 px)")
p.add_argument("--w_pred", type=float, default=1.0)
p.add_argument("--snap_every", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = get_device()
data = load_data(args.data)
ck = torch.load(args.vae, map_location=dev)
d, S, WID = ck["d"], ck.get("size", 64), ck.get("width", 1.0)
train_f = downsample(data["train_frames"], S)
val_frames, val_angles = downsample(data["val_frames"], S), data["val_angles"]
T = train_f.shape[1]
eval_x = to_tensor(val_frames[:, ::3].reshape(-1, S, S)[:2000], dev)
eval_angles = val_angles[:, ::3].reshape(-1, 2)[:2000]

model = KoopmanVAE(d, S, WID).to(dev)
model.load_state_dict(ck["state"], strict=False)                   # encoder/decoder from stage 1, K = identity
opt = torch.optim.Adam([{"params": [p_ for n, p_ in model.named_parameters() if n != "K"], "lr": args.lr},
                        {"params": [model.K], "lr": 2 * args.lr}])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
snaps = {k: [] for k in ["it", "mu", "K", "loss_rec", "loss_lat", "loss_pred", "H"]}


@torch.no_grad()
def snapshot(it, H, losses):
    model.eval()
    mu = torch.cat([model.encode(eval_x[i:i + 1000])[0] for i in range(0, len(eval_x), 1000)])
    snaps["it"].append(it); snaps["H"].append(H)
    snaps["mu"].append(mu.cpu().numpy()); snaps["K"].append(model.K.detach().cpu().numpy())
    for k, v in zip(["loss_rec", "loss_lat", "loss_pred"], losses):
        snaps[k].append(v)
    model.train()


t0 = time.time()
snapshot(0, 2, (np.nan,) * 3)
for it in range(1, args.iters + 1):
    H0 = min(2, args.H_max)
    H = int(min(args.H_max, H0 + (args.H_max - H0) * it / (0.4 * args.iters)))     # horizon curriculum
    idx = np.random.randint(0, len(train_f), args.bs)
    start = np.random.randint(0, T - H, args.bs)
    seq = np.stack([train_f[i, s:s + H + 1] for i, s in zip(idx, start)])            # (B, H+1, S, S)
    x = to_tensor(seq.reshape(-1, S, S), dev)

    logits, mu, logvar = model(x)
    l_rec = recon_loss(logits, x) + args.beta * kl_loss(mu, logvar)
    mu = mu.view(args.bs, H + 1, d)
    z_pred = model.rollout(mu[:, 0], H)[:, 1:]                                         # K^k mu_0, k = 1..H
    l_lat = ((z_pred - mu[:, 1:]) ** 2).sum(-1).mean() / mu.var((0, 1)).sum().detach()  # scale-free
    x_next = x.view(args.bs, H + 1, 1, S, S)[:, 1:].reshape(-1, 1, S, S)
    l_pred = recon_loss(model.decode(z_pred.reshape(-1, d)), x_next)

    loss = l_rec + args.w_lat * l_lat + args.w_pred * l_pred
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    opt.step(); sched.step()

    if it % args.snap_every == 0:
        snapshot(it, H, (l_rec.item(), l_lat.item(), l_pred.item()))
    if it % 500 == 0:
        ev = np.linalg.eigvals(model.K.detach().cpu().numpy())
        print(f"it {it:5d}  H {H:2d}  rec {l_rec.item():7.1f}  lat {l_lat.item():.4f}  pred {l_pred.item():7.1f}  "
              f"|lambda| {np.round(np.abs(ev), 3)}  angle(deg) {np.round(np.degrees(np.angle(ev)), 2)}  ({time.time() - t0:.0f}s)", flush=True)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
snap_path = args.out.replace(".pt", "_snapshots.npz")
torch.save({"state": model.state_dict(), "d": d, "size": S, "width": WID, "beta": args.beta}, args.out)
np.savez_compressed(snap_path, **{k: np.array(v) for k, v in snaps.items()}, eval_angles=eval_angles,
                    dt_hour=data["dt_hour"], dt_min=data["dt_min"])
print(f"saved {args.out}, {snap_path}")

"""Stage 1: a plain VAE on single clock images (no notion of time).

Every frame of every trajectory is treated as an independent image. The VAE compresses
each clock into d numbers (default d = 4) and draws it back. Defaults are the minimal
setup: 32x32 images, half-width network, 15 epochs (~45 s on Apple silicon).

    python train_vae.py                         -> runs/minimal/vae.pt (+ _snapshots.npz)
    python train_vae.py --beta 0 --out runs/minimal/kl/vae_b0.pt
"""
import argparse, os, time
import numpy as np
import torch
from models import VAE, get_device, load_data, to_tensor, recon_loss, kl_loss, downsample

p = argparse.ArgumentParser()
p.add_argument("--d", type=int, default=4, help="latent dimension")
p.add_argument("--size", type=int, default=32, help="image size: 32 (default) or 64")
p.add_argument("--width", type=float, default=0.5, help="network width multiplier")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--bs", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--beta", type=float, default=1.0, help="weight of the KL term (0 = plain autoencoder)")
p.add_argument("--snap_every", type=int, default=100, help="iterations between latent snapshots")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default="data/clocks.npz")
p.add_argument("--out", default="runs/minimal/vae.pt")
args = p.parse_args()

torch.manual_seed(args.seed)
dev = get_device()
data = load_data(args.data)
S = args.size
train = downsample(data["train_frames"], S).reshape(-1, S, S)
val_frames, val_angles = downsample(data["val_frames"], S), data["val_angles"]
eval_x = to_tensor(val_frames[:, ::3].reshape(-1, S, S)[:2000], dev)      # fixed probe set (2,000 held-out clocks)
eval_angles = val_angles[:, ::3].reshape(-1, 2)[:2000]

model = VAE(args.d, S, args.width).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
n_iter = args.epochs * (len(train) // args.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=n_iter, pct_start=0.05)
snaps = {"it": [], "mu": [], "loss": []}


@torch.no_grad()
def snapshot(it, loss):
    model.eval()
    mu = torch.cat([model.encode(eval_x[i:i + 500])[0] for i in range(0, len(eval_x), 500)])
    snaps["it"].append(it); snaps["loss"].append(loss); snaps["mu"].append(mu.cpu().numpy())
    model.train()


it, t0 = 0, time.time()
snapshot(0, float("nan"))
for ep in range(args.epochs):
    perm = np.random.permutation(len(train))
    for b in range(len(train) // args.bs):
        x = to_tensor(train[perm[b * args.bs:(b + 1) * args.bs]], dev)
        logits, mu, logvar = model(x)
        rl, kl = recon_loss(logits, x), kl_loss(mu, logvar)
        loss = rl + args.beta * kl
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        it += 1
        if it % args.snap_every == 0:
            snapshot(it, loss.item())
    print(f"epoch {ep + 1:3d}/{args.epochs}  recon {rl.item():8.2f}  kl {kl.item():6.2f}  ({time.time() - t0:.0f}s)", flush=True)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
snap_path = args.out.replace(".pt", "_snapshots.npz")
torch.save({"state": model.state_dict(), "d": args.d, "size": S, "width": args.width, "beta": args.beta}, args.out)
np.savez_compressed(snap_path, it=np.array(snaps["it"]), loss=np.array(snaps["loss"]), mu=np.stack(snaps["mu"]),
                    eval_angles=eval_angles)
print(f"saved {args.out}, {snap_path}")

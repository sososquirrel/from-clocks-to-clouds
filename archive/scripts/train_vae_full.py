"""Stage 1: a plain VAE on single clock images (no notion of time).

Every frame of every trajectory is treated as an independent image. The VAE must
compress 4096 pixels into a d-dimensional latent; the only thing that varies between
clocks is the two hands, so the latent ends up encoding the hands.

Saves runs/vae.pt and runs/vae_snapshots.npz (latent of a fixed eval set over training,
used by animate.py).
"""
import argparse, os, time
import numpy as np
import torch
from models import VAE, get_device, load_data, to_tensor, recon_loss, kl_loss, downsample

p = argparse.ArgumentParser()
p.add_argument("--d", type=int, default=16)
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--bs", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--beta", type=float, default=1.0)
p.add_argument("--snap_every", type=int, default=50, help="iterations between latent snapshots")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="", help="suffix for output files, e.g. _d4")
p.add_argument("--size", type=int, default=64, help="image size: 64, or 32 for fast experiments")
p.add_argument("--width", type=float, default=1.0, help="network width multiplier (0.5 = fast)")
args = p.parse_args()

torch.manual_seed(args.seed)
dev = get_device()
data = load_data()
S = args.size
data["train_frames"], data["val_frames"] = downsample(data["train_frames"], S), downsample(data["val_frames"], S)
train = data["train_frames"].reshape(-1, S, S)
val_frames, val_angles = data["val_frames"], data["val_angles"]
eval_x = to_tensor(val_frames[:, ::3].reshape(-1, S, S)[:2000], dev)      # fixed probe set
eval_angles = val_angles[:, ::3].reshape(-1, 2)[:2000]
recon_x = to_tensor(val_frames[:8, 0], dev)

model = VAE(args.d, S, args.width).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
n_iter = args.epochs * (len(train) // args.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=n_iter, pct_start=0.05)

snaps = {"it": [], "mu": [], "recon": [], "loss": []}


@torch.no_grad()
def snapshot(it, loss):
    model.eval()
    mu = torch.cat([model.encode(eval_x[i:i + 500])[0] for i in range(0, len(eval_x), 500)])
    rec = torch.sigmoid(model.decode(model.encode(recon_x)[0]))
    snaps["it"].append(it); snaps["loss"].append(loss)
    snaps["mu"].append(mu.cpu().numpy()); snaps["recon"].append(rec.squeeze(1).cpu().numpy())
    model.train()


os.makedirs("runs", exist_ok=True)
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
    print(f"epoch {ep + 1:3d}/{args.epochs}  recon {rl.item():8.2f}  kl {kl.item():6.2f}  "
          f"({time.time() - t0:.0f}s)", flush=True)

torch.save({"state": model.state_dict(), "d": args.d, "size": S, "width": args.width}, f"runs/vae{args.tag}.pt")
np.savez_compressed(f"runs/vae_snapshots{args.tag}.npz", it=np.array(snaps["it"]), loss=np.array(snaps["loss"]),
                    mu=np.stack(snaps["mu"]), recon=np.stack(snaps["recon"]),
                    recon_target=val_frames[:8, 0] / 255.0, eval_angles=eval_angles)
print(f"saved runs/vae{args.tag}.pt, runs/vae_snapshots{args.tag}.npz")

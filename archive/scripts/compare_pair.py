"""Experiment: Koopman loss on single pairs (x_t, x_t+1) vs on multi-step clips.

Both models start from the same stage-1 VAE and are trained identically except for the
rollout horizon H (1 vs up to 24). We roll each one out for 6 hours from held-out
clocks and measure the pixel error against the true future.

    python compare_pair.py   ->  animations/pair_vs_clip.png, animations/pair_vs_clip.mp4
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch

import clocks
from models import KoopmanVAE, get_device

dev = get_device()
STEPS = 360
data = np.load("data/clocks.npz")
dt_h, dt_m = float(data["dt_hour"]), float(data["dt_min"])
starts = data["val_angles"][:200, 0]                     # 200 held-out initial clocks
MODELS = [("pairs only (H = 1)", "runs/koopman_pair.pt", "#d1452f"),
          ("clips (H up to 24)", "runs/koopman.pt", "#2f6fdf")]


def load(path):
    ck = torch.load(path, map_location=dev)
    m = KoopmanVAE(ck["d"]).to(dev)
    m.load_state_dict(ck["state"]); m.eval()
    return m


def truth(a0, s):
    return clocks.render((a0[:, 0] + s * dt_h) % (2 * np.pi), (a0[:, 1] + s * dt_m) % (2 * np.pi))


@torch.no_grad()
def rollout(m, a0, steps):
    """Decoded K^k mu(x0) for k = 0..steps. Returns (N, steps+1, 64, 64)."""
    z = m.encode(torch.as_tensor(truth(a0, 0)[:, None], device=dev))[0]
    zs = m.rollout(z, steps)
    out = torch.cat([torch.sigmoid(m.decode(c)) for c in zs.reshape(-1, m.d).split(4096)])
    return out.view(len(a0), steps + 1, 64, 64).cpu().numpy()


models = [(name, load(p), c) for name, p, c in MODELS]
gt = np.stack([truth(starts, s) for s in range(STEPS + 1)], 1)
err = {}
for name, m, _ in models:
    pred = rollout(m, starts, STEPS)
    err[name] = ((pred - gt) ** 2).mean((0, 2, 3))
    print(f"{name:22s}  MSE @1min {err[name][1]:.4f}  @1h {err[name][60]:.4f}  @6h {err[name][STEPS]:.4f}")

# ---- static figure: error vs horizon + eigenvalues ------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2, 1, 1]})
t = np.arange(STEPS + 1) / 60
for name, m, c in models:
    ax[0].plot(t, err[name], color=c, lw=2, label=name)
blur = ((gt.mean(0, keepdims=True) - gt) ** 2).mean((0, 2, 3)).mean()
ax[0].axhline(blur, color="#999999", ls="--", lw=1, label="'average clock' (no prediction)")
ax[0].set_xlabel("prediction horizon (hours)"); ax[0].set_ylabel("pixel MSE"); ax[0].legend(frameon=False)
ax[0].set_title("Error of K^k z₀ vs. true future (200 held-out clocks)")
circ = np.linspace(0, 2 * np.pi, 300)
for a, (name, m, c) in zip(ax[1:], models):
    ev = np.linalg.eigvals(m.K.detach().cpu().numpy())
    a.plot(np.cos(circ), np.sin(circ), color="#cccccc", lw=1)
    for w in (dt_h, dt_m):
        a.scatter([np.cos(w)] * 2, [np.sin(w), -np.sin(w)], s=150, facecolor="none", edgecolor="#1a9e5a", lw=1.5)
    a.scatter(ev.real, ev.imag, color=c, s=25, zorder=3)
    a.set_xlim(0.85, 1.02); a.set_ylim(-0.25, 0.25); a.set_aspect("equal")
    a.set_title(f"eigenvalues of K: {name}", fontsize=10)
    a.text(0.86, -0.23, "green rings: true hour (0.5°) and\nminute (6°) rotations", fontsize=8, color="#1a9e5a")
plt.tight_layout(); plt.savefig("animations/pair_vs_clip.png", dpi=110)

# ---- video: one clock, truth vs both models ------------------------------------------
a0 = starts[:1]
rolls = [rollout(m, a0, STEPS)[0] for _, m, _ in models]
fig, axs = plt.subplots(1, 3, figsize=(12.8, 5.0), dpi=100)
titles = ["ground truth"] + [n for n, _, _ in models]
colors = ["#1d1d1f"] + [c for _, _, c in models]
ims = []
for a, ttl, c in zip(axs, titles, colors):
    ims.append(a.imshow(np.zeros((64, 64)), cmap="gray", vmin=0, vmax=1, interpolation="bilinear"))
    a.set_title(ttl, color=c, fontsize=14, fontweight="bold"); a.axis("off")
txt = fig.text(0.5, 0.04, "", ha="center", fontsize=14)
w = imageio.get_writer("animations/pair_vs_clip.mp4", fps=30, codec="libx264", quality=8,
                       macro_block_size=1, ffmpeg_log_level="error")
for s in range(STEPS + 1):
    for im, frame in zip(ims, [gt[0, s]] + [r[s] for r in rolls]):
        im.set_data(1 - frame)
    txt.set_text(f"{s // 60}h{s % 60:02d} into the future   (K applied {s} times)")
    fig.canvas.draw()
    w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
w.close()
print("wrote animations/pair_vs_clip.png, animations/pair_vs_clip.mp4")

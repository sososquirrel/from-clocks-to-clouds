"""Mixed vs natural modes of the minimal d = 4 Koopman model.

The model learned two complex eigen-coordinates:
    alpha = phase of mode (1,0)  -> theta_h            (0.5 deg / step)
    beta  = phase of mode (1,1)  -> theta_h + theta_m  (6.5 deg / step)
Natural hands are recovered by a change of basis: theta_h = alpha, theta_m = beta - alpha.
Rotating coordinates in latent space and decoding shows what each one means:
    rotate alpha alone          -> hour turns forward AND minute turns backward (mixed)
    rotate beta alone           -> only the minute hand turns   (= natural minute mode)
    rotate alpha and beta       -> only the hour hand turns     (= natural hour mode)
    apply K^t                   -> both turn at their true speeds

Writes report/media/modes.mp4 and report/media/modes.jpg (a still version of the same).

    python -m analysis.mode_video
"""
import base64, io, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import imageio.v2 as imageio
import torch
import clocks
from models import load_model, downsample

CKPT = "runs/minimal/koopman.pt"
m, ck = load_model(CKPT)
S = ck.get("size", 64)
data = np.load("data/clocks.npz")
ev = downsample(data["val_frames"][:, ::3].reshape(-1, 64, 64)[:2000], S).astype(np.float32) / 255
ang = data["val_angles"][:, ::3].reshape(-1, 2)[:2000]
TWO_PI = 2 * np.pi

K = m.K.detach().numpy()
lam, V = np.linalg.eig(K)
W = np.linalg.inv(V)
with torch.no_grad():
    mu = m.encode(torch.as_tensor(ev[:, None]))[0].numpy()
P = mu @ W.T
fit = lambda p, th: abs(np.mean(np.exp(1j * th) * np.conj(p) / np.abs(p)))
pos = [j for j in range(4) if lam[j].imag > 0]
ja = max(pos, key=lambda j: fit(P[:, j], ang[:, 0]))                       # (1,0): hour
jb = [j for j in pos if j != ja][0]                                         # the other one
conj = {j: int(np.argmin(np.abs(lam - np.conj(lam[j])))) for j in pos}
pq_b = max(((p, q) for p in range(-3, 4) for q in range(-3, 4) if (p, q) != (0, 0)),
           key=lambda pq: fit(P[:, jb], pq[0] * ang[:, 0] + pq[1] * ang[:, 1]))
deg_a, deg_b = np.degrees(np.angle(lam[ja])), np.degrees(np.angle(lam[jb]))
print(f"mode A (1,0): {deg_a:.3f} deg/step | mode B {pq_b}: {deg_b:.3f} deg/step | minute = B - A = {deg_b - deg_a:.3f} deg/step")

# align each phase to the true angle so that 12 o'clock is at the top (one constant per mode)
ca = np.mean(np.exp(1j * ang[:, 0]) * np.conj(P[:, ja]) / np.abs(P[:, ja])); ca /= abs(ca)
cb = np.mean(np.exp(1j * (ang[:, 0] + ang[:, 1])) * np.conj(P[:, jb]) / np.abs(P[:, jb])); cb /= abs(cb)
alpha, beta = P[:, ja] * ca, P[:, jb] * cb
derived = beta * np.conj(alpha)                                             # phase = theta_m


def kpow(t):
    """K^t for any real t: rotations can be applied for fractional time steps."""
    return (V @ np.diag(lam.astype(complex) ** t) @ W).real


def turn(z, da, db):
    p = W @ z
    p[ja] *= np.exp(1j * da); p[conj[ja]] *= np.exp(-1j * da)
    p[jb] *= np.exp(1j * db); p[conj[jb]] *= np.exp(-1j * db)
    return (V @ p).real


@torch.no_grad()
def dec(zs):
    return torch.sigmoid(m.decode(torch.as_tensor(np.atleast_2d(zs), dtype=torch.float32))).squeeze(1).numpy()


i0 = int(np.argmin(np.abs(ang[:, 0] - np.deg2rad(300)) + np.abs(ang[:, 1] - np.deg2rad(60))))   # a ~10:10 clock
z0 = mu[i0]
ROWS = [("rotate α only  (mode (1,0))", lambda d: turn(z0, d, 0), "hour forward, minute backward: a mixed mode"),
        ("rotate β only  (mode (1,1))", lambda d: turn(z0, 0, d), "only the minute hand turns"),
        ("rotate α and β together", lambda d: turn(z0, d, d), "only the hour hand turns"),
        ("time: apply Kᵗ", lambda d: kpow(np.degrees(d) / 6) @ z0, "both hands at their true speeds")]
STEPS = np.deg2rad(np.arange(0, 361, 45))
TIME_STEPS = np.arange(0, 361, 45) / 6.0                                    # minutes, so the minute hand moves 0..360 deg

cyc = lambda th: hsv_to_rgb(np.stack([(th / TWO_PI) % 1, 0.75 * np.ones_like(th), 0.9 * np.ones_like(th)], -1))

# ---------------------------------------------------------------- static figure
fig = plt.figure(figsize=(16, 10.5), dpi=90)
gs = fig.add_gridspec(8, 12, height_ratios=[0.32, 1] * 4, width_ratios=[1] * 9 + [0.25, 1.5, 1.5], hspace=0.15, wspace=0.08)
for r, (name, f, note) in enumerate(ROWS):
    t = fig.add_subplot(gs[2 * r, 0:9]); t.axis("off")
    t.text(0, 0.2, name, fontsize=12, fontweight="bold", va="bottom")
    t.text(0.36, 0.2, "→  " + note, fontsize=11, color="#555", va="bottom")
    imgs = dec([f(d) for d in STEPS]) if r < 3 else dec([kpow(t_) @ z0 for t_ in TIME_STEPS])
    for c, im in enumerate(imgs):
        a_ = fig.add_subplot(gs[2 * r + 1, c]); a_.imshow(1 - im, cmap="gray", vmin=0, vmax=1, interpolation="bilinear"); a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_visible(False)
        a_.set_xlabel(f"t = {TIME_STEPS[c]:.1f} min" if r == 3 else f"+{int(np.degrees(STEPS[c]))}°", fontsize=8, color="#555")
panels = [(alpha, 0, f"α = mode (1,0) · {deg_a:.2f}°/step\ncolour = hour angle"),
          (beta, 1, f"β = mode (1,1) · {deg_b:.2f}°/step\ncolour = minute angle (mixed)"),
          (derived, 1, f"β·conj(α) · {deg_b - deg_a:.2f}°/step\n= the minute hand, colour = minute")]
for k, (vals, col, ttl) in enumerate(panels):
    a_ = fig.add_subplot(gs[[1, 3, 5][k] - 1:[1, 3, 5][k] + 1 + (1 if k == 2 else 0), 10:12] if False else gs[2 * k + 1 + (0 if k < 2 else 0), 10])
    v = vals / np.median(np.abs(vals))
    a_.scatter(v.real, v.imag, c=cyc(ang[:, col]), s=2); a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
    a_.set_title(ttl, fontsize=8.5)
t = fig.add_subplot(gs[7, 10:12]); t.axis("off")
t.text(0, 0.95, "change of basis", fontsize=11, fontweight="bold", va="top")
t.text(0, 0.62, "θ_h = α          θ_m = β − α\nhour = α     minute = β · conj(α)", fontsize=10, va="top", family="monospace")
fig.suptitle("Minimal Koopman, d = 4: the learned modes are mixed; the natural hands are one change of basis away",
             fontsize=14, fontweight="bold", y=0.995)
os.makedirs("report/media", exist_ok=True)
plt.savefig("report/media/modes.png", bbox_inches="tight")
from PIL import Image
Image.open("report/media/modes.png").convert("RGB").save("report/media/modes.jpg", "JPEG", quality=85)
os.remove("report/media/modes.png")

# ---------------------------------------------------------------- animation: the four rows turning together
frames = []
fig2, axs = plt.subplots(1, 4, figsize=(12.8, 4.2), dpi=100)
for a, (name, _, note) in zip(axs, ROWS):
    a.set_xticks([]); a.set_yticks([]); a.set_title(name + "\n" + note, fontsize=10)
    for sp in a.spines.values():
        sp.set_visible(False)
ims = [a.imshow(np.zeros((S, S)), cmap="gray", vmin=0, vmax=1, interpolation="bilinear") for a in axs]
for d in np.deg2rad(np.arange(0, 360, 3)):
    zs = [ROWS[0][1](d), ROWS[1][1](d), ROWS[2][1](d), kpow(np.degrees(d) / 6) @ z0]
    for im, x in zip(ims, dec(zs)):
        im.set_data(1 - x)
    fig2.canvas.draw(); frames.append(np.asarray(fig2.canvas.buffer_rgba())[..., :3].copy())
imageio.mimwrite("report/media/modes.mp4", frames, fps=30, codec="libx264", quality=8, macro_block_size=1)
print("wrote report/media/modes.mp4, report/media/modes.jpg")

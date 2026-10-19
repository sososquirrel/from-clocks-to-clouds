"""Summarise any trained Koopman VAE checkpoints side by side.

    python analyze_runs.py out.png "label=runs/koopman_d4_rot.pt" "label2=runs/koopman.pt" ...

For each run: eigenvalues (labelled by hand/harmonic, with loop length), the best hour and
minute modes (held-out clocks projected, coloured by the true angle), a 6-hour rollout from one
image, and pixel error vs horizon on 200 held-out clocks. Also writes out.json with the numbers.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import torch
import clocks
from models import load_model

out_png = sys.argv[1]
runs = [a.rsplit("=", 1) for a in sys.argv[2:]]          # label=path (labels may contain "=")
data = np.load("data/clocks.npz")
dt_h, dt_m = float(data["dt_hour"]), float(data["dt_min"])
ev_frames = data["val_frames"][:, ::3].reshape(-1, 64, 64)[:2000].astype(np.float32) / 255.0
ev_ang = data["val_angles"][:, ::3].reshape(-1, 2)[:2000]
starts = data["val_angles"][:200, 0]
STEPS, TWO_PI = 360, 2 * np.pi
NAME = {0: "hour", 1: "minute"}
cyc = lambda th: hsv_to_rgb(np.stack([(th / TWO_PI) % 1, 0.75 * np.ones_like(th), 0.9 * np.ones_like(th)], -1))
gt = np.stack([clocks.render((starts[:, 0] + s * dt_h) % TWO_PI, (starts[:, 1] + s * dt_m) % TWO_PI) for s in range(STEPS + 1)], 1)


def score(p, th):
    return abs(np.mean(np.exp(1j * th) * np.conj(p) / (np.abs(p) + 1e-12)))


def fmt_loop(deg):
    n = 360 / abs(deg)
    return f"{n:.0f} steps = {int(n // 60)} h {int(round(n % 60)):02d}" if n >= 60 else f"{n:.0f} steps = {n:.0f} min"


results = []
for label, path in runs:
    m, ck = load_model(path)
    S = ck.get("size", 64)
    pool = lambda a: a if S == 64 else a.reshape(*a.shape[:-2], S, 64 // S, S, 64 // S).mean((-3, -1))
    ev, gt_m = pool(ev_frames), pool(gt)
    K = m.K.detach().numpy()
    lam, V = np.linalg.eig(K); W = np.linalg.inv(V)
    with torch.no_grad():
        mu = torch.cat([m.encode(torch.as_tensor(ev[i:i + 500, None], dtype=torch.float32))[0]
                        for i in range(0, 2000, 500)]).numpy()
        rec = torch.sigmoid(m.decode(torch.as_tensor(mu))).squeeze(1).numpy()
    rec_err = float(((rec - ev) ** 2).mean() * 1000)
    P = mu @ W.T
    modes = []
    for j in range(len(lam)):
        if lam[j].imag <= 1e-6:
            continue
        h, k, sc = max(((h, k, score(P[:, j], k * ev_ang[:, h])) for h in (0, 1) for k in range(1, 31)), key=lambda t: t[2])
        modes.append({"j": j, "deg": float(np.degrees(np.angle(lam[j]))), "abs": float(abs(lam[j])),
                      "hand": h if sc > 0.5 else -1, "k": k if sc > 0.5 else 0, "score": float(sc)})
    # best single eigenfunction per hand, judged on the fundamental phase (k = 1)
    best = {}
    for h in (0, 1):
        cands = [(score(P[:, md["j"]], ev_ang[:, h]), md) for md in modes]
        best[h] = max(cands, key=lambda t: t[0]) if cands else (0.0, None)
    # rollouts
    with torch.no_grad():
        z = m.encode(torch.as_tensor(gt_m[:, 0, None]))[0].numpy()
    zs = [z]
    for _ in range(STEPS):
        zs.append(zs[-1] @ K.T)
    zs = np.stack(zs, 1)
    with torch.no_grad():
        pred = torch.cat([torch.sigmoid(m.decode(torch.as_tensor(c, dtype=torch.float32)))
                          for c in np.array_split(zs.reshape(-1, ck["d"]), 30)]).view(len(z), STEPS + 1, S, S).numpy()
    err = ((pred - gt_m) ** 2).mean((0, 2, 3)) * 1000
    results.append({"label": label, "path": path, "d": ck["d"], "lam": lam, "modes": modes, "best": best, "P": P,
                    "pred0": pred[0] if S == 64 else np.repeat(np.repeat(pred[0], 64 // S, -2), 64 // S, -1),
                    "err": err, "rec_err": rec_err, "size": S})
    print(f"\n{label} (d = {ck['d']}, {S}px): reconstruction MSE {rec_err:.2f}e-3 | error @1h {err[60]:.2f}e-3 @6h {err[360]:.2f}e-3")
    for md in sorted(modes, key=lambda t: t["deg"]):
        tag = f"{NAME[md['hand']]} k={md['k']}" if md["hand"] >= 0 else "unassigned"
        print(f"   {md['deg']:7.3f} deg/step  |lam| {md['abs']:.4f}  {fmt_loop(md['deg']):>22s}  -> {tag:12s} (phase fit {md['score']:.2f})")
    for h in (0, 1):
        print(f"   best {NAME[h]}-hand eigenfunction (fundamental phase fit): {best[h][0]:.2f}")

# ---------------------------------------------------------------- figure
SHOW = [0, 30, 60, 120, 240, 360]
n = len(results)
fig = plt.figure(figsize=(22, 4.3 * n), dpi=80)
gs = fig.add_gridspec(n, 6, width_ratios=[1.1, 1, 1, 2.6, 1.5, 0.01], wspace=0.28, hspace=0.45)
circ = np.linspace(0, TWO_PI, 300)
for r, R in enumerate(results):
    ax = fig.add_subplot(gs[r, 0])
    ax.plot(np.cos(circ), np.sin(circ), color="#bbb", lw=1)
    for kk in range(1, 5):
        a = kk * dt_m
        ax.scatter([np.cos(a)] * 2, [np.sin(a), -np.sin(a)], s=120, facecolor="none", edgecolor="#1a9e5a", lw=1.6 if kk == 1 else 0.8)
    ax.scatter(R["lam"].real, R["lam"].imag, color="#2f62d6", s=35, zorder=3)
    for md in R["modes"]:
        tag = f"{NAME[md['hand']][0]}{md['k']}" if md["hand"] >= 0 else "?"
        ax.annotate(f"{tag} {md['deg']:.2f}°", (np.cos(np.deg2rad(md["deg"])) * md["abs"], np.sin(np.deg2rad(md["deg"])) * md["abs"]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8)
    ax.set_xlim(0.6, 1.12); ax.set_ylim(-0.45, 0.45); ax.set_aspect("equal")
    ax.set_title(f"{R['label']}  (d = {R['d']})\neigenvalues of K", fontsize=10, loc="left")
    for c, h in enumerate((0, 1)):
        ax = fig.add_subplot(gs[r, 1 + c])
        sc, md = R["best"][h]
        if md is not None:
            p = R["P"][:, md["j"]]; p = p / np.median(np.abs(p))
            ax.scatter(p.real, p.imag, c=cyc(ev_ang[:, h]), s=3)
            ax.set_title(f"best {NAME[h]} mode: {md['deg']:.2f}°/step\nphase fit to {NAME[h]} angle {sc:.2f}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[r, 3])
    strip = np.concatenate([np.concatenate([1 - gt[0, s] for s in SHOW], 1), np.concatenate([1 - R["pred0"][s] for s in SHOW], 1)], 0)
    ax.imshow(strip, cmap="gray", vmin=0, vmax=1); ax.set_xticks([32 + 64 * i for i in range(len(SHOW))])
    ax.set_xticklabels([f"+{s // 60}h{s % 60:02d}" for s in SHOW], fontsize=8); ax.set_yticks([32, 96]); ax.set_yticklabels(["truth", "Kᵗz₀"], fontsize=8)
    ax.set_title(f"rollout from one image · reconstruction MSE {R['rec_err']:.2f}×10⁻³", fontsize=9)
    ax = fig.add_subplot(gs[r, 4])
    ax.plot(np.arange(STEPS + 1) / 60, R["err"], color="#2f62d6", lw=2)
    ax.axhline(((gt.mean(0, keepdims=True) - gt) ** 2).mean() * 1000, color="#999", ls="--", lw=1)
    ax.set_ylim(0, 40); ax.set_xlabel("hours ahead", fontsize=8); ax.set_ylabel("pixel MSE ×10⁻³", fontsize=8)
    ax.set_title(f"error: {R['err'][60]:.2f} @1h, {R['err'][360]:.2f} @6h", fontsize=9)
plt.savefig(out_png, bbox_inches="tight")
json.dump([{"label": R["label"], "d": R["d"], "rec_err": R["rec_err"], "err_1h": float(R["err"][60]), "err_6h": float(R["err"][360]),
            "modes": [{k: v for k, v in md.items() if k != "j"} for md in R["modes"]],
            "best_fit": {NAME[h]: float(R["best"][h][0]) for h in (0, 1)}} for R in results],
          open(out_png.replace(".png", ".json"), "w"), indent=1)
print("\nwrote", out_png)

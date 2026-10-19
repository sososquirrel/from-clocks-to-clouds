"""What does the KL term bring? Same d = 4 pipeline with beta = 0 (plain autoencoder), 1 (VAE), 4 (strong KL).

For each beta: VAE stage (reconstruction, KL, sigma, latent scale, robustness to latent noise, prior samples)
and Koopman stage (modes, reconstruction, prediction error). Writes report/kl.json.

    bash analysis/kl_sweep.sh     # trains the models (~25 min)
    python -m analysis.kl_study
"""
import base64, io, json
import numpy as np
import torch
from PIL import Image
import clocks
from models import load_model, downsample

import os
KL = "runs/minimal/kl"                                                   # trained by analysis/kl_sweep.sh
RUNS = [(0, f"{KL}/vae_b0.pt", [f"{KL}/koopman_b0.pt", f"{KL}/koopman_b0_s1.pt"]),
        (0.25, f"{KL}/vae_b0p25.pt", [f"{KL}/koopman_b0p25.pt"]),
        (0.5, f"{KL}/vae_b0p5.pt", [f"{KL}/koopman_b0p5.pt"]),
        (1, f"{KL}/vae_b1.pt", [f"{KL}/koopman_b1.pt", f"{KL}/koopman_b1_s1.pt"]),
        (2, f"{KL}/vae_b2.pt", [f"{KL}/koopman_b2.pt"]),
        (4, f"{KL}/vae_b4.pt", [f"{KL}/koopman_b4.pt"])]
RUNS = [(b, v, [k for k in ks if os.path.exists(k)]) for b, v, ks in RUNS if os.path.exists(v)]
RUNS = [r for r in RUNS if r[2]]                                        # skip betas whose Koopman run is missing

TWO_PI = 2 * np.pi
rng = np.random.default_rng(0)
data = np.load("data/clocks.npz")
dt_h, dt_m = float(data["dt_hour"]), float(data["dt_min"])
ang = data["val_angles"][:, ::3].reshape(-1, 2)[:2000]
pool = lambda a: a.reshape(*a.shape[:-2], 32, 2, 32, 2).mean((-3, -1))
# one-step latent linearity on held-out trajectories
VT = data["val_frames"][:100]                                           # 100 held-out runs x 40 minutes
VT = pool(VT.reshape(-1, 64, 64).astype(np.float32) / 255).reshape(100, -1, 32, 32)
render = lambda th, tm: pool(clocks.render(np.asarray(th, np.float32) % TWO_PI, np.asarray(tm, np.float32) % TWO_PI))
X = render(ang[:, 0], ang[:, 1])
blank = float(((X.mean(0, keepdims=True) - X) ** 2).mean() * 1000)


def png(img, size=72):
    b = io.BytesIO()
    Image.fromarray((255 - np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


@torch.no_grad()
def enc(m, x):
    mu, lv = m.encode(torch.as_tensor(np.asarray(x, np.float32)[:, None])); return mu.numpy(), lv.numpy()


@torch.no_grad()
def dec(m, z):
    return np.concatenate([torch.sigmoid(m.decode(torch.as_tensor(c, dtype=torch.float32))).squeeze(1).numpy()
                           for c in np.array_split(np.atleast_2d(z), max(1, len(np.atleast_2d(z)) // 1000))])


# nearest real clock for generated images (grid search + local refine, at 32 px)
g = np.deg2rad(np.arange(0, 360, 4.0)).astype(np.float32)
GH, GM = [a.ravel() for a in np.meshgrid(g, g, indexing="ij")]
GRID = np.concatenate([render(GH[i:i + 1500], GM[i:i + 1500]).reshape(-1, 1024) for i in range(0, len(GH), 1500)])


def nearest_mse(imgs):
    f = imgs.reshape(len(imgs), -1)
    d = (f ** 2).sum(1)[:, None] - 2 * f @ GRID.T + (GRID ** 2).sum(1)[None]
    return d.min(1) / 1024


starts = data["val_angles"][:200, 0]
gt = np.stack([render(starts[:, 0] + s * dt_h, starts[:, 1] + s * dt_m) for s in (0, 60, 360)], 1)
fit = lambda p, th: abs(np.mean(np.exp(1j * th) * np.conj(p) / np.abs(p)))
out = []
def linearity(m, K=None):
    """Normalised one-step latent error on held-out runs; with K=None use the best least-squares map."""
    z = enc(m, VT.reshape(-1, 32, 32))[0].reshape(100, -1, 4)
    Z0, Z1 = z[:, :-1].reshape(-1, 4), z[:, 1:].reshape(-1, 4)
    M = np.linalg.lstsq(Z0, Z1, rcond=None)[0].T if K is None else K
    return float(((Z0 @ M.T - Z1) ** 2).sum(1).mean() / z.reshape(-1, 4).var(0).sum())


SHOW_I = rng.choice(len(X), 8, replace=False)                          # same 8 clocks for every beta
PCA_I = rng.choice(len(X), 600, replace=False)


def stage_stats(stage, path):
    m, _ = load_model(path)
    mu, lv = enc(m, X); sig = np.exp(0.5 * lv)
    rec = dec(m, mu)
    c_ = mu - mu.mean(0)
    pcs = c_ @ np.linalg.svd(c_, full_matrices=False)[2].T
    pcs = pcs / np.sqrt((pcs ** 2).sum(1).mean())
    vis = {"rec_imgs": [png(rec[i]) for i in SHOW_I],
           "pca": np.round(pcs[PCA_I], 2).tolist()}
    S = {"rec": float(((rec - X) ** 2).mean() * 1000),
         "kl": float((0.5 * (mu ** 2 + sig ** 2 - 1 - lv)).mean(0).sum()),
         "sigma": float(sig.mean()), "scale": float(np.sqrt((mu ** 2).mean()))}
    S["noise"] = {}
    for eps in (0.05, 0.1, 0.2):
        zn = mu + eps * S["scale"] * rng.standard_normal(mu.shape).astype(np.float32)
        S["noise"][str(eps)] = float(((dec(m, zn) - X) ** 2).mean() * 1000)
    zp = rng.standard_normal((600, 4)).astype(np.float32)
    gi = dec(m, zp); nm = nearest_mse(gi)
    S["gen_clean"] = float((nm < 1e-3).mean()); S["gen_median"] = float(np.median(nm) * 1000)
    S["gen_imgs"] = [png(im) for im in gi[:12]]
    S.update(vis)
    if stage == "vae":
        S["lin_best"] = linearity(m)                                    # how linear the plain VAE latent already is
        return S
    K = m.K.detach().numpy(); lam, V = np.linalg.eig(K); P = mu @ np.linalg.inv(V).T
    modes = []
    for j in range(4):
        if lam[j].imag <= 1e-9:
            continue
        p, q, f = max(((p, q, fit(P[:, j], p * ang[:, 0] + q * ang[:, 1])) for p in range(-14, 15) for q in range(-4, 5) if (p, q) != (0, 0)), key=lambda t: t[2])
        modes.append({"deg": float(np.degrees(np.angle(lam[j]))), "abs": float(abs(lam[j])), "p": int(p), "q": int(q), "fit": float(f)})
    S["modes"] = sorted(modes, key=lambda e: e["deg"])
    S["lin_K"] = linearity(m, K)
    z0 = enc(m, gt[:, 0])[0]
    pr = {k: dec(m, z0 @ np.linalg.matrix_power(K, k).T) for k in (60, 360)}
    S["err_1h"] = float(((pr[60] - gt[:, 1]) ** 2).mean() * 1000)
    S["err_6h"] = float(((pr[360] - gt[:, 2]) ** 2).mean() * 1000)
    S["roll"] = [png(gt[0, 0]), png(pr[60][0]), png(pr[360][0]), png(gt[0, 1]), png(gt[0, 2])]
    return S


for beta, vp, kps in RUNS:
    R = {"beta": beta, "vae": stage_stats("vae", vp), "seeds": [stage_stats("koop", kp) for kp in kps]}
    for s_ in R["seeds"][1:]:
        for k in ("rec_imgs", "pca", "gen_imgs", "roll"):
            s_.pop(k, None)
    mean = lambda k: float(np.mean([s_[k] for s_ in R["seeds"]]))
    R["koop"] = dict(R["seeds"][0]); R["koop"].update({k: mean(k) for k in ("rec", "kl", "err_1h", "err_6h", "lin_K", "sigma", "scale")})
    R["koop"]["noise"] = {e: float(np.mean([s_["noise"][e] for s_ in R["seeds"]])) for e in ("0.05", "0.1", "0.2")}
    R["koop"]["err_6h_seeds"] = [s_["err_6h"] for s_ in R["seeds"]]
    out.append(R)
    v, k_ = R["vae"], R["koop"]
    print(f"beta {beta:4}: VAE rec {v['rec']:.2f} KL {v['kl']:5.1f} lin(best A) {v['lin_best']:.3f} noise10% {v['noise']['0.1']:.2f} | "
          f"Koopman rec {k_['rec']:.2f} KL {k_['kl']:5.1f} lin(K) {k_['lin_K']:.4f} err 1h {k_['err_1h']:.2f} 6h {k_['err_6h']:.2f} (seeds {[round(e, 2) for e in k_['err_6h_seeds']]}) "
          f"modes {[(e['p'], e['q'], round(e['deg'], 3)) for e in k_['modes']]}")
json.dump({"runs": out, "blank": blank, "orig_imgs": [png(X[i]) for i in SHOW_I], "pca_ang": np.round(ang[PCA_I], 3).tolist()}, open("report/kl.json", "w"), separators=(",", ":"))
print("wrote report/kl.json")

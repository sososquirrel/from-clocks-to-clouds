"""What the stage-1 VAE (d = 16) learned: reconstructions, per-dimension KL, traversals,
posterior samples, prior samples, interpolation. Writes report/vae_explain.json for the report.

    python vae_explain.py
"""
import base64, io, json, re
import numpy as np
import torch
from PIL import Image

import clocks
from models import VAE

ck = torch.load("runs/vae.pt", map_location="cpu")
vae = VAE(ck["d"]); vae.load_state_dict(ck["state"], strict=False); vae.eval()
D = ck["d"]
data = np.load("data/clocks.npz")
rng = np.random.default_rng(1)
angles = rng.uniform(0, 2 * np.pi, (3000, 2)).astype(np.float32)          # fresh, uniformly random clocks
X = torch.as_tensor(clocks.render(angles[:, 0], angles[:, 1])[:, None])


def png(img, size=80):
    b = io.BytesIO()
    Image.fromarray((255 - np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def err_png(e, size=80):
    """Signed error map (amplified x4): red = missing ink, blue = extra ink, white = correct."""
    e = np.clip(e * 4, -1, 1)                                                   # x4 so small errors are visible
    rgb = np.ones(e.shape + (3,))
    rgb[..., 1] -= np.abs(e)
    rgb[..., 0] -= np.clip(-e, 0, 1)                                           # extra ink -> blue
    rgb[..., 2] -= np.clip(e, 0, 1)                                            # missing ink -> red
    b = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.NEAREST).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


dec = lambda z: torch.sigmoid(vae.decode(torch.as_tensor(np.atleast_2d(z), dtype=torch.float32))).squeeze(1).detach().numpy()
with torch.no_grad():
    mu, logvar = vae.encode(X)
    mu, logvar = mu.numpy(), logvar.numpy()
    rec = dec(mu)
x = X.squeeze(1).detach().numpy()
sigma = np.exp(0.5 * logvar)
fmt = lambda th, tm: clocks.angle_to_time(th, tm)

# ---- 1. reconstructions -----------------------------------------------------------------
err = ((rec - x) ** 2).mean((1, 2))
order = np.argsort(err)
pick_typ = rng.choice(len(x), 8, replace=False)
pick_worst = order[-4:][::-1]
recon = [{"x": png(x[i]), "r": png(rec[i]), "e": err_png(x[i] - rec[i]), "mse": float(err[i] * 1000),
          "t": fmt(*angles[i]), "kind": "typical"} for i in pick_typ] + \
        [{"x": png(x[i]), "r": png(rec[i]), "e": err_png(x[i] - rec[i]), "mse": float(err[i] * 1000),
          "t": fmt(*angles[i]), "kind": "worst"} for i in pick_worst]
err_stats = {"median": float(np.median(err) * 1000), "p95": float(np.percentile(err, 95) * 1000),
             "max": float(err.max() * 1000), "blank": float(((x.mean(0, keepdims=True) - x) ** 2).mean() * 1000)}

# showcase: 24 random clocks, and a picker over every hour x every 5 minutes (real clock times)
gallery = [{"x": png(x[i], 112), "r": png(rec[i], 112), "t": fmt(*angles[i]), "mse": float(err[i] * 1000)}
           for i in rng.choice(len(x), 24, replace=False)]
H_, M_ = np.meshgrid(np.arange(12), np.arange(0, 60, 5), indexing="ij")
g_th = ((H_ + M_ / 60) / 12 * 2 * np.pi).ravel().astype(np.float32)
g_tm = (M_ / 60 * 2 * np.pi).ravel().astype(np.float32)
gx = clocks.render(g_th, g_tm)
with torch.no_grad():
    gr = dec(vae.encode(torch.as_tensor(gx[:, None]))[0].numpy())
picker = [{"x": png(gx[i], 160), "r": png(gr[i], 160),
           "mse": float(((gr[i] - gx[i]) ** 2).mean() * 1000)} for i in range(len(gx))]

# ---- generation: decode z ~ N(0, tau^2 I); match every sample to its closest real clock -----
def nearest_clock(imgs):
    """Best (theta_h, theta_m, mse) per image: 4-degree grid search, then a 0.5-degree local refine."""
    g = np.deg2rad(np.arange(0, 360, 4.0)).astype(np.float32)
    GH, GM = np.meshgrid(g, g, indexing="ij"); GH, GM = GH.ravel(), GM.ravel()
    flat = imgs.reshape(len(imgs), -1)
    best = np.full(len(imgs), np.inf); bh = np.zeros(len(imgs)); bm = np.zeros(len(imgs))
    for i in range(0, len(GH), 1500):
        R = clocks.render(GH[i:i + 1500], GM[i:i + 1500]).reshape(-1, 4096)
        d = ((flat ** 2).sum(1)[:, None] - 2 * flat @ R.T + (R ** 2).sum(1)[None]) / 4096
        j = d.argmin(1); upd = d[np.arange(len(imgs)), j] < best
        best[upd] = d[np.arange(len(imgs)), j][upd]; bh[upd] = GH[i + j[upd]]; bm[upd] = GM[i + j[upd]]
    off = np.deg2rad(np.arange(-2, 2.01, 0.5)).astype(np.float32)
    OH, OM = np.meshgrid(off, off, indexing="ij"); OH, OM = OH.ravel(), OM.ravel()
    for k in range(len(imgs)):
        R = clocks.render((bh[k] + OH) % (2 * np.pi), (bm[k] + OM) % (2 * np.pi)).reshape(-1, 4096)
        d = ((R - flat[k]) ** 2).mean(1); j = d.argmin()
        if d[j] < best[k]:
            best[k], bh[k], bm[k] = d[j], (bh[k] + OH[j]) % (2 * np.pi), (bm[k] + OM[j]) % (2 * np.pi)
    return bh, bm, best


VALID = 1e-3          # a sample counts as a clean clock if some real clock is within this pixel MSE
gen = {}
schemes = {"prior": lambda n: rng.standard_normal((n, D)),
           "prior_half": lambda n: 0.5 * rng.standard_normal((n, D)),
           "aggregate": lambda n: (lambda i: mu[i] + sigma[i] * rng.standard_normal((n, D)))(rng.integers(0, len(mu), n))}
# minimal 4-D Koopman model: its latent is two circles, so sample one random angle per circle
from models import KoopmanVAE
ckm = torch.load("runs/koopman_min.pt", map_location="cpu")
km = KoopmanVAE(4); km.load_state_dict(ckm["state"]); km.eval()
dec4 = lambda z: torch.sigmoid(km.decode(torch.as_tensor(np.atleast_2d(z), dtype=torch.float32))).squeeze(1).detach().numpy()
lam4, V4 = np.linalg.eig(km.K.detach().numpy())
with torch.no_grad():
    phi4 = km.encode(X)[0].numpy() @ np.linalg.inv(V4).T                      # eigen-coordinates of real clocks
pos4 = [j for j in range(4) if lam4[j].imag > 0]
conj4 = {j: int(np.argmin(np.abs(lam4 - np.conj(lam4[j])))) for j in pos4}
rad4 = {j: float(np.median(np.abs(phi4[:, j]))) for j in pos4}


def circles(n):
    p = np.zeros((n, 4), complex)
    for j in pos4:                                                              # one uniform angle per hand circle
        w = rad4[j] * np.exp(1j * rng.uniform(0, 2 * np.pi, n))
        p[:, j], p[:, conj4[j]] = w, np.conj(w)
    return (p @ V4.T).real


schemes["circles"] = circles
# plain 4-D VAE, sampled from its prior
ck4 = torch.load("runs/vae_d4.pt", map_location="cpu")
v4 = VAE(4); v4.load_state_dict(ck4["state"], strict=False); v4.eval()
dec_v4 = lambda z: torch.sigmoid(v4.decode(torch.as_tensor(np.atleast_2d(z), dtype=torch.float32))).squeeze(1).detach().numpy()
schemes["prior_d4"] = lambda n: rng.standard_normal((n, 4))
for tau, sampler in schemes.items():
    zs = sampler(600).astype(np.float32)
    dfn = {"circles": dec4, "prior_d4": dec_v4}.get(tau, dec)
    imgs = np.concatenate([dfn(zs[i:i + 200]) for i in range(0, len(zs), 200)])
    bh, bm, mse = nearest_clock(imgs)
    gen[tau] = {"valid": float((mse < VALID).mean()), "median": float(np.median(mse) * 1000),
                "loose": float((mse < 3e-3).mean()),
                     "samples": [{"img": png(imgs[i], 96), "t": fmt(bh[i], bm[i]), "ok": bool(mse[i] < VALID)} for i in range(24)]}
    print(f"{tau:10s}: {100 * (mse < VALID).mean():5.1f}% within 1e-3, {100 * (mse < 3e-3).mean():5.1f}% within 3e-3, median {np.median(mse) * 1000:.2f}e-3")

# ---- 2. per-dimension KL and what each dimension encodes ----------------------------------
kl_dim = (0.5 * (mu ** 2 + sigma ** 2 - 1 - logvar)).mean(0)                # nats per image, per dimension
feats = np.c_[np.cos(angles[:, 0]), np.sin(angles[:, 0]), np.cos(angles[:, 1]), np.sin(angles[:, 1])]
# how much of each dimension is explained by hour / minute (harmonics 1..4), linear least squares R^2
def r2(y, F):
    F1 = np.c_[F, np.ones(len(F))]
    res = y - F1 @ np.linalg.lstsq(F1, y, rcond=None)[0]
    return float(1 - res.var() / y.var())
harm = lambda th: np.concatenate([np.c_[np.cos(k * th), np.sin(k * th)] for k in range(1, 5)], 1)
dims = []
for j in range(D):
    dims.append({"kl": float(kl_dim[j]), "sigma": float(sigma[:, j].mean()), "mu_std": float(mu[:, j].std()),
                 "r2_hour": r2(mu[:, j], harm(angles[:, 0])), "r2_min": r2(mu[:, j], harm(angles[:, 1])),
                 "r2_both": r2(mu[:, j], np.c_[harm(angles[:, 0]), harm(angles[:, 1])])})
active = [j for j in np.argsort(-kl_dim) if kl_dim[j] > 0.1]

# traversals: one base clock, move one dimension at a time from -2.5 to +2.5 (prior units)
base_i = int(np.argmin(np.abs(mu).sum(1)))                                     # a clock near the latent centre
steps = np.linspace(-2.5, 2.5, 9)
trav = []
for j in active:
    zs = np.repeat(mu[base_i:base_i + 1], len(steps), 0); zs[:, j] = steps
    trav.append({"dim": int(j), "imgs": [png(im, 64) for im in dec(zs)]})

# ---- 3. KL: posterior samples, prior samples, interpolation, curves ------------------------
post_i = int(pick_typ[0])
eps = rng.standard_normal((8, D)).astype(np.float32)
post = [png(im, 64) for im in dec(mu[post_i] + eps * sigma[post_i])]
prior = [png(im, 64) for im in dec(rng.standard_normal((16, D)).astype(np.float32))]
a_i, b_i = int(pick_typ[1]), int(pick_typ[2])
lerp = np.linspace(0, 1, 9)[:, None]
interp = [png(im, 64) for im in dec((1 - lerp) * mu[a_i] + lerp * mu[b_i])]
# what a linear path in *angle space* looks like (the "right" interpolation)
dth = (angles[b_i] - angles[a_i] + np.pi) % (2 * np.pi) - np.pi
ang_path = angles[a_i] + lerp * dth
interp_true = [png(im, 64) for im in clocks.render(ang_path[:, 0] % (2 * np.pi), ang_path[:, 1] % (2 * np.pi))]

curve = [tuple(map(float, m.groups())) for m in re.finditer(r"epoch\s+(\d+)/\d+\s+recon\s+([\d.]+)\s+kl\s+([\d.]+)",
                                                            open("runs/vae_log.txt").read())]

out = {"d": D, "gen": gen, "gen_valid_thr": VALID, "gallery": gallery, "picker": picker, "err": err_stats, "dims": dims, "active": [int(j) for j in active],
       "total_kl": float(kl_dim.sum()),
       "trav": trav, "trav_steps": steps.tolist(), "base_t": fmt(*angles[base_i]), "base": png(x[base_i], 64),
       "post": post, "post_x": png(x[post_i], 64), "post_sigma": sigma[post_i].round(3).tolist(), "post_t": fmt(*angles[post_i]),
       "prior": prior, "interp": interp, "interp_true": interp_true,
       "interp_ends": [fmt(*angles[a_i]), fmt(*angles[b_i])],
       "curve": [{"epoch": int(e), "recon": r, "kl": k} for e, r, k in curve]}
json.dump(out, open("report/vae_explain.json", "w"), separators=(",", ":"))
print(f"active dims (KL > 0.1 nats): {len(active)} of {D}: {active}")
print("KL per dim:", np.round(kl_dim, 2), " total", round(kl_dim.sum(), 1))
for j in active:
    d_ = dims[j]; print(f"  z{j:<2d} KL {d_['kl']:.2f}  sigma {d_['sigma']:.3f}  R2 hour {d_['r2_hour']:.2f}  minute {d_['r2_min']:.2f}  both {d_['r2_both']:.2f}")
print("recon MSE x1e-3:", {k: round(v, 2) for k, v in err_stats.items()})

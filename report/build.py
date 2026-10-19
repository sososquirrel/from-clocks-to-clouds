"""Data for the dedicated report on the minimal d = 4 Koopman VAE (trained from scratch, fast setup).

Stage 1: runs/minimal/vae.pt (plain VAE, 4 latent numbers, 32x32 images)
Stage 2: runs/minimal/koopman.pt (same network + Koopman loss, 12,000 iterations)
Reads report/timeline.json (analysis.timeline), report/kl.json (analysis.kl_study) and
report/media/modes.mp4 (analysis.mode_video); writes report/data.json and report/index.html,
and the same data split into one file per visualisation in viz/ (for slides, see viz/README.md).

    python -m report.build
"""
import base64, io, json, os, shutil
import numpy as np
import torch
from PIL import Image
import clocks
from models import load_model, downsample

OUT = "report"
os.makedirs(f"{OUT}/media", exist_ok=True)
rng = np.random.default_rng(3)
r3 = lambda a: np.round(np.asarray(a, dtype=float), 3).tolist()
TWO_PI = 2 * np.pi
data = np.load("data/clocks.npz")
dt_h, dt_m = float(data["dt_hour"]), float(data["dt_min"])
ang = data["val_angles"][:, ::3].reshape(-1, 2)[:2000]

vae, ckv = load_model("runs/minimal/vae.pt")
kvae, ckk = load_model("runs/minimal/koopman.pt")
S = ckk["size"]
pool = lambda a: a.reshape(*a.shape[:-2], S, 64 // S, S, 64 // S).mean((-3, -1)) if S != 64 else a
render = lambda th, tm: pool(clocks.render(np.asarray(th, np.float32) % TWO_PI, np.asarray(tm, np.float32) % TWO_PI))


def png(img, size=96):
    b = io.BytesIO()
    Image.fromarray((255 - np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


@torch.no_grad()
def enc(m, imgs):
    mu, lv = m.encode(torch.as_tensor(np.asarray(imgs, np.float32)[:, None]))
    return mu.numpy(), lv.numpy()


@torch.no_grad()
def dec(m, z):
    return torch.sigmoid(m.decode(torch.as_tensor(np.atleast_2d(z), dtype=torch.float32))).squeeze(1).numpy()


fmt = lambda th, tm: clocks.angle_to_time(th % TWO_PI, tm % TWO_PI)
X = render(ang[:, 0], ang[:, 1])

# =========================================================== 1. the VAE (stage 1)
mu_v, lv_v = enc(vae, X)
rec_v = dec(vae, mu_v)
err_v = ((rec_v - X) ** 2).mean((1, 2))
blank = float(((X.mean(0, keepdims=True) - X) ** 2).mean() * 1000)
H_, M_ = np.meshgrid(np.arange(12), np.arange(0, 60, 5), indexing="ij")
g_th, g_tm = ((H_ + M_ / 60) / 12 * TWO_PI).ravel(), (M_ / 60 * TWO_PI).ravel()
gx = render(g_th, g_tm)
gr = dec(vae, enc(vae, gx)[0])
picker = [{"x": png(gx[i], 160), "r": png(gr[i], 160), "mse": float(((gr[i] - gx[i]) ** 2).mean() * 1000)} for i in range(len(gx))]
gal_i = rng.choice(len(X), 16, replace=False)
gallery = [{"x": png(X[i], 112), "r": png(rec_v[i], 112), "t": fmt(*ang[i])} for i in gal_i]
sig_v = np.exp(0.5 * lv_v)
kl_dim = (0.5 * (mu_v ** 2 + sig_v ** 2 - 1 - lv_v)).mean(0)
base_i = int(np.argmin(np.abs(mu_v).sum(1)))
steps = np.linspace(-2.5, 2.5, 9)
trav = []
for j in range(4):
    zs = np.repeat(mu_v[base_i:base_i + 1], 9, 0); zs[:, j] = steps
    trav.append([png(im, 72) for im in dec(vae, zs)])
vae_sec = {"picker": picker, "gallery": gallery, "err_median": float(np.median(err_v) * 1000), "err_p95": float(np.percentile(err_v, 95) * 1000),
           "blank": blank, "kl": r3(kl_dim), "sigma": r3(sig_v.mean(0)), "trav": trav, "trav_steps": steps.tolist()}

# =========================================================== 3. PCA + trajectory, VAE vs Koopman
STEPS = 360
pa = data["val_angles"][0, 0]
tr = np.arange(STEPS + 1)
th_p, tm_p = (pa[0] + tr * dt_h) % TWO_PI, (pa[1] + tr * dt_m) % TWO_PI
xp = render(th_p, tm_p)


def pca_block(model, with_K):
    mu, _ = enc(model, X)
    zp, _ = enc(model, xp)
    mean = mu.mean(0)
    _, Sv, Vt = np.linalg.svd(mu - mean, full_matrices=False)
    out = {"pc": r3((mu - mean) @ Vt.T), "probe": r3((zp - mean) @ Vt.T), "var": r3(Sv ** 2 / (Sv ** 2).sum())}
    if with_K:
        K = model.K.detach().numpy()
        lam, V = np.linalg.eig(K); c = np.linalg.inv(V) @ zp[0]
        tt = np.arange(0, STEPS + 1, 2); al = np.linspace(0, TWO_PI, 73)
        modes = []
        for j in range(4):
            if lam[j].imag <= 1e-9:
                continue
            term = 2.0 * np.real((lam[j] ** tt)[:, None] * c[j] * V[:, j][None, :])
            ell = 2.0 * np.real(np.exp(1j * al)[:, None] * c[j] * V[:, j][None, :])
            deg = float(np.degrees(np.angle(lam[j])))
            modes.append({"deg": deg, "abs": float(abs(lam[j])), "hand": 0 if deg < 3 else 1,
                          "pc": r3(term @ Vt.T), "ell": r3(ell @ Vt.T), "amp": float(np.abs(term @ Vt.T).max())})
        modes.sort(key=lambda e: -e["amp"])
        roll = [zp[0]]
        for _ in range(STEPS):
            roll.append(roll[-1] @ K.T)
        out["epi"] = {"origin": r3(-mean @ Vt.T), "modes": modes, "roll_pc": r3(((np.array(roll) - mean) @ Vt.T)[::2])}
    return out


pca = {"vae": pca_block(vae, False), "koop": pca_block(kvae, True), "probe_th": r3(th_p), "probe_tm": r3(tm_p)}

# prediction from one image, 200 held-out clocks: Koopman K vs best linear map A on the plain VAE latent
starts = data["val_angles"][:200, 0]
gt = np.stack([render(starts[:, 0] + s * dt_h, starts[:, 1] + s * dt_m) for s in range(STEPS + 1)], 1)
tf = data["train_frames"]
mu_tr = enc(vae, downsample(tf.reshape(-1, 64, 64), S).astype(np.float32) / 255)[0].reshape(len(tf), tf.shape[1], 4)
A = np.linalg.lstsq(mu_tr[:, :-1].reshape(-1, 4), mu_tr[:, 1:].reshape(-1, 4), rcond=None)[0].T


def rollout(model, M):
    z = enc(model, gt[:, 0])[0]
    zs = [z]
    for _ in range(STEPS):
        zs.append(zs[-1] @ M.T)
    zs = np.stack(zs, 1).reshape(-1, 4)
    return np.concatenate([dec(model, c) for c in np.array_split(zs, 20)]).reshape(len(z), STEPS + 1, S, S)


K = kvae.K.detach().numpy()
pk, pv = rollout(kvae, K), rollout(vae, A)
err = {"koop": r3(((pk - gt) ** 2).mean((0, 2, 3)) * 1000), "vae": r3(((pv - gt) ** 2).mean((0, 2, 3)) * 1000), "blank": blank}
SHOW = [0, 30, 60, 120, 240, 360]
film = {"steps": SHOW, "truth": [png(gt[0, s]) for s in SHOW], "koop": [png(pk[0, s]) for s in SHOW], "vae": [png(pv[0, s]) for s in SHOW]}

# =========================================================== 4. spectrum
lam, V = np.linalg.eig(K)
W = np.linalg.inv(V)
mu_k = enc(kvae, X)[0]
P = mu_k @ W.T
fit = lambda p, th: float(abs(np.mean(np.exp(1j * th) * np.conj(p) / (np.abs(p) + 1e-12))))
spec = []
for j in range(4):
    best = max(((p, q, fit(P[:, j], p * ang[:, 0] + q * ang[:, 1])) for p in range(-14, 15) for q in range(-4, 5) if (p, q) != (0, 0)),
               key=lambda t: t[2])
    spec.append({"re": float(lam[j].real), "im": float(lam[j].imag), "deg": float(np.degrees(np.angle(lam[j]))), "abs": float(abs(lam[j])),
                 "p": int(best[0]), "q": int(best[1]), "fit": best[2]})
spec_A = [{"re": float(l.real), "im": float(l.imag)} for l in np.linalg.eigvals(A)]

# =========================================================== 5. main modes: projections, orbits, turning a mode
pos = [j for j in range(4) if lam[j].imag > 0]
ja = max(pos, key=lambda j: fit(P[:, j], ang[:, 0]))
jb = [j for j in pos if j != ja][0]
cj = {j: int(np.argmin(np.abs(lam - np.conj(lam[j])))) for j in pos}
pb, qb = spec[jb]["p"], spec[jb]["q"]
align = lambda v, th: v * (lambda c: c / abs(c))(np.mean(np.exp(1j * th) * np.conj(v) / np.abs(v))) / np.median(np.abs(v))
OI = [0, 400, 800, 1200, 1600]
orb_z = [mu_k[OI]]
for _ in range(STEPS):
    orb_z.append(orb_z[-1] @ K.T)
orb_P = (np.stack(orb_z, 1) @ W.T)                                    # (5, STEPS+1, 4)
deg_a, deg_b = float(np.degrees(np.angle(lam[ja]))), float(np.degrees(np.angle(lam[jb])))
z0 = mu_k[int(np.argmin(np.abs(ang[:, 0] - np.deg2rad(300)) + np.abs(ang[:, 1] - np.deg2rad(60))))]


def turn(da, db):
    p = W @ z0
    p[ja] *= np.exp(1j * da); p[cj[ja]] *= np.exp(-1j * da)
    p[jb] *= np.exp(1j * db); p[cj[jb]] *= np.exp(-1j * db)
    return (V @ p).real


ROT = np.deg2rad(np.arange(0, 361, 45))
mode_defs = [
    ("alpha", f"α · mode (1, 0)", deg_a, lambda Q: Q[..., ja], 0, lambda d: turn(d, 0), "learned mode: turning it moves the hour hand forward and the minute hand backward"),
    ("beta", f"β · mode ({pb}, {qb})", deg_b, lambda Q: Q[..., jb], 1, lambda d: turn(0, d), "learned mode: turning it moves only the minute hand"),
    ("hour", "natural hour · α", deg_a, lambda Q: Q[..., ja], 0, lambda d: turn(d, d), "change of basis: turn α and β together, only the hour hand moves"),
    ("minute", "natural minute · β · conj(α)", deg_b - deg_a, lambda Q: Q[..., jb] * np.conj(Q[..., ja]), 1, lambda d: turn(0, d),
     "change of basis: phase of β minus phase of α; turning it moves only the minute hand"),
]
modes_sec = []
ang3 = np.c_[ang, (ang[:, 0] + ang[:, 1]) % TWO_PI]                   # column 2 = theta_h + theta_m (what beta tracks)
mode_defs[1] = mode_defs[1][:4] + (2,) + mode_defs[1][5:]
for key, name, deg, getv, hand, turnf, note in mode_defs:
    v = getv(P)
    cal = np.mean(np.exp(1j * ang3[:, hand]) * np.conj(v) / np.abs(v)); cal = cal / abs(cal) / np.median(np.abs(v))
    pts, orbs = v * cal, getv(orb_P) * cal
    modes_sec.append({"key": key, "name": name, "deg": deg, "hand": hand, "note": note, "fit": fit(v, ang3[:, hand]),
                      "loop": 360 / deg, "pts": r3(np.c_[pts.real, pts.imag]),
                      "orbits": [r3(np.c_[o.real, o.imag][::2]) for o in orbs],
                      "turn": [png(im, 72) for im in dec(kvae, [turnf(d) for d in ROT])]})
orbit_starts = [fmt(*ang[i]) for i in OI]

# =========================================================== 2. training timeline (built by analysis/timeline.py)
tl = json.load(open(f"{OUT}/timeline.json"))

# the alpha / beta recipes: complex weights on (z1..z4), calibrated so the angle of alpha(z) is theta_h
# and the angle of beta(z) is theta_h + theta_m (measured from the x axis, counter-clockwise), median |.| = 1
def recipe(j, th):
    v = P[:, j]
    c = np.mean(np.exp(1j * th) * np.conj(v) / np.abs(v)); c = c / abs(c) / np.median(np.abs(v))
    w = W[j] * c
    return {"x": r3(w.real), "y": r3(w.imag), "deg": float(np.degrees(np.angle(lam[j])))}


recipes = {"alpha": recipe(ja, ang[:, 0]), "beta": recipe(jb, ang[:, 0] + ang[:, 1])}
ex_i = int(np.argmin(np.abs(ang[:, 0] - np.deg2rad(60)) + np.abs(ang[:, 1] - np.deg2rad(0))))
ex_z = mu_k[ex_i]
ex = {"t": fmt(*ang[ex_i]), "z": r3(ex_z), "th": float(np.degrees(ang[ex_i, 0])), "tm": float(np.degrees(ang[ex_i, 1]))}
for k_ in ("alpha", "beta"):
    w_ = np.array(recipes[k_]["x"]) + 1j * np.array(recipes[k_]["y"]); f_ = w_ @ ex_z
    ex[k_] = {"x": float(f_.real), "y": float(f_.imag), "angle": float(np.degrees(np.angle(f_)) % 360)}

kl = json.load(open(f"{OUT}/kl.json")) if os.path.exists(f"{OUT}/kl.json") else None      # written by kl_study.py
out = {"kl_study": kl, "recipes": recipes, "example": ex, "ang": r3(ang3), "tl_idx": tl["idx"], "vae": vae_sec, "pca": pca, "err": err, "film": film, "spec": spec, "spec_A": spec_A,
       "modes": modes_sec, "orbit_starts": orbit_starts, "dt": {"h": np.degrees(dt_h), "m": np.degrees(dt_m)},
       "losses": {"vae_rec": float(np.median(err_v) * 1000)}}
json.dump(out, open(f"{OUT}/data.json", "w"), separators=(",", ":"))
print("report/data.json", os.path.getsize(f"{OUT}/data.json") // 1024, "KB")
# ---- the same data, one file per visualisation, for reuse in slides (see viz/README.md) -----------------
VIZ = "viz"
os.makedirs(f"{VIZ}/media", exist_ok=True)
dump = lambda name, obj: json.dump(obj, open(f"{VIZ}/{name}.json", "w"), separators=(",", ":"))
dump("common", {"angles": out["ang"], "dt_deg": out["dt"],
                "note": "angles[i] = [theta_h, theta_m, theta_h + theta_m] in radians for the 2,000 held-out clocks"})
dump("vae", out["vae"])
dump("latent", {"vae": out["pca"]["vae"], "koopman": out["pca"]["koop"],
                "probe_theta_h": out["pca"]["probe_th"], "probe_theta_m": out["pca"]["probe_tm"]})
dump("prediction", {"error": out["err"], "film": out["film"]})
dump("spectrum", {"eigenvalues": out["spec"], "eigenvalues_vae_linear_fit": out["spec_A"],
                  "recipes": out["recipes"], "example": out["example"]})
dump("modes", {"modes": out["modes"], "orbit_starts": out["orbit_starts"]})
for name, src in (("timeline.json", f"{OUT}/timeline.json"), ("kl.json", f"{OUT}/kl.json"),
                  ("media/modes.mp4", f"{OUT}/media/modes.mp4"), ("media/modes.jpg", f"{OUT}/media/modes.jpg")):
    if os.path.exists(src):
        shutil.copy(src, f"{VIZ}/{name}")
print("viz/:", ", ".join(sorted(f for f in os.listdir(VIZ) if f.endswith(".json"))))

template = open(f"{OUT}/template.html").read()
open(f"{OUT}/index.html", "w").write(template.replace("__DATA__", open(f"{OUT}/data.json").read().replace("</", "<\\/")))
print("report/index.html", os.path.getsize(f"{OUT}/index.html") // 1024, "KB")
print("spectrum:", [(round(s["deg"], 3), round(s["abs"], 4), (s["p"], s["q"]), round(s["fit"], 2)) for s in spec])
print("errors 1h/6h  koop", err["koop"][60], err["koop"][360], " vae+A", err["vae"][60], err["vae"][360], " blank", round(blank, 2))
print("VAE recon median", round(vae_sec["err_median"], 3), "e-3 | KL per dim", vae_sec["kl"])
print("mode fits:", [(m["key"], round(m["fit"], 2), round(m["loop"], 1)) for m in modes_sec])

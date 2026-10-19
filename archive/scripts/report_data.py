"""Export everything the HTML report needs into report/data.json (+ images as data URIs).

    python report_data.py
"""
import base64, io, json, os
import numpy as np
import torch
from PIL import Image

import animate as A          # loads models, Koopman eigen-decomposition, hand-mode recipes
import clocks
from models import KoopmanVAE, VAE

os.makedirs("report", exist_ok=True)
r3 = lambda x: np.round(np.asarray(x, dtype=float), 3).tolist()
ang = A.eval_angles

# ---- latent spaces ---------------------------------------------------------------
vae_mu = A.ks["vae_mu"]                                        # stage-1 latent of the eval set
mean = vae_mu.mean(0)
_, S, Vt = np.linalg.svd(vae_mu - mean, full_matrices=False)
pca = lambda m: (m - mean) @ Vt[:4].T
vae_pc = pca(vae_mu)
mh, mm = A.modes(A.ks["mu"][-1])

# a probe clock followed for 6 hours, encoded by both models
STEPS = 360
pa = A.ks["probe_angles"][0]
tr = np.arange(STEPS + 1)
th, tm = (pa[0] + tr * A.dt_h) % (2 * np.pi), (pa[1] + tr * A.dt_m) % (2 * np.pi)
x = torch.as_tensor(clocks.render(th, tm)[:, None])
with torch.no_grad():
    z_vae = A.vae.encode(x)[0].numpy()
    z_k = A.kvae.encode(x)[0].numpy()
ph, pm = A.modes(z_k)
probe = {"th": r3(th), "tm": r3(tm), "vae_pc": r3(pca(z_vae)),
         "k_h": r3(np.c_[ph.real, ph.imag]), "k_m": r3(np.c_[pm.real, pm.imag])}

# ---- eigenvalues -----------------------------------------------------------------
ck_pair = torch.load("runs/koopman_pair.pt", map_location="cpu")
kp = KoopmanVAE(ck_pair["d"]); kp.load_state_dict(ck_pair["state"]); kp.eval()
eig = lambda M: [[float(l.real), float(l.imag)] for l in np.linalg.eigvals(M)]
labels = {}
for j, (h, k, sc) in A.ASSIGN.items():
    labels[j] = f"{A.HAND_NAME[h]} k={k}"
eig_k = [{"re": float(l.real), "im": float(l.imag), "label": labels.get(j, labels.get(A.CONJ.get(j, -1), "")),
          "deg": float(np.degrees(np.angle(l))), "abs": float(abs(l))} for j, l in enumerate(A.lam)]
for j, (h, k, sc) in A.ASSIGN.items():                         # label the conjugates too
    eig_k[A.CONJ[j]]["label"] = f"{A.HAND_NAME[h]} k={k}"
assign = sorted([{"hand": A.HAND_NAME[h], "k": k, "deg": float(np.degrees(np.angle(A.lam[j]))),
                  "abs": float(abs(A.lam[j])), "score": float(sc)} for j, (h, k, sc) in A.ASSIGN.items()],
                key=lambda t: (t["hand"] != "hour", t["k"]))

# ---- long-horizon error, 200 held-out clocks ----------------------------------------
data = np.load("data/clocks.npz")
starts = data["val_angles"][:200, 0]
gt = np.stack([clocks.render((starts[:, 0] + s * A.dt_h) % (2 * np.pi), (starts[:, 1] + s * A.dt_m) % (2 * np.pi))
               for s in range(STEPS + 1)], 1)


@torch.no_grad()
def roll(model, M, z0):
    zs = [z0]
    for _ in range(STEPS):
        zs.append(zs[-1] @ M.T)
    zs = np.stack(zs, 1).reshape(-1, z0.shape[1])
    out = torch.cat([torch.sigmoid(model.decode(torch.as_tensor(c, dtype=torch.float32))) for c in np.array_split(zs, 20)])
    return out.view(len(z0), STEPS + 1, 64, 64).numpy()


x0 = torch.as_tensor(gt[:, 0, None])
with torch.no_grad():
    z0k, z0p, z0v = (m.encode(x0)[0].numpy() for m in (A.kvae, kp, A.vae))
rolls = {"koopman": roll(A.kvae, A.K, z0k), "pairs": roll(kp, kp.K.detach().numpy(), z0p),
         "vae_linear": roll(A.vae, A.A_base, z0v)}
err = {k: r3(((v - gt) ** 2).mean((0, 2, 3)) * 1000) for k, v in rolls.items()}   # x1e-3
err["average_clock"] = float(((gt.mean(0, keepdims=True) - gt) ** 2).mean() * 1000)


def png(img):
    b = io.BytesIO()
    Image.fromarray((255 - np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((96, 96), Image.BILINEAR).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


SHOW = [0, 30, 60, 120, 240, 360]
film = {"steps": SHOW, "truth": [png(gt[0, s]) for s in SHOW]}
for k, v in rolls.items():
    film[k] = [png(v[0, s]) for s in SHOW]

# ---- training curves ---------------------------------------------------------------
vs, ks = A.vs, A.ks
curves = {"vae_it": vs["it"][1:].tolist(), "vae_loss": r3(vs["loss"][1:]),
          "k_it": ks["it"][1:].tolist(), "k_pred": r3(ks["loss_pred"][1:]), "k_rec": r3(ks["loss_rec"][1:])}

# ---- Koopman latent: PCA (same recipe as for the VAE) --------------------------------
k_mu = A.ks["mu"][-1]
k_mean = k_mu.mean(0)
_, kS, kVt = np.linalg.svd(k_mu - k_mean, full_matrices=False)
kpca = lambda m: (m - k_mean) @ kVt[:4].T
probe["k_pc"] = r3(kpca(z_k))

# ---- every complex eigenmode: projection of all clocks + orbits under K ---------------
P_all = A.phi(k_mu)
ORBIT_IDX = [0, 400, 800, 1200, 1600]                          # a few held-out clocks to roll out
orb_z = [k_mu[ORBIT_IDX]]
for _ in range(STEPS):
    orb_z.append(orb_z[-1] @ A.K.T)
orb_phi = A.phi(np.stack(orb_z, 1).reshape(-1, A.D)).reshape(len(ORBIT_IDX), STEPS + 1, -1)
emodes = []
for j in range(len(A.lam)):
    if A.lam[j].imag <= 1e-6:
        continue
    if j in A.ASSIGN:
        h, k, sc = A.ASSIGN[j]
        theta = k * ang[:, h]
        c = np.mean(np.exp(1j * theta) * np.conj(P_all[:, j]) / np.abs(P_all[:, j]))
        rot, label = c / abs(c), f"{A.HAND_NAME[h]} · k = {k}"
    else:
        h, k, sc, rot, label = -1, 0, 0.0, 1.0, "unassigned (decaying)"
    norm = rot / np.median(np.abs(P_all[:, j]))
    p, o = P_all[:, j] * norm, orb_phi[:, :, j] * norm
    pr = A.phi(z_k)[:, j] * norm                                   # the followed clock, encoded each minute
    emodes.append({"label": label, "hand": int(h), "k": int(k), "deg": float(np.degrees(np.angle(A.lam[j]))),
                   "abs": float(abs(A.lam[j])), "score": float(sc),
                   "pts": r3(np.c_[p.real, p.imag]), "probe": r3(np.c_[pr.real, pr.imag]), "orbits": [r3(np.c_[q.real, q.imag][::2]) for q in o]})
emodes.sort(key=lambda e: (e["hand"] < 0, e["hand"], e["k"]))

# ---- Koopman mode decomposition of the followed clock in the PCA planes ---------------
#   K^t z0 = sum_j lam_j^t phi_j(z0) v_j ; each conjugate pair -> real term 2 Re(lam^t phi v)
z0 = z_k[0]
c0 = A.Winv @ z0
tt = np.arange(0, STEPS + 1, 2)
proj = lambda v: v @ kVt[:4].T                                  # linear part of the PCA map
epi = []
for j in range(len(A.lam)):
    if A.lam[j].imag < -1e-9:
        continue
    mult = 2.0 if A.lam[j].imag > 1e-9 else 1.0
    term = mult * np.real((A.lam[j] ** tt)[:, None] * c0[j] * A.V[:, j][None, :])   # (T, d)
    h, k, _ = A.ASSIGN.get(j, (-1, 0, 0))
    lab = f"{A.HAND_NAME[h]} k={k}" if h >= 0 else ("decaying" if abs(A.lam[j]) < 0.97 else "other")
    al = np.linspace(0, 2 * np.pi, 73)
    ell = mult * np.real(np.exp(1j * al)[:, None] * c0[j] * A.V[:, j][None, :]) if mult > 1 else term[:1]
    epi.append({"label": lab, "hand": int(h), "deg": float(np.degrees(np.angle(A.lam[j]))),
                "abs": float(abs(A.lam[j])), "ell": r3(proj(ell)),
                "pc": r3(proj(term)), "amp": float(np.abs(proj(term)).max())})
epi.sort(key=lambda e: -e["amp"])
kroll = [z0]
for _ in range(STEPS):
    kroll.append(kroll[-1] @ A.K.T)
epi_out = {"origin": r3(-k_mean @ kVt[:4].T), "modes": epi, "roll_pc": r3(kpca(np.array(kroll))[::2])}

# ---- plain VAE with d = 4 (stage 1 only) ------------------------------------------------
ck_v4 = torch.load("runs/vae_d4.pt", map_location="cpu")
v4 = VAE(4); v4.load_state_dict(ck_v4["state"], strict=False); v4.eval()
with torch.no_grad():
    mu_v4 = v4.encode(torch.as_tensor(clocks.render(ang[:, 0], ang[:, 1])[:, None]))[0].numpy()
    pr_v4 = v4.encode(x)[0].numpy()
v4_mean = mu_v4.mean(0)
_, v4S, v4Vt = np.linalg.svd(mu_v4 - v4_mean, full_matrices=False)
vae4 = {"pc": r3((mu_v4 - v4_mean) @ v4Vt.T), "probe": r3((pr_v4 - v4_mean) @ v4Vt.T), "var": r3(v4S ** 2 / (v4S ** 2).sum())}

# ---- minimal d = 4 trained from scratch (fast setup, 32 px, 12,000 it.): PCA + epicycles --------------
from models import load_model, downsample


def pca_row(path):
    """PCA of a Koopman model's latent, the followed clock, and its mode decomposition in the PCA planes."""
    mk, ckk = load_model(path)
    Sk = ckk.get("size", 64)
    pool = lambda a: a if Sk == 64 else a.reshape(*a.shape[:-2], Sk, 64 // Sk, Sk, 64 // Sk).mean((-3, -1))
    with torch.no_grad():
        mu_ = mk.encode(torch.as_tensor(pool(clocks.render(ang[:, 0], ang[:, 1]))[:, None]))[0].numpy()
        zp = mk.encode(torch.as_tensor(pool(x.numpy())))[0].numpy()          # followed clock, every minute
    mean_ = mu_.mean(0)
    _, S_, Vt_ = np.linalg.svd(mu_ - mean_, full_matrices=False)
    Kk = mk.K.detach().numpy()
    lam_, V_ = np.linalg.eig(Kk); c_ = np.linalg.inv(V_) @ zp[0]
    tt_ = np.arange(0, STEPS + 1, 2); al_ = np.linspace(0, 2 * np.pi, 73)
    modes_ = []
    for j in range(len(lam_)):
        if lam_[j].imag <= 1e-9:
            continue
        term = 2.0 * np.real((lam_[j] ** tt_)[:, None] * c_[j] * V_[:, j][None, :])
        ell = 2.0 * np.real(np.exp(1j * al_)[:, None] * c_[j] * V_[:, j][None, :])
        deg = float(np.degrees(np.angle(lam_[j])))
        modes_.append({"label": f"{deg:.2f}°", "hand": 0 if deg < 3 else 1, "deg": deg, "abs": float(abs(lam_[j])),
                       "pc": r3(term @ Vt_.T), "ell": r3(ell @ Vt_.T), "amp": float(np.abs(term @ Vt_.T).max())})
    modes_.sort(key=lambda e: -e["amp"])
    roll_ = [zp[0]]
    for _ in range(STEPS):
        roll_.append(roll_[-1] @ Kk.T)
    return {"pc": r3((mu_ - mean_) @ Vt_.T), "probe": r3((zp - mean_) @ Vt_.T), "var": r3(S_ ** 2 / (S_ ** 2).sum()),
            "epi": {"origin": r3(-mean_ @ Vt_.T), "modes": modes_, "roll_pc": r3(((np.array(roll_) - mean_) @ Vt_.T)[::2])}}


scratch4 = pca_row("runs/koopman_f4_12k.pt") if os.path.exists("runs/koopman_f4_12k.pt") else None

# ---- minimal Koopman model (d = 4) -----------------------------------------------------
ck_min = torch.load("runs/koopman_min.pt", map_location="cpu")
km = KoopmanVAE(4); km.load_state_dict(ck_min["state"]); km.eval()
K_min = km.K.detach().numpy()
with torch.no_grad():
    mu_min = km.encode(torch.as_tensor(clocks.render(ang[:, 0], ang[:, 1])[:, None]))[0].numpy()
    z_min_probe = km.encode(x)[0].numpy()
    z0m = km.encode(x0)[0].numpy()
rolls["minimal"] = roll(km, K_min, z0m)
err["minimal"] = r3(((rolls["minimal"] - gt) ** 2).mean((0, 2, 3)) * 1000)
film["minimal"] = [png(rolls["minimal"][0, s]) for s in SHOW]
lam_m = np.linalg.eigvals(K_min)
m_mean = mu_min.mean(0)
_, mS, mVt = np.linalg.svd(mu_min - m_mean, full_matrices=False)
mpca = lambda m: (m - m_mean) @ mVt.T
# mode decomposition of the followed clock in the 4-D PCA planes (same as for d = 16)
lam4m, V4m = np.linalg.eig(K_min)
c4 = np.linalg.inv(V4m) @ z_min_probe[0]
tt4 = np.arange(0, STEPS + 1, 2)
epi4 = []
for j in range(4):
    if lam4m[j].imag <= 1e-9:
        continue
    term = 2.0 * np.real((lam4m[j] ** tt4)[:, None] * c4[j] * V4m[:, j][None, :])
    al = np.linspace(0, 2 * np.pi, 73)
    ell = 2.0 * np.real(np.exp(1j * al)[:, None] * c4[j] * V4m[:, j][None, :])
    deg = float(np.degrees(np.angle(lam4m[j])))
    epi4.append({"label": "hour k=1" if deg < 3 else "minute k=1", "hand": 0 if deg < 3 else 1, "deg": deg,
                 "abs": float(abs(lam4m[j])), "pc": r3(term @ mVt.T), "ell": r3(ell @ mVt.T), "amp": float(np.abs(term @ mVt.T).max())})
epi4.sort(key=lambda e: -e["amp"])
r4 = [z_min_probe[0]]
for _ in range(STEPS):
    r4.append(r4[-1] @ K_min.T)
epi4_out = {"origin": r3(-m_mean @ mVt.T), "modes": epi4, "roll_pc": r3(mpca(np.array(r4))[::2])}
_chk = np.array(epi4_out["origin"]) + sum(np.array(e["pc"]) for e in epi4)
print("4-D epicycle check, max |sum - rollout|:", np.abs(_chk - np.array(epi4_out["roll_pc"])).max())

minimal = {"epi": epi4_out, "mu": r3(mu_min), "probe": r3(z_min_probe), "pc": r3(mpca(mu_min)), "pc_probe": r3(mpca(z_min_probe)),
           "pc_var": r3(mS ** 2 / (mS ** 2).sum()), "pc_axes": r3(mVt), "K": np.round(K_min, 4).tolist(),
           "eig": [{"deg": float(np.degrees(np.angle(l))), "abs": float(abs(l))} for l in lam_m if l.imag > 0],
           "loss": {"rec": float(np.load("runs/minimal_snapshots.npz")["rec"][-1]),
                    "pred": float(np.load("runs/minimal_snapshots.npz")["pred"][-1])}}
print("minimal K:\n", np.round(K_min, 4), "\neig deg", np.round(np.degrees(np.angle(lam_m)), 3), "|lam|", np.round(np.abs(lam_m), 4))

# ---- training timeline: latent of 500 held-out clocks at each saved snapshot ------------
assert np.allclose(A.vs["eval_angles"], ang)
TL_N = 300
tl_idx = np.random.default_rng(0).choice(len(ang), TL_N, replace=False)
def pick(n, keep):
    """~keep indices out of n: dense at the start (fast changes), then evenly spaced, always the last."""
    head = list(range(min(n, keep // 4)))
    tail = np.linspace(len(head), n - 1, keep - len(head)).round().astype(int).tolist()
    return sorted(set(head + tail))


v_sel = pick(len(A.vs["it"]), 40)
k_sel = [i + 1 for i in pick(len(A.ks["it"]) - 1, 40)]
frames = [("vae", int(A.vs["it"][i]), A.vs["mu"][i][tl_idx], float(A.vs["loss"][i]), None, None) for i in v_sel] + \
         [("koopman", int(A.ks["it"][i]), A.ks["mu"][i][tl_idx], float(A.ks["loss_rec"][i]), float(A.ks["loss_pred"][i]),
           np.linalg.eigvals(A.ks["K"][i])) for i in k_sel]


def build_timeline(frames, modes_fn):
    out_, prev = [], None
    for stage, it_, mu_, l1, l2, ev_ in frames:
        c = mu_ - mu_.mean(0)
        P = c @ np.linalg.svd(c, full_matrices=False)[2][:4].T
        P = P / np.sqrt((P ** 2).sum(1).mean())                  # unit RMS radius
        if prev is not None:                                      # rotate to stay continuous with the previous frame
            U, _, Wt = np.linalg.svd(P.T @ prev)
            P = P @ (U @ Wt)
        prev = P
        h, m = modes_fn(mu_)
        q = lambda a: np.clip(np.round(a * 100), -32000, 32000).astype(int).ravel().tolist()   # flat ints, x100
        out_.append({"stage": stage, "it": it_, "pc": q(P), "modes": q(np.c_[h.real, h.imag, m.real, m.imag]),
                     "loss": None if np.isnan(l1) else round(l1, 1), "pred": None if (l2 is None or np.isnan(l2)) else round(l2, 1),
                     "eig": None if ev_ is None else np.round(np.c_[ev_.real, ev_.imag], 4).tolist()})
    return out_


tl = build_timeline(frames, A.modes)
timeline = {"idx": tl_idx.tolist(), "frames": tl}
print("timeline frames", len(tl))

# same for the minimal model: distillation (K frozen) then Koopman fine-tune
ms = np.load("runs/minimal_snapshots.npz")
K_fin = ms["snap_K"][-1]
lam4, V4 = np.linalg.eig(K_fin)
W4 = np.linalg.inv(V4)
pos4 = sorted([j for j in range(4) if lam4[j].imag > 0], key=lambda j: np.angle(lam4[j]))   # hour (slow), minute
P4 = ms["snap_mu"][-1] @ W4.T
align4 = []
for j, hcol in zip(pos4, (0, 1)):
    c = np.mean(np.exp(1j * ang[:, hcol]) * np.conj(P4[:, j]) / np.abs(P4[:, j]))
    align4.append((c / abs(c)) / np.median(np.abs(P4[:, j])))
modes4 = lambda mu_: tuple((mu_ @ W4.T)[:, j] * al for j, al in zip(pos4, align4))
m_sel = pick(len(ms["snap_it"]), 80)
frames4 = [("distil" if ms["snap_phase"][i] else "koopman", int(ms["snap_it"][i]), ms["snap_mu"][i][tl_idx],
            float(ms["snap_rec"][i]), float(ms["snap_pred"][i]), np.linalg.eigvals(ms["snap_K"][i])) for i in m_sel]
timeline_min = {"idx": tl_idx.tolist(), "frames": build_timeline(frames4, modes4)}
print("minimal timeline frames", len(m_sel))

# Koopman trained from scratch on small latents: VAE stage + Koopman stage, modes of each final K
def modes_from_final(K_fin, mu_fin):
    """Best single eigenfunction per hand (phase fit to the fundamental angle), aligned and scaled."""
    lam_, V_ = np.linalg.eig(K_fin); W_ = np.linalg.inv(V_)
    P_ = mu_fin @ W_.T
    picks = []
    for hcol in (0, 1):
        cand = [j for j in range(len(lam_)) if lam_[j].imag > 1e-6] or list(range(len(lam_)))
        j = max(cand, key=lambda j: abs(np.mean(np.exp(1j * ang[:, hcol]) * np.conj(P_[:, j]) / (np.abs(P_[:, j]) + 1e-12))))
        c = np.mean(np.exp(1j * ang[:, hcol]) * np.conj(P_[:, j]) / (np.abs(P_[:, j]) + 1e-12))
        picks.append((j, (c / (abs(c) + 1e-12)) / (np.median(np.abs(P_[:, j])) + 1e-12)))
    return lambda mu_: tuple((mu_ @ W_.T)[:, j] * al for j, al in picks)


timeline_extra = {}
for key, vae_snap, k_snap in [("d4_id", "runs/vae_snapshots_d4.npz", "runs/koopman_snapshots_d4_id.npz"),
                              ("d4_rot", "runs/vae_snapshots_d4.npz", "runs/koopman_snapshots_d4_rot.npz"),
                              ("d8_id", "runs/vae_snapshots_d8.npz", "runs/koopman_snapshots_d8_id.npz"),
                              ("d8_rot", "runs/vae_snapshots_d8.npz", "runs/koopman_snapshots_d8_rot.npz")]:
    if not (os.path.exists(vae_snap) and os.path.exists(k_snap)):
        print("skipping timeline", key, "(missing snapshots)")
        continue
    vs_, ks_ = np.load(vae_snap), np.load(k_snap)
    assert np.allclose(vs_["eval_angles"], ang) and np.allclose(ks_["eval_angles"], ang)
    fr = [("vae", int(vs_["it"][i]), vs_["mu"][i][tl_idx], float(vs_["loss"][i]), None, None) for i in pick(len(vs_["it"]), 30)] + \
         [("koopman", int(ks_["it"][i]), ks_["mu"][i][tl_idx], float(ks_["loss_rec"][i]), float(ks_["loss_pred"][i]),
           np.linalg.eigvals(ks_["K"][i])) for i in [j + 1 for j in pick(len(ks_["it"]) - 1, 50)]]
    timeline_extra[key] = {"idx": tl_idx.tolist(), "frames": build_timeline(fr, modes_from_final(ks_["K"][-1], ks_["mu"][-1]))}
    print("timeline", key, len(fr), "frames")

out = {"angles": r3(ang), "vae_pc": r3(vae_pc), "vae_var": r3((S[:8] ** 2) / (S ** 2).sum()),
       "k_pc": r3(kpca(k_mu)), "k_var": r3((kS[:8] ** 2) / (kS ** 2).sum()),
       "emodes": emodes, "epi": epi_out, "minimal": minimal, "vae4": vae4, "scratch4": scratch4, "timeline": timeline, "timeline_min": timeline_min, "timeline_extra": timeline_extra, "orbit_idx": ORBIT_IDX, "k_h": r3(np.c_[mh.real, mh.imag]), "k_m": r3(np.c_[mm.real, mm.imag]), "probe": probe,
       "eig_k": eig_k, "eig_pair": eig(kp.K.detach().numpy()), "eig_A": eig(A.A_base), "assign": assign,
       "err": err, "film": film, "curves": curves, "d": int(A.D),
       "n_images": int(A.TRAJ_SHAPE[0] * A.TRAJ_SHAPE[1]), "traj": list(map(int, A.TRAJ_SHAPE))}
# training timelines live in report/timelines/*.json (report_timelines.py), loaded on demand by the page
for k in ("timeline", "timeline_min", "timeline_extra"):
    out.pop(k, None)
out["timeline_index"] = json.load(open("report/timelines/index.json"))
out["vae_explain"] = json.load(open("report/vae_explain.json"))       # written by vae_explain.py
out["harmonics"] = json.load(open("report/harmonics.json"))         # written by harmonics_data.py
out["fast_figs"] = {k: "data:image/jpeg;base64," + base64.b64encode(open(f"report/{k}.jpg", "rb").read()).decode()
                    for k in ("fast_small", "fast_slowness", "d4_modes") if os.path.exists(f"report/{k}.jpg")}
if os.path.exists("report/slowness.json"):                           # written by analyze_runs.py
    out["slowness"] = {"runs": json.load(open("report/slowness.json")),
                       "img": "data:image/png;base64," + base64.b64encode(open("report/slowness.png", "rb").read()).decode()}
json.dump(out, open("report/data.json", "w"), separators=(",", ":"))
print("report/data.json", os.path.getsize("report/data.json") // 1024, "KB")
print({k: (v[60], v[360]) for k, v in err.items() if isinstance(v, list)}, err["average_clock"])

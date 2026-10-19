"""Training timelines for every experiment: latent of 300 held-out clocks at each saved checkpoint.

Writes report/timelines/<key>.json (one per experiment, loaded on demand by the report) and
report/timelines/index.json (the menu). Two projections per frame:
  pc    : PCA of that checkpoint's latent, rotated to stay continuous with the previous frame
  modes : the hour / minute circles of the model's *final* K (single eigenfunction or a product of two)

    python report_timelines.py
"""
import json, os
import numpy as np

OUT = "report/timelines"
os.makedirs(OUT, exist_ok=True)
data = np.load("data/clocks.npz")
ANG = data["val_angles"][:, ::3].reshape(-1, 2)[:2000]
TL_N = 300
IDX = np.random.default_rng(0).choice(len(ANG), TL_N, replace=False)


def pick(n, keep):
    """~keep indices out of n: dense at the start (fast changes), then evenly spaced, always the last."""
    if n <= keep:
        return list(range(n))
    head = list(range(keep // 4))
    tail = np.linspace(len(head), n - 1, keep - len(head)).round().astype(int).tolist()
    return sorted(set(head + tail))


def fit(p, th):
    return abs(np.mean(np.exp(1j * th) * np.conj(p) / (np.abs(p) + 1e-12)))


def hand_readout(K_fin, mu_fin):
    """For each hand, the best of: one eigenfunction, or a product phi_a * phi_b^(+-1) of two.
    Returns a function mu -> (hour circle, minute circle) and a description."""
    lam, V = np.linalg.eig(K_fin)
    W = np.linalg.inv(V)
    P = mu_fin @ W.T
    pos = [j for j in range(len(lam)) if lam[j].imag > 1e-6] or list(range(len(lam)))
    cands = [((j,), (1,)) for j in pos] + [((a, b), (1, e)) for a in pos for b in pos if a != b for e in (1, -1)]
    val = lambda Pm, c: np.prod([Pm[:, j] if e > 0 else np.conj(Pm[:, j]) for j, e in zip(*c)], axis=0)
    picks, desc = [], []
    for h in (0, 1):
        c = max(cands, key=lambda c: fit(val(P, c), ANG[:, h]))
        v = val(P, c)
        a = np.mean(np.exp(1j * ANG[:, h]) * np.conj(v) / (np.abs(v) + 1e-12))
        picks.append((c, (a / (abs(a) + 1e-12)) / (np.median(np.abs(v)) + 1e-12)))
        desc.append({"modes": [float(np.degrees(np.angle(lam[j]))) for j in c[0]], "exps": list(c[1]),
                     "fit": float(fit(v, ANG[:, h]))})
    return (lambda mu: tuple(val(mu @ W.T, c) * al for c, al in picks)), desc


def build(frames, readout):
    out, prev = [], None
    q = lambda a: np.clip(np.round(a * 100), -32000, 32000).astype(int).ravel().tolist()     # flat ints x100
    for stage, it, mu, l1, l2, ev in frames:
        mu = mu[IDX]
        c = mu - mu.mean(0)
        P = c @ np.linalg.svd(c, full_matrices=False)[2][:4].T
        if P.shape[1] < 4:
            P = np.c_[P, np.zeros((len(P), 4 - P.shape[1]))]
        P = P / np.sqrt((P ** 2).sum(1).mean() + 1e-12)
        if prev is not None:                                      # rotate to stay continuous with the previous frame
            U, _, Wt = np.linalg.svd(P.T @ prev)
            P = P @ (U @ Wt)
        prev = P
        h, m = readout(mu)
        out.append({"stage": stage, "it": int(it), "pc": q(P), "modes": q(np.c_[h.real, h.imag, m.real, m.imag]),
                    "loss": None if l1 is None or np.isnan(l1) else round(float(l1), 1),
                    "pred": None if l2 is None or np.isnan(l2) else round(float(l2), 1),
                    "eig": None if ev is None else np.round(np.c_[ev.real, ev.imag], 4).tolist()})
    return out


def standard(vae_snap, k_snap, n_vae=30, n_k=50):
    """Stage 1 = VAE snapshots, stage 2 = Koopman snapshots."""
    vs, ks = np.load(vae_snap), np.load(k_snap)
    assert np.allclose(vs["eval_angles"], ANG) and np.allclose(ks["eval_angles"], ANG)
    fr = [("vae", vs["it"][i], vs["mu"][i], vs["loss"][i], None, None) for i in pick(len(vs["it"]), n_vae)]
    fr += [("koopman", ks["it"][i], ks["mu"][i], ks["loss_rec"][i], ks["loss_pred"][i], np.linalg.eigvals(ks["K"][i]))
           for i in [j + 1 for j in pick(len(ks["it"]) - 1, n_k)]]
    return fr, ks["K"][-1], ks["mu"][-1]


def distilled():
    ms = np.load("runs/minimal_snapshots.npz")
    fr = [("distil" if ms["snap_phase"][i] else "koopman", ms["snap_it"][i], ms["snap_mu"][i], ms["snap_rec"][i],
           ms["snap_pred"][i], np.linalg.eigvals(ms["snap_K"][i])) for i in pick(len(ms["snap_it"]), 70)]
    return fr, ms["snap_K"][-1], ms["snap_mu"][-1]


FULL, FAST = "Full size (64 px)", "Fast (32 px)"
EXPS = [  # key, label, group, source
    ("d16", "d = 16 · Koopman (reference)", FULL, ("runs/vae_snapshots.npz", "runs/koopman_snapshots.npz")),
    ("d4_distil", "d = 4 · distilled from d = 16", FULL, "distilled"),
    ("d16_onestep", "d = 16 · one-step loss (pairs)", FULL, ("runs/vae_snapshots.npz", "runs/koopman_snapshots_pair.npz")),
    ("d16_slow1000", "d = 16 · slowness 1000", FULL, ("runs/vae_snapshots.npz", "runs/koopman_snapshots_d16_slow.npz")),
    ("d16_sf100", "d = 16 · slowness 100 first, then Koopman", FULL, ("runs/vae_snapshots.npz", "runs/koopman_snapshots_d16_sf.npz")),
    ("d4_slow1000", "d = 4 · slowness 1000", FULL, ("runs/vae_snapshots_d4.npz", "runs/koopman_snapshots_d4_slow.npz")),
    ("d4_sf100", "d = 4 · slowness 100 first, then Koopman", FULL, ("runs/vae_snapshots_d4.npz", "runs/koopman_snapshots_d4_sf.npz")),
    ("f16_H24", "d = 16 · horizon 24", FAST, ("runs/vae_snapshots_fast16.npz", "runs/koopman_snapshots_fast16_w125_H24.npz")),
    ("f16_H36", "d = 16 · horizon 36", FAST, ("runs/vae_snapshots_fast16.npz", "runs/koopman_snapshots_fast16_H36.npz")),
    ("f16_H39", "d = 16 · horizon 39", FAST, ("runs/vae_snapshots_fast16.npz", "runs/koopman_snapshots_fast16_H39.npz")),
    ("f16_H12", "d = 16 · horizon 12", FAST, ("runs/vae_snapshots_fast16.npz", "runs/koopman_snapshots_fast16_w125_H12.npz")),
    ("f4_12k", "d = 4 · 12,000 it. (minimal model)", FAST, ("runs/vae_snapshots_fast4.npz", "runs/koopman_snapshots_f4_12k.npz")),
    ("f4_6k", "d = 4 · 6,000 it.", FAST, ("runs/vae_snapshots_fast4.npz", "runs/koopman_snapshots_f4_6k.npz")),
    ("f4_s0", "d = 4 · seed 0", FAST, ("runs/vae_snapshots_fast4.npz", "runs/koopman_snapshots_fast4_s0.npz")),
    ("f4_s1", "d = 4 · seed 1", FAST, ("runs/vae_snapshots_fast4.npz", "runs/koopman_snapshots_fast4_s1.npz")),
    ("f4_s2", "d = 4 · seed 2", FAST, ("runs/vae_snapshots_fast4.npz", "runs/koopman_snapshots_fast4_s2.npz")),
    ("f8_s0", "d = 8 · seed 0 (3,000 it.)", FAST, ("runs/vae_snapshots_fast8.npz", "runs/koopman_snapshots_fast8_s0.npz")),
    ("f8_s1", "d = 8 · seed 1 (3,000 it.)", FAST, ("runs/vae_snapshots_fast8.npz", "runs/koopman_snapshots_fast8_s1.npz")),
    ("f8_s2", "d = 8 · seed 2 (3,000 it.)", FAST, ("runs/vae_snapshots_fast8.npz", "runs/koopman_snapshots_fast8_s2.npz")),
    ("f8_long", "d = 8 · 6,000 it.", FAST, ("runs/vae_snapshots_fast8.npz", "runs/koopman_snapshots_f8_long.npz")),
    ("f8_w50", "d = 8 · latent weight 50", FAST, ("runs/vae_snapshots_fast8.npz", "runs/koopman_snapshots_f8_w50.npz")),
    ("f8_w250", "d = 8 · latent weight 250", FAST, ("runs/vae_snapshots_fast8.npz", "runs/koopman_snapshots_f8_w250.npz")),
    ("f8_H24", "d = 8 · horizon 24", FAST, ("runs/vae_snapshots_fast8.npz", "runs/koopman_snapshots_f8_H24.npz")),
    ("f8_slow25", "d = 8 · slowness 25", FAST, ("runs/vae_snapshots_fast8.npz", "runs/koopman_snapshots_f8_slow25.npz")),
    ("f8_slow100", "d = 8 · slowness 100", FAST, ("runs/vae_snapshots_fast8.npz", "runs/koopman_snapshots_f8_slow100.npz")),
    ("f8_slow250", "d = 8 · slowness 250", FAST, ("runs/vae_snapshots_fast8.npz", "runs/koopman_snapshots_f8_slow250.npz")),
]

index = []
for key, label, group, src in EXPS:
    if src == "distilled":
        fr, K_fin, mu_fin = distilled()
    else:
        if not all(os.path.exists(p) for p in src):
            print("skip", key, "(missing", [p for p in src if not os.path.exists(p)], ")")
            continue
        fr, K_fin, mu_fin = standard(*src)
    readout, desc = hand_readout(K_fin, mu_fin)
    frames = build(fr, readout)
    json.dump({"idx": IDX.tolist(), "frames": frames, "readout": desc}, open(f"{OUT}/{key}.json", "w"), separators=(",", ":"))
    index.append({"key": key, "label": label, "group": group, "frames": len(frames), "readout": desc})
    print(f"{key:14s} {len(frames):3d} frames  {os.path.getsize(f'{OUT}/{key}.json') // 1024:5d} KB  "
          f"hour fit {desc[0]['fit']:.2f} minute fit {desc[1]['fit']:.2f}")
json.dump({"idx": IDX.tolist(), "items": index}, open(f"{OUT}/index.json", "w"), indent=1)
print("wrote", len(index), "timelines to", OUT)

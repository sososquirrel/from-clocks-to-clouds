"""Training timeline of the minimal model: latent of 300 held-out clocks at each saved checkpoint.

Stage 1 frames come from runs/minimal/vae_snapshots.npz, stage 2 frames from
runs/minimal/koopman_snapshots.npz. Two projections per frame:
  pc    : PCA of that checkpoint's latent, rotated to stay continuous with the previous frame
  modes : the final model's hour and minute readouts, alpha(z) and beta(z)*conj(alpha(z)),
          applied to every checkpoint (a fixed lens borrowed from the end of training)

Writes report/timeline.json.

    python -m analysis.timeline
"""
import json, os
import numpy as np

VAE_SNAP, K_SNAP, OUT = "runs/minimal/vae_snapshots.npz", "runs/minimal/koopman_snapshots.npz", "report/timeline.json"
data = np.load("data/clocks.npz")
ANG = data["val_angles"][:, ::3].reshape(-1, 2)[:2000]
IDX = np.random.default_rng(0).choice(len(ANG), 300, replace=False)


def pick(n, keep):
    """~keep indices out of n: dense at the start (fast changes), then evenly spaced, always the last."""
    if n <= keep:
        return list(range(n))
    head = list(range(keep // 4))
    tail = np.linspace(len(head), n - 1, keep - len(head)).round().astype(int).tolist()
    return sorted(set(head + tail))


def fit(p, th):
    """1 if phase(p) = th + constant for every clock, ~0 if unrelated."""
    return abs(np.mean(np.exp(1j * th) * np.conj(p) / (np.abs(p) + 1e-12)))


def hand_readout(K_fin, mu_fin):
    """For each hand, the best of: one eigen-coordinate of K, or a product of two (e.g. beta * conj(alpha)).
    Returns a function mu -> (hour circle, minute circle) and a description of the choice."""
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
        P = np.c_[P, np.zeros((len(P), 4 - P.shape[1]))] if P.shape[1] < 4 else P
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


if __name__ == "__main__":
    vs, ks = np.load(VAE_SNAP), np.load(K_SNAP)
    assert np.allclose(vs["eval_angles"], ANG) and np.allclose(ks["eval_angles"], ANG)
    frames = [("vae", vs["it"][i], vs["mu"][i], vs["loss"][i], None, None) for i in pick(len(vs["it"]), 30)]
    frames += [("koopman", ks["it"][i], ks["mu"][i], ks["loss_rec"][i], ks["loss_pred"][i], np.linalg.eigvals(ks["K"][i]))
               for i in [j + 1 for j in pick(len(ks["it"]) - 1, 50)]]
    readout, desc = hand_readout(ks["K"][-1], ks["mu"][-1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"idx": IDX.tolist(), "frames": build(frames, readout), "readout": desc}, open(OUT, "w"), separators=(",", ":"))
    print(f"wrote {OUT}: {len(frames)} frames | hour fit {desc[0]['fit']:.2f}, minute fit {desc[1]['fit']:.2f}")

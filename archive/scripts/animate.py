"""Make the explainer animations (1280x720 MP4s) from the trained models + snapshots.

    animations/01_dataset.mp4          the data: clocks, and their true 2-angle state
    animations/02_vae_training.mp4     stage 1: plain VAE, latent space during training
    animations/03_koopman_training.mp4 stage 2: + Koopman loss, latent unfolds into circles
    animations/04_rollout.mp4          predicting the future: VAE+linear fit vs Koopman VAE
    animations/05_modes.mp4            each Koopman mode *is* one hand
    animations/full_explainer.mp4      everything above, with title cards
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import imageio.v2 as imageio
import torch

import clocks
from models import VAE, KoopmanVAE

FPS = 30
W, H_PX = 12.8, 7.2
OUT = "animations"
os.makedirs(OUT, exist_ok=True)
TWO_PI = 2 * np.pi
INK, MUTED, ACCENT, GOOD, BAD = "#1d1d1f", "#6e6e73", "#2f6fdf", "#1a9e5a", "#d1452f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": "#bbbbbb",
                     "axes.titlesize": 13, "axes.titleweight": "bold", "figure.facecolor": "white"})

# ----------------------------------------------------------------------------- data
vs = np.load("runs/vae_snapshots.npz")
ks = np.load("runs/koopman_snapshots.npz")
dt_h, dt_m = float(ks["dt_hour"]), float(ks["dt_min"])
eval_angles = ks["eval_angles"]

dev = torch.device("cpu")
ck = torch.load("runs/koopman.pt", map_location=dev)
kvae = KoopmanVAE(ck["d"]); kvae.load_state_dict(ck["state"]); kvae.eval()
vae = VAE(ck["d"]); vae.load_state_dict(torch.load("runs/vae.pt", map_location=dev)["state"], strict=False); vae.eval()
A_base = ck["A_base"].numpy()
D = ck["d"]
_d = np.load("data/clocks.npz")
TRAJ_SHAPE = (len(_d["train_angles"]) + len(_d["val_angles"]), _d["train_angles"].shape[1])
K = kvae.K.detach().numpy()

# Koopman eigen-decomposition: K = V diag(lam) V^-1, eigenfunctions phi(x) = V^-1 mu(x)
lam, V = np.linalg.eig(K)
Winv = np.linalg.inv(V)


def phi(mu):
    return mu @ Winv.T                                       # complex (N, d)


# Which eigenfunction is which hand? Every rotating mode should behave like e^{i k theta}
# for one hand and some harmonic k (k=1 fundamental, k=2,3,... harmonics). We find the
# (hand, k) whose phase it tracks best.
mu_final = ks["mu"][-1]
P = phi(mu_final)
HAND_DT = {0: dt_h, 1: dt_m}
HAND_NAME = {0: "hour", 1: "minute"}


def score(p, theta):
    """|mean(e^{i theta} conj(p/|p|))|: 1 if phase(p) = theta + const, ~0 if unrelated."""
    return abs(np.mean(np.exp(1j * theta) * np.conj(p) / (np.abs(p) + 1e-12)))


ASSIGN = {}                                                    # mode index -> (hand, k, score)
for j in range(len(lam)):
    if lam[j].imag <= 1e-6 or abs(lam[j]) < 0.97:
        continue
    best = max(((h, k, score(P[:, j], k * eval_angles[:, h])) for h in (0, 1) for k in range(1, 31)),
               key=lambda t: t[2])
    if best[2] > 0.5:
        ASSIGN[j] = best
CONJ = {j: int(np.argmin(np.abs(lam - np.conj(lam[j])))) for j in ASSIGN}
print("Koopman eigenvalues (deg):", np.round(np.degrees(np.angle(lam)), 2))
for j, (h, k, sc) in sorted(ASSIGN.items(), key=lambda t: (t[1][0], t[1][1])):
    print(f"  mode {np.degrees(np.angle(lam[j])):6.2f} deg  |lambda| {abs(lam[j]):.3f}  ->  "
          f"{HAND_NAME[h]} hand, harmonic k={k}  (phase score {sc:.2f})")


def hand_recipe(h):
    """How to build a function ~ e^{i theta_h} from the learned eigenfunctions:
    the fundamental if it was learned, else phi_{k+1} * conj(phi_k) (also an eigenfunction!)."""
    ms = {k: j for j, (hh, k, sc) in sorted(ASSIGN.items(), key=lambda t: t[1][2]) if hh == h}
    if 1 in ms:
        return [(ms[1], 1)]
    pairs = [(k, k + 1) for k in ms if k + 1 in ms]
    assert pairs, f"no fundamental or neighbouring harmonics for the {HAND_NAME[h]} hand: {ASSIGN}"
    k = min(pairs)[0]
    return [(ms[k + 1], 1), (ms[k], -1)]


RECIPE = {h: hand_recipe(h) for h in (0, 1)}
for h in (0, 1):
    desc = " * ".join(f"{'conj ' if e < 0 else ''}phi[{np.degrees(np.angle(lam[j])):.1f} deg]" for j, e in RECIPE[h])
    print(f"{HAND_NAME[h]} hand circle = {desc}")


def _raw_mode(p, h):
    out = np.ones(len(p), complex)
    for j, e in RECIPE[h]:
        out = out * (p[:, j] if e > 0 else np.conj(p[:, j]))
    return out


# cosmetic: rotate/scale each hand's circle so its phase lines up with the true hand angle
ALIGN = {}
for h in (0, 1):
    q = _raw_mode(P, h)
    c = np.mean(np.exp(1j * eval_angles[:, h]) * np.conj(q) / np.abs(q))
    ALIGN[h] = (c / abs(c)) / np.median(np.abs(q))


def modes(mu):
    p = phi(np.atleast_2d(mu))
    return _raw_mode(p, 0) * ALIGN[0], _raw_mode(p, 1) * ALIGN[1]


def turn_hand(z, h, alpha):
    """Turn one hand by alpha *in the latent*: harmonic k of that hand gets phase e^{i k alpha}."""
    p = Winv @ z
    for j, (hh, k, _) in ASSIGN.items():
        if hh == h:
            p[j] *= np.exp(1j * k * alpha); p[CONJ[j]] *= np.exp(-1j * k * alpha)
    return (V @ p).real


def cyc(theta):
    """Cyclic colour for an angle."""
    h = (np.asarray(theta) / TWO_PI) % 1.0
    return hsv_to_rgb(np.stack([h, 0.75 * np.ones_like(h), 0.9 * np.ones_like(h)], -1))


def mat_pow(M, s):
    l, v = np.linalg.eig(M)
    return (v @ np.diag(l.astype(complex) ** s) @ np.linalg.inv(v)).real


@torch.no_grad()
def decode(model, z):
    return torch.sigmoid(model.decode(torch.as_tensor(np.atleast_2d(z), dtype=torch.float32))).squeeze(1).numpy()


def clock_img(ax, img, title=None, color=INK):
    ax.imshow(1 - img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, color=color)


def unit_circle(ax, r=1.0, **kw):
    t = np.linspace(0, TWO_PI, 200)
    ax.plot(r * np.cos(t), r * np.sin(t), **{"color": "#cccccc", "lw": 1, "zorder": 0, **kw})


def fig_new():
    fig = plt.figure(figsize=(W, H_PX), dpi=100)
    return fig


def grab(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


class Video:
    """Writes every frame both to the scene file and to the full explainer."""

    def __init__(self, full):
        self.full, self.w = full, None

    def start(self, name):
        self.w = imageio.get_writer(f"{OUT}/{name}.mp4", fps=FPS, codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_log_level="error")

    def add(self, frame, n=1):
        for _ in range(n):
            self.w.append_data(frame)
            self.full.append_data(frame)

    def stop(self):
        self.w.close()


def title_card(video, title, lines, seconds=4.0):
    fig = fig_new()
    fig.text(0.08, 0.68, title, fontsize=34, fontweight="bold", color=INK)
    for k, l in enumerate(lines):
        fig.text(0.08, 0.55 - 0.075 * k, l, fontsize=19, color=MUTED)
    video.full.append_data(f := grab(fig))
    for _ in range(int(seconds * FPS) - 1):
        video.full.append_data(f)
    plt.close(fig)


# ============================================================== scene 1: dataset
def scene_dataset(video):
    video.start("01_dataset")
    rng = np.random.default_rng(3)
    n = 36
    th0, tm0 = rng.uniform(0, TWO_PI, n), rng.uniform(0, TWO_PI, n)
    fig = fig_new()
    fig.suptitle("The data: clocks whose only degrees of freedom are two hands", fontsize=18, fontweight="bold", y=0.96)
    gax = fig.add_axes([0.03, 0.08, 0.5, 0.8])
    sax = fig.add_axes([0.6, 0.14, 0.36, 0.64])
    sax.set_xlim(0, 360); sax.set_ylim(0, 360)
    sax.set_xlabel("hour-hand angle  θ_h  (deg)"); sax.set_ylabel("minute-hand angle  θ_m  (deg)")
    sax.set_xticks([0, 90, 180, 270, 360]); sax.set_yticks([0, 90, 180, 270, 360])
    sax.set_title("True state space = a torus (both axes wrap)")
    fig.text(0.6, 0.86, "64×64 px = 4096 numbers per image…\n…but only 2 numbers really matter.", fontsize=13, color=MUTED)
    fig.text(0.6, 0.04, "1 step = 1 minute:  minute hand +6°,  hour hand +0.5°",
             fontsize=12, color=MUTED)
    gim = gax.imshow(np.zeros((6 * 64, 6 * 64)), cmap="gray", vmin=0, vmax=1); gax.axis("off")
    nt, T_ = TRAJ_SHAPE
    gax.set_title(f"{nt * T_:,} images ({nt:,} trajectories × {T_} minutes)", fontsize=13)
    trail_len = 25
    hist = []
    sc = sax.scatter([], [], s=40, zorder=3, edgecolor="white", linewidth=0.5)
    trails = sax.scatter([], [], s=6, zorder=2)
    steps = np.arange(0, 150, 0.5)
    for s in steps:
        th, tm = (th0 + s * dt_h) % TWO_PI, (tm0 + s * dt_m) % TWO_PI
        im = clocks.render(th, tm).reshape(6, 6, 64, 64).transpose(0, 2, 1, 3).reshape(6 * 64, 6 * 64)
        gim.set_data(1 - im)
        hist.append(np.stack([np.degrees(th), np.degrees(tm)], -1))
        hist = hist[-trail_len:]
        sc.set_offsets(hist[-1]); sc.set_facecolor(cyc(th0))
        tr = np.concatenate(hist[:-1]) if len(hist) > 1 else np.zeros((0, 2))
        trails.set_offsets(tr); trails.set_facecolor(np.concatenate([cyc(th0)] * (len(hist) - 1)) if len(hist) > 1 else "none")
        trails.set_alpha(0.35)
        video.add(grab(fig))
    plt.close(fig)
    video.stop()


# ============================================================== scene 2: VAE training
def scene_vae(video):
    video.start("02_vae_training")
    mu_all = vs["mu"]
    final = mu_all[-1]
    mean = final.mean(0)
    _, _, Vt = np.linalg.svd(final - mean, full_matrices=False)       # fixed PCA frame
    proj = lambda m: (m - mean) @ Vt.T
    lim = np.abs(proj(final)).max() * 1.1
    ang = vs["eval_angles"]
    fig = fig_new()
    fig.suptitle("Stage 1 — VAE:  loss = reconstruction + KL", fontsize=18, fontweight="bold", y=0.97)
    a1 = fig.add_axes([0.04, 0.38, 0.27, 0.48]); a2 = fig.add_axes([0.35, 0.38, 0.27, 0.48])
    for a, t in [(a1, "latent PCA 1-2 — colour = hour hand"), (a2, "latent PCA 3-4 — colour = minute hand")]:
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal"); a.set_title(t, fontsize=11)
        a.set_xticks([]); a.set_yticks([])
    s1 = a1.scatter(*proj(mu_all[0])[:, :2].T, c=cyc(ang[:, 0]), s=4)
    s2 = a2.scatter(*proj(mu_all[0])[:, 2:4].T, c=cyc(ang[:, 1]), s=4)
    lax = fig.add_axes([0.70, 0.52, 0.27, 0.32])
    it, loss = vs["it"], vs["loss"]
    lax.plot(it[1:], loss[1:], color="#cccccc"); lax.set_yscale("log"); lax.set_title("training loss", fontsize=11)
    lax.set_xlabel("iteration")
    cur, = lax.plot([], [], color=ACCENT, lw=2)
    fig.text(0.70, 0.36, f"The {D}-D latent learns to encode the hands:\nreconstructions become sharp.\n\n"
             "But nothing asks the latent to be simple\nin *time*: it is a tangled Gaussian blob.",
             fontsize=12, color=MUTED, va="top")
    ims = []
    tgt = vs["recon_target"]
    for k in range(6):
        ax = fig.add_axes([0.04 + 0.1 * k, 0.17, 0.09, 0.16]); clock_img(ax, tgt[k])
        ims.append(fig.add_axes([0.04 + 0.1 * k, 0.01, 0.09, 0.16]))
    fig.text(0.005, 0.25, "input", rotation=90, va="center", color=MUTED)
    fig.text(0.005, 0.09, "recon", rotation=90, va="center", color=MUTED)
    rec_artists = []
    for k, ax in enumerate(ims):
        clock_img(ax, vs["recon"][0][k]); rec_artists.append(ax.images[0])
    itxt = fig.text(0.70, 0.88, "", fontsize=12, color=ACCENT)
    for f in range(len(it)):
        P = proj(mu_all[f])
        s1.set_offsets(P[:, :2]); s2.set_offsets(P[:, 2:4])
        for k, art in enumerate(rec_artists):
            art.set_data(1 - vs["recon"][f][k])
        if f > 0:
            cur.set_data(it[1:f + 1], loss[1:f + 1])
        itxt.set_text(f"iteration {it[f]:,}")
        video.add(grab(fig), n=2 if f < 10 else 1)
    video.add(grab(fig), n=FPS * 2)
    plt.close(fig)
    video.stop()


# ============================================================== scene 3: Koopman training
def scene_koopman(video):
    video.start("03_koopman_training")
    mu_all, Ks, it = ks["mu"], ks["K"], ks["it"]
    mh, mm = modes(mu_all[-1])
    lim = 1.6
    fig = fig_new()
    fig.suptitle("Stage 2 — add the Koopman loss:  z(t+1) ≈ K z(t),  x(t+k) ≈ decode(Kᵏ z(t))",
                 fontsize=17, fontweight="bold", y=0.97)
    a1 = fig.add_axes([0.03, 0.40, 0.25, 0.46]); a2 = fig.add_axes([0.30, 0.40, 0.25, 0.46])
    ea = fig.add_axes([0.575, 0.36, 0.27, 0.52])
    for a, t in [(a1, "hour-hand mode  (colour = hour hand)"), (a2, "minute-hand mode  (colour = minute)")]:
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal"); a.set_title(t, fontsize=11)
        a.set_xticks([]); a.set_yticks([]); unit_circle(a)
    m0h, m0m = modes(mu_all[0])
    s1 = a1.scatter(m0h.real, m0h.imag, c=cyc(eval_angles[:, 0]), s=4)
    s2 = a2.scatter(m0m.real, m0m.imag, c=cyc(eval_angles[:, 1]), s=4)
    # eigenvalues
    ea.set_xlim(0.5, 1.1); ea.set_ylim(-0.55, 0.55); ea.set_xticks([]); ea.set_yticks([]); ea.set_aspect("equal"); ea.set_title("eigenvalues of K", fontsize=11)
    ea.axhline(0, color="#eeeeee", lw=1); ea.axvline(0, color="#eeeeee", lw=1); unit_circle(ea)
    for kk in range(1, 6):                                        # minute hand: e^{±i k 6°}
        a_ = kk * dt_m
        ea.scatter([np.cos(a_)] * 2, [np.sin(a_), -np.sin(a_)], marker="o", s=130, facecolor="none",
                   edgecolor=GOOD, lw=1.2 if kk > 1 else 2, alpha=1 if kk == 1 else 0.5, zorder=2)
        ea.text(np.cos(a_) - 0.07, np.sin(a_) + 0.035, f"{6 * kk}°", color=GOOD, fontsize=8)
    ea.text(0.52, -0.52, "green rings: minute hand 6° and harmonics k·6°\nhour hand 0.5° (+ harmonics) sit near 1",
            color=GOOD, fontsize=8, va="bottom")
    ev_sc = ea.scatter([], [], c=ACCENT, s=30, zorder=3)
    ev_hist = np.array([np.linalg.eigvals(k) for k in Ks])
    trail = ea.scatter([], [], c=ACCENT, s=2, alpha=0.3)
    # losses
    lax = fig.add_axes([0.875, 0.40, 0.11, 0.46])
    lat, pred = ks["loss_lat"], ks["loss_pred"]
    lax.plot(it[1:], pred[1:], color="#dddddd"); lax.set_yscale("log"); lax.set_title("prediction loss", fontsize=11)
    lax.set_xticks([]); cur, = lax.plot([], [], color=ACCENT, lw=2)
    # rollout strip
    stride = int(ks["rollout_stride"])
    idx_show = np.linspace(0, ks["rollout"].shape[2] - 1, 10).astype(int)
    steps_show = idx_show * stride
    pa = ks["probe_angles"][0]
    truth = clocks.render((pa[0] + steps_show * dt_h) % TWO_PI, (pa[1] + steps_show * dt_m) % TWO_PI)
    roll_arts = []
    n = len(steps_show)
    for k, s in enumerate(steps_show):
        ax = fig.add_axes([0.03 + k * 0.097, 0.17, 0.09, 0.16]); clock_img(ax, truth[k])
        ax.set_title(f"+{s // 60}h{s % 60:02d}", fontsize=10, fontweight="normal", color=MUTED)
        ax = fig.add_axes([0.03 + k * 0.097, 0.0, 0.09, 0.16]); clock_img(ax, ks["rollout"][0][0, idx_show[k]] / 255)
        roll_arts.append(ax.images[0])
    fig.text(0.008, 0.255, "truth", rotation=90, va="center", color=MUTED, fontsize=10)
    fig.text(0.008, 0.085, "Kᵏ z₀", rotation=90, va="center", color=ACCENT, fontsize=10)
    itxt = fig.text(0.03, 0.90, "", fontsize=12, color=ACCENT)
    for f in range(len(it)):
        mh, mm = modes(mu_all[f])
        s1.set_offsets(np.c_[mh.real, mh.imag]); s2.set_offsets(np.c_[mm.real, mm.imag])
        ev = ev_hist[f]
        ev_sc.set_offsets(np.c_[ev.real, ev.imag])
        tr = ev_hist[max(0, f - 40):f + 1].ravel()
        trail.set_offsets(np.c_[tr.real, tr.imag])
        if f > 0:
            cur.set_data(it[1:f + 1], pred[1:f + 1])
        for k, s in enumerate(steps_show):
            roll_arts[k].set_data(1 - ks["rollout"][f][0, idx_show[k]] / 255)
        itxt.set_text(f"iteration {it[f]:,}   (rollout horizon H = {ks['H'][f]})")
        video.add(grab(fig), n=3 if f < 12 else 1)
    video.add(grab(fig), n=FPS * 3)
    plt.close(fig)
    video.stop()


# ============================================================== scene 4: rollout
def scene_rollout(video, n_steps=360, sub=1):
    video.start("04_rollout")
    pa = ks["probe_angles"][0]
    x0 = torch.as_tensor(clocks.render([pa[0]], [pa[1]])[:, None])
    with torch.no_grad():
        z_k = kvae.encode(x0)[0].numpy()[0]
        z_b = vae.encode(x0)[0].numpy()[0]
    mh_bg, mm_bg = modes(ks["mu"][-1])
    fig = fig_new()
    fig.suptitle("Predicting the future from a single image", fontsize=18, fontweight="bold", y=0.97)
    axT = fig.add_axes([0.02, 0.36, 0.22, 0.45]); axB = fig.add_axes([0.26, 0.36, 0.22, 0.45]); axK = fig.add_axes([0.50, 0.36, 0.22, 0.45])
    img0 = clocks.render([pa[0]], [pa[1]])[0]
    clock_img(axT, img0, "ground truth"); clock_img(axB, img0, "VAE + best linear map A", BAD)
    clock_img(axK, img0, "Koopman VAE:  K^t z₀", ACCENT)
    fig.text(0.26, 0.31, "A = least-squares fit on the\nstage-1 latent. |eigenvalues| < 1,\nso predictions fade to a\nblurry average clock.",
             fontsize=11, color=MUTED, va="top")
    fig.text(0.50, 0.31, "K: |eigenvalues| = 1.\nRepeated multiplication is a\nrotation forever: no drift,\nno blur.",
             fontsize=11, color=MUTED, va="top")
    m1 = fig.add_axes([0.76, 0.52, 0.2, 0.36]); m2 = fig.add_axes([0.76, 0.10, 0.2, 0.36])
    for a, bg, col, t in [(m1, mh_bg, 0, "mode 1 (hour)"), (m2, mm_bg, 1, "mode 2 (minute)")]:
        a.scatter(bg.real, bg.imag, c=cyc(eval_angles[:, col]), s=2, alpha=0.25)
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(t, fontsize=11)
    p1, = m1.plot([], [], "o", color=INK, ms=9); p2, = m2.plot([], [], "o", color=INK, ms=9)
    ttxt = fig.text(0.02, 0.88, "", fontsize=14, color=INK)
    mse_txt = fig.text(0.02, 0.06, "", fontsize=12, color=MUTED)
    for f in range(n_steps * sub + 1):
        s = f / sub
        truth = clocks.render([(pa[0] + s * dt_h) % TWO_PI], [(pa[1] + s * dt_m) % TWO_PI])[0]
        zk = mat_pow(K, s) @ z_k
        zb = mat_pow(A_base, s) @ z_b
        ik, ib = decode(kvae, zk)[0], decode(vae, zb)[0]
        axT.images[0].set_data(1 - truth); axB.images[0].set_data(1 - ib); axK.images[0].set_data(1 - ik)
        h, m = modes(zk[None])
        p1.set_data([h.real[0]], [h.imag[0]]); p2.set_data([m.real[0]], [m.imag[0]])
        ttxt.set_text(f"{int(s) // 60}h{int(s) % 60:02d} ahead  =  K applied {int(s)} times     ({clocks.angle_to_time((pa[0] + s * dt_h) % TWO_PI, (pa[1] + s * dt_m) % TWO_PI)})")
        mse_txt.set_text(f"pixel error   VAE+A: {np.mean((ib - truth) ** 2):.4f}     Koopman: {np.mean((ik - truth) ** 2):.4f}")
        video.add(grab(fig), n=FPS if f == 0 else 1)
    video.add(grab(fig), n=FPS * 2)
    plt.close(fig)
    video.stop()


# ============================================================== scene 5: modes are hands
def scene_modes(video):
    video.start("05_modes")
    pa = ks["probe_angles"][1]
    x0 = torch.as_tensor(clocks.render([pa[0]], [pa[1]])[:, None])
    with torch.no_grad():
        z0 = kvae.encode(x0)[0].numpy()[0]
    mh_bg, mm_bg = modes(ks["mu"][-1])
    fig = fig_new()
    fig.suptitle("The Koopman modes are the hands", fontsize=18, fontweight="bold", y=0.97)
    fig.text(0.5, 0.875, "Rotate the phase of one hand's Koopman modes (leave the other hand's fixed) and decode:",
             ha="center", fontsize=13, color=MUTED)
    rows = []
    for r, (mode, bg, col, name) in enumerate([(0, mh_bg, 0, "the hour modes"), (1, mm_bg, 1, "the minute modes")]):
        y = 0.45 - r * 0.44
        a = fig.add_axes([0.18, y, 0.22, 0.38]); a.scatter(bg.real, bg.imag, c=cyc(eval_angles[:, col]), s=2, alpha=0.25)
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"rotate {name}", fontsize=12)
        pt, = a.plot([], [], "o", color=INK, ms=10)
        c = fig.add_axes([0.48, y, 0.22, 0.38]); clock_img(c, decode(kvae, z0)[0])
        lbl = fig.text(0.74, y + 0.19, "", fontsize=16, va="center")
        rows.append((mode, pt, c, lbl))
    n = 150
    for f in range(2 * n):
        r_active = 0 if f < n else 1
        alpha = TWO_PI * (f % n) / n
        for r, (mode, pt, c, lbl) in enumerate(rows):
            al = alpha if r == r_active else (TWO_PI if r < r_active else 0.0)
            z = turn_hand(z0, mode, al)
            c.images[0].set_data(1 - decode(kvae, z)[0])
            h, m = modes(z[None])
            v = h if r == 0 else m
            pt.set_data([v.real[0]], [v.imag[0]])
            lbl.set_text(("→ only the HOUR hand moves" if r == 0 else "→ only the MINUTE hand moves") if r == r_active else "")
            lbl.set_color(ACCENT)
        video.add(grab(fig))
    plt.close(fig)
    video.stop()


if __name__ == "__main__":
    full = imageio.get_writer(f"{OUT}/full_explainer.mp4", fps=FPS, codec="libx264", quality=8,
                              macro_block_size=1, ffmpeg_log_level="error")
    v = Video(full)
    title_card(v, "VAE + Koopman, on clocks", [
        "1.  The data: images of clocks",
        "2.  A VAE compresses each image into a small latent vector",
        "3.  A Koopman loss makes time evolution LINEAR in that latent",
        "4.  Result: predict the future with one matrix, forever"])
    scene_dataset(v)
    title_card(v, "Stage 1 — a plain VAE", [
        f"encoder: image → z ({D} numbers),   decoder: z → image",
        "loss = reconstruction + KL",
        "No notion of time: every image is seen independently."])
    scene_vae(v)
    title_card(v, "Stage 2 — add a Koopman operator K", [
        f"K is a learnable {D}×{D} matrix acting on the latent.",
        "On short trajectories we ask:  z(t+k) ≈ Kᵏ z(t)   and   x(t+k) ≈ decode(Kᵏ z(t))",
        "The encoder must find coordinates where 'one tick' is a matrix multiply.",
        "For clocks the answer is known: a rotation per hand, eigenvalues e^{±iω}.",
        "One step = 1 minute; a 5-minute tick is simply K⁵."])
    scene_koopman(v)
    title_card(v, "Why it matters: long-term prediction", [
        "Encode one image, multiply by K again and again, decode.",
        "Compared with the best linear map fitted on the plain VAE latent."])
    scene_rollout(v)
    title_card(v, "Interpretation", [
        "Koopman eigenfunctions are the physically meaningful coordinates:",
        "here, each complex mode is literally one hand of the clock."])
    scene_modes(v)
    full.close()
    print("wrote", sorted(os.listdir(OUT)))

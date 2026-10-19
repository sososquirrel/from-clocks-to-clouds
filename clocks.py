"""Procedural clock-image dataset.

Each image is a 64x64 grayscale clock face with an hour hand (short, thick) and a
minute hand (long, thin). The *state* of a clock is just two angles:

    theta_h  (hour hand)    and    theta_m  (minute hand)

measured clockwise from 12 o'clock. We sample the two hands independently, so the
set of all clocks is a 2-D torus living inside a 4096-D pixel space -- exactly the
kind of low-dimensional structure a VAE should discover.

Dynamics: one time step = 1 minute -> the minute hand turns 6 deg, the hour hand
turns 0.5 deg (--minutes_per_step changes this; a 5-minute tick is just K^5). In angle space this is a pure rotation, so in the coordinates
(cos theta, sin theta) of each hand the dynamics are *linear*. That is the Koopman
picture we want the network to find.

Run:  python clocks.py           -> writes data/clocks.npz + previews
"""
import os
import numpy as np

IMG = 64
DT_MIN = np.deg2rad(6.0)         # minute hand rotation per step (1 minute)
DT_HOUR = DT_MIN / 12.0          # hour hand rotation per step

_lin = np.linspace(-1, 1, IMG, dtype=np.float32)
_X, _Y = np.meshgrid(_lin, -_lin)  # y up


def _soft(dist, half_width, aa=0.035):
    """Anti-aliased stroke: 1 inside, 0 outside, smooth over ~aa."""
    return np.clip((half_width - dist) / aa + 0.5, 0.0, 1.0)


def _seg_dist(px, py, ax, ay, bx, by):
    """Distance from pixel grid to segments A->B (broadcast over leading dims)."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    t = np.clip((wx * vx + wy * vy) / (vx * vx + vy * vy + 1e-8), 0, 1)
    dx, dy = wx - t * vx, wy - t * vy
    return np.sqrt(dx * dx + dy * dy)


def _face():
    r = np.sqrt(_X ** 2 + _Y ** 2)
    img = _soft(np.abs(r - 0.92), 0.035)                      # rim
    for k in range(12):                                       # ticks
        a = 2 * np.pi * k / 12
        s, c = np.sin(a), np.cos(a)
        r0 = 0.70 if k % 3 == 0 else 0.76
        d = _seg_dist(_X, _Y, r0 * s, r0 * c, 0.84 * s, 0.84 * c)
        img = np.maximum(img, _soft(d, 0.035 if k % 3 == 0 else 0.022))
    img = np.maximum(img, _soft(r, 0.06))                     # hub
    return img.astype(np.float32)


FACE = _face()


def render(theta_h, theta_m):
    """Render clocks. theta_h, theta_m: arrays of shape (N,). Returns (N,64,64) float32 in [0,1]."""
    th = np.asarray(theta_h, dtype=np.float32).reshape(-1, 1, 1)
    tm = np.asarray(theta_m, dtype=np.float32).reshape(-1, 1, 1)
    X, Y = _X[None], _Y[None]
    dh = _seg_dist(X, Y, -0.08 * np.sin(th), -0.08 * np.cos(th), 0.48 * np.sin(th), 0.48 * np.cos(th))
    dm = _seg_dist(X, Y, -0.10 * np.sin(tm), -0.10 * np.cos(tm), 0.80 * np.sin(tm), 0.80 * np.cos(tm))
    hands = np.maximum(_soft(dh, 0.075), _soft(dm, 0.035))
    return np.maximum(FACE[None], hands).astype(np.float32)


def make_trajectories(n_traj, T, rng, dt_min=DT_MIN, dt_hour=DT_HOUR):
    """Random initial hands, then tick forward T-1 steps. Returns angles (n_traj, T, 2)."""
    th0 = rng.uniform(0, 2 * np.pi, n_traj)
    tm0 = rng.uniform(0, 2 * np.pi, n_traj)
    k = np.arange(T)
    th = (th0[:, None] + dt_hour * k[None]) % (2 * np.pi)
    tm = (tm0[:, None] + dt_min * k[None]) % (2 * np.pi)
    return np.stack([th, tm], -1).astype(np.float32)


def render_batched(angles, chunk=4096):
    flat = angles.reshape(-1, 2)
    out = np.empty((len(flat), IMG, IMG), np.uint8)
    for i in range(0, len(flat), chunk):
        a = flat[i:i + chunk]
        out[i:i + chunk] = (render(a[:, 0], a[:, 1]) * 255).round().astype(np.uint8)
    return out.reshape(*angles.shape[:-1], IMG, IMG)


def angle_to_time(th, tm):
    """Human-readable label (hour from hour hand, minutes from minute hand)."""
    h = int(th / (2 * np.pi) * 12) % 12 or 12
    m = int(round(tm / (2 * np.pi) * 60)) % 60
    return f"{h}:{m:02d}"


if __name__ == "__main__":
    import argparse
    from PIL import Image

    p = argparse.ArgumentParser()
    p.add_argument("--n_traj", type=int, default=3000)
    p.add_argument("--T", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--minutes_per_step", type=float, default=1.0)
    p.add_argument("--out", default="data/clocks.npz")
    args = p.parse_args()
    dt_min = np.deg2rad(6.0 * args.minutes_per_step)
    dt_hour = dt_min / 12.0

    rng = np.random.default_rng(args.seed)
    os.makedirs("data/samples", exist_ok=True)

    angles = make_trajectories(args.n_traj, args.T, rng, dt_min, dt_hour)
    frames = render_batched(angles)
    n_val = args.n_traj // 10
    np.savez_compressed(args.out,
                        train_frames=frames[n_val:], train_angles=angles[n_val:],
                        val_frames=frames[:n_val], val_angles=angles[:n_val],
                        dt_hour=dt_hour, dt_min=dt_min)
    print(f"saved {args.out}: {frames.shape[0]} trajectories x {frames.shape[1]} steps "
          f"= {frames.shape[0] * frames.shape[1]:,} images ({n_val} trajectories held out)")

    # previews: a 12x12 grid of random clocks, and a few trajectories as filmstrips
    grid = frames[:144, 0].reshape(12, 12, IMG, IMG).transpose(0, 2, 1, 3).reshape(12 * IMG, 12 * IMG)
    Image.fromarray(255 - grid).save("data/preview_grid.png")
    strip = frames[:8, :12].transpose(0, 2, 1, 3).reshape(8 * IMG, 12 * IMG)
    Image.fromarray(255 - strip).save("data/preview_trajectories.png")
    for i in range(200):
        Image.fromarray(255 - frames[i, 0]).save(f"data/samples/clock_{i:03d}.png")
    print("wrote data/preview_grid.png, data/preview_trajectories.png, data/samples/*.png")

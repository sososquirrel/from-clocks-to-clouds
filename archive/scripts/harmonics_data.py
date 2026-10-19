"""Fourier / Koopman harmonics of the clock images, for the report's "Harmonics, explained" section.

On a grid over the torus (theta_h, theta_m), the image is a Fourier series
    x(s) = sum_{p,q} X_pq exp(i (p theta_h + q theta_m)),
and each exp(i(p theta_h + q theta_m)) is a Koopman eigenfunction. Writes report/harmonics.json:
  - share of image variation per harmonic, per hand
  - mode images Re X_(k,0) and Re X_(0,k)
  - a clock rebuilt (linear readout) from the strongest n harmonic pairs, for several n
  - pixel error vs latent size for a linear readout
"""
import base64, io, json
import numpy as np
from PIL import Image
import clocks

N = 120                                                   # 3-degree grid
g = np.linspace(0, 2 * np.pi, N, endpoint=False).astype(np.float32)
TH, TM = np.meshgrid(g, g, indexing="ij")
X = np.empty((N, N, 4096), np.float32)
for i in range(N):
    X[i] = clocks.render(TH[i], TM[i]).reshape(N, -1)
F = np.fft.fft2(X, axes=(0, 1)) / (N * N)                 # F[p, q, pixel]
del X
E = (np.abs(F) ** 2).sum(-1)
freq = np.fft.fftfreq(N, 1 / N).astype(int)
P, Q = np.meshgrid(freq, freq, indexing="ij")
var = E.sum() - E[0, 0]


def png_gray(img, size=96):
    b = io.BytesIO()
    Image.fromarray((255 - np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def png_div(img, size=96):
    """Diverging map for a signed mode image: blue negative, white zero, red positive."""
    m = np.abs(img).max() + 1e-12
    v = img / m
    rgb = np.ones(img.shape + (3,))
    rgb[..., 0] -= np.clip(-v, 0, 1) * 0.85
    rgb[..., 1] -= np.abs(v) * 0.75
    rgb[..., 2] -= np.clip(v, 0, 1) * 0.85
    b = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


# 1. share of image variation per harmonic, per hand (both signs of k)
KMAX = 16
share = {"hour": [], "minute": []}
for k in range(1, KMAX + 1):
    share["hour"].append(float((E[(P == k) & (Q == 0)].sum() + E[(P == -k) & (Q == 0)].sum()) / var))
    share["minute"].append(float((E[(P == 0) & (Q == k)].sum() + E[(P == 0) & (Q == -k)].sum()) / var))
mixed = float(1 - (E[(P == 0) ^ (Q == 0)].sum()) / var)

# 2. mode images (real part of the Fourier coefficient image; phase at 12 o'clock)
modes = {"hour": [png_div(F[k, 0].real.reshape(64, 64)) for k in range(1, 5)],
         "minute": [png_div(F[0, k].real.reshape(64, 64)) for k in range(1, 5)]}

# 3. linear readout: keep the n strongest conjugate pairs
mask_rep = (P > 0) | ((P == 0) & (Q > 0))
idx = np.argwhere(mask_rep)
e_pair = 2 * E[mask_rep]
order = np.argsort(-e_pair)
cum = np.cumsum(e_pair[order])
mse_after = (var - cum) / 4096
curve_n = sorted(set(np.unique(np.round(np.logspace(0, np.log10(600), 60)).astype(int)).tolist()))
curve = [{"d": 2 * n + 1, "mse": float(mse_after[n - 1] * 1000)} for n in curve_n if n <= len(mse_after)]
need = {}
for thr in (5e-3, 1e-3, 0.14e-3):
    n = int(np.searchsorted(-mse_after, -thr)) + 1
    need[str(thr)] = {"modes": n, "d": 2 * n + 1}
# example clock (10:10-ish on the grid) rebuilt from n pairs
ih, im = int(round(np.deg2rad(305) / (2 * np.pi) * N)) % N, int(round(np.deg2rad(60) / (2 * np.pi) * N)) % N
truth = clocks.render(g[ih:ih + 1], g[im:im + 1])[0]
rebuild = []
for n in (1, 4, 10, 21, 58, 229):
    x = F[0, 0].real.copy()
    for (p_i, q_i) in idx[order[:n]]:
        ph = np.exp(2j * np.pi * (p_i * ih + q_i * im) / N)
        x = x + 2 * (F[p_i, q_i] * ph).real
    rebuild.append({"modes": n, "d": 2 * n + 1, "img": png_gray(x.reshape(64, 64)),
                    "mse": float(((x.reshape(64, 64) - truth) ** 2).mean() * 1000)})
top = [{"p": int(freq[p_i]), "q": int(freq[q_i]), "share": float(e_pair[o] / var)} for o, (p_i, q_i) in zip(order[:24], idx[order[:24]])]

out = {"share": share, "mixed": mixed, "modes": modes, "curve": curve, "need": need, "rebuild": rebuild,
       "truth": png_gray(truth), "top": top, "avg_mse": float(var / 4096 * 1000)}
json.dump(out, open("report/harmonics.json", "w"), separators=(",", ":"))
print("hour  k1-6:", [round(100 * v, 1) for v in share["hour"][:6]])
print("minute k1-6:", [round(100 * v, 1) for v in share["minute"][:6]])
print("mixed (both hands) share:", round(100 * mixed, 1), "%")
print("need:", need)
print("rebuild:", [(r["d"], round(r["mse"], 2)) for r in rebuild])
print("top pairs:", [(t["p"], t["q"], round(100 * t["share"], 1)) for t in top[:10]])

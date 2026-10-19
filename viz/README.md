# Visualisation data for slides

Everything the minimal-model report draws, as plain JSON, one file per visualisation. Written by
`python -m report.build` from the trained model in `runs/minimal/`; no retraining needed.
Load with `const d = await (await fetch("viz/spectrum.json")).json();` (serve the folder over
HTTP, e.g. `python -m http.server`, since browsers block `fetch` on `file://`).

## Conventions

- **Hand angles** (θ_h, θ_m) are in **radians, clockwise from 12 o'clock**. Time steps are minutes:
  the minute hand turns 6° per step, the hour hand 0.5°.
- **Complex coordinates** (α, β, modes) are stored as `[re, im]`; plot x = re, y = im. Their angle,
  measured from the x axis counter-clockwise, is the hand angle the model reads.
- **Images** are PNG data URLs (`"data:image/png;base64,..."`), black clock on white; use them
  directly as `<img src>`.
- **Errors** are pixel MSE × 10⁻³ (32×32 images). The reference "average clock" error is `blank`.
- Unless stated otherwise, point arrays have one entry per held-out clock, in the same order as
  `common.angles` (2,000 clocks).

## Files

**`common.json`**
- `angles[i] = [θ_h, θ_m, θ_h + θ_m]` for the 2,000 held-out clocks (use for colouring points).
- `dt_deg = {h: 0.5, m: 6}`: degrees per step.

**`vae.json`**: stage 1, the VAE alone
- `picker[h*12 + m]`: hour index h = 0..11 (0 is 12 o'clock), minutes = 5·m. Each `{x: original, r: reconstruction, mse}`, 160 px.
- `gallery[16]`: `{x, r, t}`, original, reconstruction and time label.
- `err_median`, `err_p95`, `blank`: reconstruction error statistics.
- `kl[4]`, `sigma[4]`: KL (nats) and mean encoder σ per latent dimension.
- `trav[4][9]`: decode with one latent dimension set to `trav_steps[9]` (−2.5…2.5), others fixed.

**`latent.json`**: PCA of the latent, VAE vs Koopman, and one clock followed for 6 hours
- `vae`, `koopman`: `{pc[2000][4], probe[361][4], var[4]}`. `pc` holds PC1–PC4 coordinates of every clock, `probe` the followed clock at minutes 0..360, `var` the share of variance per PC.
- `koopman.epi`: mode decomposition of the followed clock, Kᵗz₀ = Σ 2·Re(λᵗ c v), in PCA coordinates.
  `origin[4]`; `modes[2]` with `{deg, abs, hand (0 hour mode, 1 mixed mode), pc[181][4], ell[73][4], amp}`, where `pc[i]` is the mode's term at minute 2i and `ell` the full ellipse of that term; `roll_pc[181][4]` is Kᵗz₀ itself (minutes 0, 2, …, 360). Chain the terms tip to tail from `origin` to draw the epicycles.
- `probe_theta_h[361]`, `probe_theta_m[361]`: true angles of the followed clock.

**`prediction.json`**: encode one image, apply K t times, decode (200 held-out clocks)
- `error.koop[361]`, `error.vae[361]`: error at minutes 0..360 for the Koopman model and for the best linear map fitted on the plain VAE latent; `error.blank`.
- `film`: `steps[6]` (minutes) and `truth`, `koop`, `vae` image rows for one clock.

**`spectrum.json`**
- `eigenvalues[4]`: `{re, im, deg (angle per step), abs, p, q, fit}`. `(p, q)` means the mode turns with p·θ_h + q·θ_m; `fit` (1 = perfect) is how well its phase tracks that angle.
- `eigenvalues_vae_linear_fit[4]`: `{re, im}` for the linear map on the plain VAE latent (inside the unit circle).
- `recipes.alpha`, `recipes.beta`: `{x[4], y[4], deg}`. α(z) = Σ (x_j + i·y_j) z_j, so x = Re and y = Im. arg α = θ_h, arg β = θ_h + θ_m.
- `example`: one clock (`t`, `z[4]`, `th`, `tm` in degrees) with its α and β (`x, y, angle` in degrees).

**`modes.json`**
- `modes[4]` with keys `alpha`, `beta`, `hour` (= α) and `minute` (= β·conj α). Each has:
  - `deg`: angle per step; `loop`: steps per turn; `fit`;
  - `hand`: column of `common.angles` to colour by;
  - `pts[2000][2]`: every clock projected on the mode, scaled to median radius 1;
  - `orbits[5][181][2]`: 5 clocks pushed forward by K only, at minutes 0, 2, …, 360;
  - `turn[9]`: decode after turning the mode by 0°, 45°, …, 360°;
  - `name`, `note`: labels.
- `orbit_starts[5]`: start time of each orbit clock.

**`timeline.json`**: the latent during training (300 clocks: `idx` indexes `common.angles`)
- `frames[80]`: `{stage: "vae" | "koopman", it, pc, modes, loss, pred, eig}`.
  - `pc` and `modes` are flat integer arrays ×100 of length 300·4. `pc` is (PC1, PC2, PC3, PC4) per clock; `modes` is (Re α, Im α, Re m, Im m) with m = β·conj α, using the *final* model's α and β.
  - `eig` holds the 4 eigenvalues of K as `[re, im]` (null during stage 1).
- `readout`: which eigen-coordinates make the hour and minute readouts, and their fit.

**`kl.json`**: the same pipeline with KL weight β ∈ {0, 0.25, 0.5, 1, 2, 4}
- `orig_imgs[8]`: originals; `pca_ang[600]`: `[θ_h, θ_m]` for the PCA points.
- `runs[i]`: `{beta, vae, seeds, koop}`.
  - `vae` (after stage 1) and `seeds[0]` (after Koopman) both have `rec_imgs[8]`, `gen_imgs[12]` (decoded z ~ N(0, I)), `pca[600][4]`, `rec`, `kl`, `sigma`, `scale` and `noise` (error after latent noise of 5, 10, 20% of the latent scale).
  - `koop` holds the seed-averaged metrics: `err_1h`, `err_6h`, `err_6h_seeds`, `lin_K`, `modes`, `roll[5]` (start, pred 1 h, pred 6 h, truth 1 h, truth 6 h).

**`media/`**: `modes.mp4` (turning α, β, both, and time, decoded) and `modes.jpg` (a still of the same).

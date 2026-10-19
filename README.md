# From clocks to clouds

**A minimal Koopman VAE, explained on clock images** — a testbed before applying the same ideas to
atmospheric fields.

Images of clocks are compressed into **4 numbers**, and a single **4×4 matrix K** moves those
numbers forward in time. Stage 1 is a plain VAE; stage 2 adds a Koopman loss that makes the
latent evolve linearly. The trained K is two rotations, one per clock hand (in a mixed basis, see
below), and predicts the clock hours ahead from a single image.

Report: `report/index.html`, published at https://claude.ai/artifact/TU88ws4tAXifM4EGhiKYbD

## Run

The trained model (`runs/minimal/`) and all visualisation data (`viz/`, `report/`) are in the
repository, so nothing needs to be retrained. To rebuild the data and report from the model:

```bash
pip install -r requirements.txt        # + torch for your platform (see requirements.txt)
python clocks.py                       # regenerate data/clocks.npz (~5 s, not stored in git)
python -m analysis.timeline && python -m analysis.mode_video && python -m report.build
```

To retrain everything from scratch: `./run.sh` (~6 min on Apple silicon), `./run.sh --kl` adds the
KL study (+25 min). Set `PY=...` to choose the Python interpreter (default `python`).

## Visualisations for slides

`viz/` holds every visualisation as plain JSON (one file per figure) plus the mode video; the format
is documented in `viz/README.md`. The report page `report/index.html` is self-contained.

## The recipe

| | |
|---|---|
| data | 64×64 clock images, hands independent, 1 step = 1 minute (minute +6°, hour +0.5°), runs of 40 minutes; trained at 32×32 |
| stage 1 | VAE, latent d = 4, half-width conv net, loss = reconstruction + KL, 15 epochs |
| stage 2 | same network + K (4×4, starts at the identity), 12,000 iterations, clips of up to 36 minutes |
| stage-2 loss | `BCE(decode(z), x) + KL + 125·‖Kᵏμ₀ − μₖ‖²/Var(μ) + BCE(decode(Kᵏμ₀), xₖ)` |

What matters, from the experiments in `archive/`: a time step small compared with the fastest
rotation (30°/step fails, 6°/step works), a horizon long compared with the slowest one (12 fails,
36 works), a scale-free latent loss with a weight matched to the image size, and enough training
(d = 4 keeps improving up to 12,000 iterations).

## Results

| | |
|---|---|
| eigenvalues of K | e^{±i·0.504°}, e^{±i·6.503°}, all \|λ\| = 1.000 |
| what the modes are | α = θ_h (mode (1,0)), β = θ_h + θ_m (mode (1,1)); minute = arg β − arg α = 5.999°/min |
| one loop | hour 714 steps (11 h 54, true 12 h), minute 60.0 steps |
| reconstruction (pixel MSE) | 0.79×10⁻³ (average clock: 18×10⁻³) |
| prediction from one image | 1.08×10⁻³ at 1 h, 1.75×10⁻³ at 6 h (best linear map on the plain VAE latent: 35 and 27) |

**Why 4 dimensions.** A real 4×4 matrix can rotate two planes, one per hand. Because the minute
hand turns exactly 12× faster, 3 dimensions would suffice in theory (hour rotation + the conserved
c = θ_m − 12θ_h), but reading c requires the hour hand 12× more precisely than the image allows,
so training never finds it.

**Mixed modes.** Koopman fixes the spectrum (the frequencies), not which basis of it the latent
stores: the network learns (hour, hour + minute) rather than (hour, minute). The natural hands are
one change of basis away: θ_h = arg α, θ_m = arg β − arg α.

**What the KL brings.** It does not change the modes. It fixes the latent's scale and makes the
decoder smooth, which keeps long rollouts accurate: 6-hour error ≈ 10–11×10⁻³ for β ≤ 0.5, ≈ 4×10⁻³
for β ≥ 1 (two seeds at β = 0 and 1). Too much KL blurs. Choose β by long-horizon prediction error.

## Layout

```
clocks.py            renders the clocks, writes data/clocks.npz
models.py            encoder, decoder, VAE, KoopmanVAE (VAE + matrix K), losses, load_model
train_vae.py         stage 1   -> runs/minimal/vae.pt (+ _snapshots.npz)
train_koopman.py     stage 2   -> runs/minimal/koopman.pt (+ _snapshots.npz)
analysis/
  timeline.py        latent during training (PCA and final-mode views) -> report/timeline.json
  mode_video.py      turning α, β, both, and time, decoded -> report/media/modes.mp4
  kl_sweep.sh        trains the beta sweep -> runs/minimal/kl/
  kl_study.py        reconstruction, samples, PCA, trade-off per beta -> report/kl.json
viz/                 every visualisation as JSON for slides (format: viz/README.md)
report/
  template.html      the report page
  build.py           computes report/data.json and writes report/index.html
runs/minimal/        the minimal model and the KL sweep
archive/             everything exploratory (d = 16, slowness, distillation, harmonics, fast sweeps, ...)
```

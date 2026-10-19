# Archive

Everything from the exploration that led to the minimal model. Nothing here is needed to
reproduce it (see the top-level README). The scripts are kept as they were: they reference
`runs/...`, `report/...` and `animations/...`, which now live under `archive/runs/`,
`archive/report_full/` and `archive/animations/`, so paths need adjusting to rerun them.

| folder | what |
|---|---|
| `scripts/` | full-option training (`train_koopman_full.py`: slowness, least-squares K, spread/rotation init, slowness-first), distillation (`train_minimal.py`), the 64-px pipeline (`animate.py`, `report_data.py`, `harmonics_data.py`, `vae_explain.py`, `report_timelines.py`), comparisons (`compare_pair.py`, `analyze_runs.py`), and the pre-refactor versions of the minimal scripts |
| `runs/` | every checkpoint, snapshot and log: d = 16 / 8 / 4 / 3, full-size and fast, one-step loss, slowness, horizons, seeds, distilled d = 4 |
| `report_full/` | the exploratory report (https://claude.ai/artifact/ECsqfMzejotonZDY9vAYQ2) with its data and the 24 training timelines |
| `animations/` | the 64-px explainer videos (dataset, VAE training, Koopman training, rollout, modes) |
| `data/` | the 2-hour trajectories used for the long-horizon d = 3 test |

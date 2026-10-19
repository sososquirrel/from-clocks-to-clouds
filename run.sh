#!/usr/bin/env bash
# Minimal Koopman VAE on clock images, end to end (~6 min; add --kl for the KL study, +25 min).
#   data -> stage 1 (VAE, d = 4) -> stage 2 (Koopman) -> analysis -> report/index.html
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}                                 # e.g. PY=.venv/bin/python ./run.sh

$PY clocks.py                                    # data/clocks.npz: 3,000 runs x 40 minutes
$PY -u train_vae.py                              # runs/minimal/vae.pt
$PY -u train_koopman.py                          # runs/minimal/koopman.pt
$PY -m analysis.timeline                         # report/timeline.json
$PY -m analysis.mode_video                       # report/media/modes.mp4
if [[ "${1:-}" == "--kl" ]]; then
  bash analysis/kl_sweep.sh                      # runs/minimal/kl/*.pt
  $PY -m analysis.kl_study                       # report/kl.json
fi
$PY -m report.build                              # report/data.json, report/index.html

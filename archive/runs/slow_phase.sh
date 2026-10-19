#!/usr/bin/env bash
# Slowness first (1,000 it.), then Koopman on (3,000 it.), w_slow = 100; then compare all slowness runs.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while pgrep -f "w_slow 1000" >/dev/null; do sleep 15; done          # let the first slowness runs finish
$PY -u train_koopman.py --vae runs/vae.pt    --iters 4000 --koopman_start 1000 --w_slow 100 --tag _d16_sf > runs/koopman_d16_sf_log.txt 2>&1 &
$PY -u train_koopman.py --vae runs/vae_d4.pt --iters 4000 --koopman_start 1000 --w_slow 100 --tag _d4_sf  > runs/koopman_d4_sf_log.txt 2>&1 &
wait
$PY analyze_runs.py report/slowness.png \
  "d=16, no slowness (reference)=runs/koopman.pt" \
  "d=16, slowness 1000 from the start=runs/koopman_d16_slow.pt" \
  "d=16, slowness 100 first, then Koopman=runs/koopman_d16_sf.pt" \
  "d=4, slowness 1000 from the start=runs/koopman_d4_slow.pt" \
  "d=4, slowness 100 first, then Koopman=runs/koopman_d4_sf.pt" > runs/slowness_analysis.txt 2>&1
echo ALL DONE

#!/usr/bin/env bash
# Koopman from scratch with small latents (d = 4 and d = 8), half-length training
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -u train_koopman.py --vae runs/vae_d4.pt --iters 3000 --tag _d4_id > runs/koopman_d4_id_log.txt 2>&1 &
$PY -u train_koopman.py --vae runs/vae_d4.pt --iters 3000 --K_init_deg 3,10 --tag _d4_rot > runs/koopman_d4_rot_log.txt 2>&1 &
$PY -u train_vae.py --d 8 --epochs 10 --tag _d8 > runs/vae_d8_log.txt 2>&1
$PY -u train_koopman.py --vae runs/vae_d8.pt --iters 3000 --tag _d8_id > runs/koopman_d8_id_log.txt 2>&1 &
$PY -u train_koopman.py --vae runs/vae_d8.pt --iters 3000 --K_init_deg 1,3,10,20 --tag _d8_rot > runs/koopman_d8_rot_log.txt 2>&1 &
wait
$PY analyze_runs.py report/small_latents.png "d=4, K starts at identity=runs/koopman_d4_id.pt" "d=4, K starts as rotations (3°, 10°)=runs/koopman_d4_rot.pt" \
  "d=8, K starts at identity=runs/koopman_d8_id.pt" "d=8, K starts as rotations (1°, 3°, 10°, 20°)=runs/koopman_d8_rot.pt" \
  "d=16 reference (6,000 it.)=runs/koopman.pt" > runs/small_latents_analysis.txt 2>&1
echo ALL DONE

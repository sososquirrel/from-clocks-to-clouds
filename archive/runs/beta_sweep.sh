#!/usr/bin/env bash
# beta sweep for the d = 4 pipeline (fast setup), plus a second seed for beta 0 and 1
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
K="--bs 16 --snap_every 100 --iters 6000 --H_max 36 --w_lat 125"
for b in 0.25 0.5 2; do
  t=$(echo $b | tr . p)
  $PY -u train_vae.py --d 4 --size 32 --width 0.5 --epochs 15 --snap_every 100 --beta $b --tag _fast4_b$t > runs/vae_fast4_b${t}_log.txt 2>&1
  $PY -u train_koopman.py --vae runs/vae_fast4_b$t.pt $K --beta $b --tag _f4_b$t > runs/koopman_f4_b${t}_log.txt 2>&1
done
$PY -u train_koopman.py --vae runs/vae_fast4_b0.pt $K --beta 0 --seed 1 --tag _f4_b0_s1 > runs/koopman_f4_b0_s1_log.txt 2>&1
$PY -u train_koopman.py --vae runs/vae_fast4.pt    $K --beta 1 --seed 1 --tag _f4_b1_s1 > runs/koopman_f4_b1_s1_log.txt 2>&1
echo ALL DONE

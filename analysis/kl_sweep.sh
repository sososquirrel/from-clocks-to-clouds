#!/usr/bin/env bash
# KL study: the minimal pipeline trained with different KL weights beta (both stages), Koopman stage at
# 6,000 iterations, plus a second seed for beta = 0 and 1. ~25 min on Apple silicon.
# Then: python -m analysis.kl_study
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=runs/minimal/kl
mkdir -p $OUT
for b in 0 0.25 0.5 1 2 4; do
  t=$(echo $b | tr . p)
  $PY -u train_vae.py --beta $b --out $OUT/vae_b$t.pt
  $PY -u train_koopman.py --vae $OUT/vae_b$t.pt --beta $b --iters 6000 --out $OUT/koopman_b$t.pt
done
for b in 0 1; do
  $PY -u train_koopman.py --vae $OUT/vae_b$b.pt --beta $b --iters 6000 --seed 1 --out $OUT/koopman_b${b}_s1.pt
done

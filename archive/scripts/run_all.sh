#!/usr/bin/env bash
# Full pipeline: data -> stage 1 (VAE) -> stage 2 (Koopman VAE) -> animations
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p runs
$PY clocks.py
$PY -u train_vae.py --epochs 20     | tee runs/vae_log.txt
$PY -u train_koopman.py             | tee runs/koopman_log.txt
$PY animate.py

# Optional experiment: Koopman loss on single pairs (H = 1) vs clips
$PY -u train_koopman.py --H_max 1 --tag _pair | tee runs/koopman_pair_log.txt
$PY compare_pair.py

# Minimal Koopman (d = 4), distilled from the 16-D model's hand eigenfunctions
$PY -u train_minimal.py | tee runs/minimal_log.txt

# HTML report (timelines are separate files under report/timelines/, published next to the page)
$PY harmonics_data.py
$PY vae_explain.py
$PY report_timelines.py
$PY report_data.py
$PY - <<'PYEOF'
t = open("report/template.html").read(); d = open("report/data.json").read()
open("report/index.html", "w").write(t.replace("__DATA__", d.replace("</", "<\\/")))
PYEOF

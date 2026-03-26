#!/usr/bin/env bash
set -euo pipefail

python main.py \
  -eta1 8 -eta2 6 --k 5 -step 20 \
  -dn Salinas --task inpainting --task_params 0.8 \
  --dataroot data --data_file chaos_traffic.mat \
  --rank 6 --posterior_update_steps 1 \
  --adapter_lr 1e-4 --factor_lr 5e-3 --adapter_hidden 16 \
  -gpu 0 --beta_schedule exp "$@"

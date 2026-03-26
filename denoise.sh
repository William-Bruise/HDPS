#!/usr/bin/env bash
set -euo pipefail

python main.py \
  -eta1 16 -eta2 10 --k 8 -step 20 \
  -dn Houston --task denoise --task_params 50 \
  --dataroot data --data_file car.mat \
  --rank 6 --posterior_update_steps 1 \
  --adapter_lr 1e-4 --factor_lr 5e-3 --adapter_hidden 16 \
  -gpu 0 --beta_schedule exp "$@"

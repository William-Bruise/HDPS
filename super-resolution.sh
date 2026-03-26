#!/usr/bin/env bash
set -euo pipefail

python main.py \
  -eta1 500 -eta2 12 --k 8 -step 20 \
  -dn WDC --task sr --task_params 0.25 \
  --dataroot data --data_file animal_garden.mat \
  --rank 6 --posterior_update_steps 1 \
  --adapter_lr 1e-4 --factor_lr 5e-3 --adapter_hidden 16 \
  -gpu 0 --beta_schedule exp "$@"

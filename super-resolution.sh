#!/usr/bin/env bash
set -euo pipefail

# Grid search for super-resolution task.
# Data reading related args are intentionally kept unchanged.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

dataroot="data"
data_file="animal_garden.mat"
dataname="WDC"
task="sr"
task_params="0.25"
gpu="2"
beta_schedule="exp"

eta1_grid=(300 500 700)
eta2_grid=(8 12)
k_grid=(6 8)
step_grid=(20 30)
rank_grid=(4 6)
posterior_steps_grid=(1 2)
adapter_lr_grid=(1e-4 5e-5)
factor_lr_grid=(5e-3 1e-3)
adapter_hidden_grid=(16)

extra_args=("$@")

best_psnr="-inf"
best_cfg=""
run_id=0

for eta1 in "${eta1_grid[@]}"; do
  for eta2 in "${eta2_grid[@]}"; do
    for k in "${k_grid[@]}"; do
      for step in "${step_grid[@]}"; do
        for rank in "${rank_grid[@]}"; do
          for posterior_steps in "${posterior_steps_grid[@]}"; do
            for adapter_lr in "${adapter_lr_grid[@]}"; do
              for factor_lr in "${factor_lr_grid[@]}"; do
                for adapter_hidden in "${adapter_hidden_grid[@]}"; do
                  run_id=$((run_id + 1))
                  log_file="grid_sr_run_${run_id}.log"

                  echo "[GRID][sr][run ${run_id}] eta1=${eta1} eta2=${eta2} k=${k} step=${step} rank=${rank} posterior_steps=${posterior_steps} adapter_lr=${adapter_lr} factor_lr=${factor_lr} adapter_hidden=${adapter_hidden}"

                  if python main.py \
                    -eta1 "${eta1}" -eta2 "${eta2}" --k "${k}" -step "${step}" \
                    -dn "${dataname}" --task "${task}" --task_params "${task_params}" \
                    --dataroot "${dataroot}" --data_file "${data_file}" \
                    --rank "${rank}" --posterior_update_steps "${posterior_steps}" \
                    --adapter_lr "${adapter_lr}" --factor_lr "${factor_lr}" --adapter_hidden "${adapter_hidden}" \
                    -gpu "${gpu}" --beta_schedule "${beta_schedule}" "${extra_args[@]}" | tee "${log_file}"; then
                    run_status="ok"
                  else
                    run_status="failed"
                  fi

                  if [[ "${run_status}" == "failed" ]]; then
                    if grep -qiE "outofmemoryerror|cuda out of memory" "${log_file}"; then
                      echo "[GRID][sr][run ${run_id}] OOM detected, skip this config and continue."
                      sleep 2
                      continue
                    fi
                    echo "[GRID][sr][run ${run_id}] failed (non-OOM), stop search."
                    exit 1
                  fi

                  run_psnr=$(python - "${log_file}" <<'PY'
import re, sys
text = open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').read()
vals = re.findall(r'best psnr:\s*([0-9]+(?:\.[0-9]+)?)', text)
print(vals[-1] if vals else "nan")
PY
)

                  if [[ "${run_psnr}" != "nan" ]]; then
                    better=$(python - "${run_psnr}" "${best_psnr}" <<'PY'
import sys
cur = float(sys.argv[1])
best = float('-inf') if sys.argv[2] == '-inf' else float(sys.argv[2])
print('1' if cur > best else '0')
PY
)
                    if [[ "${better}" == "1" ]]; then
                      best_psnr="${run_psnr}"
                      best_cfg="eta1=${eta1}, eta2=${eta2}, k=${k}, step=${step}, rank=${rank}, posterior_update_steps=${posterior_steps}, adapter_lr=${adapter_lr}, factor_lr=${factor_lr}, adapter_hidden=${adapter_hidden}"
                    fi
                  fi

                  echo "[GRID][sr][run ${run_id}] psnr=${run_psnr} | best_psnr=${best_psnr}"
                  sleep 1
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "[GRID][sr] search done"
echo "[GRID][sr] best_psnr=${best_psnr}"
echo "[GRID][sr] best_cfg=${best_cfg}"

#!/usr/bin/env bash
set -euo pipefail

# Full grid search for inpainting hyperparameters.
# This script exhausts all combinations in the configured grids.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# data / task
dataroot="data"
data_file="chaos_traffic.mat"
dataname="Salinas"
task="inpainting"
task_params="0.8"
gpu="2"
beta_schedule="exp"

# compact candidate sets (further reduced)
eta1_grid=(6 8 12)
eta2_grid=(2 4 6)
k_grid=(4 6)
step_grid=(20 24)
rank_grid=(4 6)
posterior_steps_grid=(20 40)
adapter_lr="1e-3"
factor_lr="5e-3"
adapter_hidden="128"

extra_args=("$@")
best_psnr="-inf"
best_cfg=""
run_id=0
total_combos=0
log_file="search_inpainting.log"
result_file="search_inpainting_results.tsv"
: > "${log_file}"
: > "${result_file}"
declare -A seen_configs

is_oom_risk() {
  # Pre-filter is intentionally disabled: always try sampled configs.
  return 1
}

run_config() {
  local eta1="$1" eta2="$2" k="$3" step="$4" rank="$5" posterior_steps="$6"
  local key="${eta1}|${eta2}|${k}|${step}|${rank}|${posterior_steps}|${adapter_lr}|${factor_lr}|${adapter_hidden}"

  if [[ -n "${seen_configs[$key]:-}" ]]; then
    return 0
  fi
  seen_configs[$key]=1

  if is_oom_risk "${step}" "${rank}" "${posterior_steps}" "${adapter_hidden}"; then
    echo "[SEARCH][inpainting_factor] skip high-risk OOM config: ${key}" | tee -a "${log_file}"
    return 0
  fi

  run_id=$((run_id + 1))
  local run_log=".search_run_${run_id}.log"

  echo "[SEARCH][inpainting_factor][run ${run_id}/${total_combos}] eta1=${eta1} eta2=${eta2} k=${k} step=${step} rank=${rank} posterior_steps=${posterior_steps} adapter_lr=${adapter_lr} factor_lr=${factor_lr} adapter_hidden=${adapter_hidden}" | tee -a "${log_file}"

  if python main_factor_adapter.py \
    -eta1 "${eta1}" -eta2 "${eta2}" --k "${k}" -step "${step}" \
    -dn "${dataname}" --task "${task}" --task_params "${task_params}" \
    --dataroot "${dataroot}" --data_file "${data_file}" \
    --rank "${rank}" --posterior_update_steps "${posterior_steps}" \
    --adapter_lr "${adapter_lr}" --factor_lr "${factor_lr}" --adapter_hidden "${adapter_hidden}" \
    -gpu "${gpu}" --beta_schedule "${beta_schedule}" "${extra_args[@]}" | tee -a "${log_file}" "${run_log}"; then
    run_status="ok"
  else
    run_status="failed"
  fi

  if [[ "${run_status}" == "failed" ]]; then
    if grep -qiE "outofmemoryerror|cuda out of memory" "${run_log}"; then
      echo "[SEARCH][inpainting_factor][run ${run_id}] OOM detected, skip and continue." | tee -a "${log_file}"
      rm -f "${run_log}"
      sleep 1
      return 0
    fi
    echo "[SEARCH][inpainting_factor][run ${run_id}] failed (non-OOM), stop." | tee -a "${log_file}"
    rm -f "${run_log}"
    exit 1
  fi

  local run_psnr
  run_psnr=$(python - "${run_log}" <<'PY'
import re, sys
text = open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').read()
vals = re.findall(r'best psnr:\s*([0-9]+(?:\.[0-9]+)?)', text)
print(vals[-1] if vals else "nan")
PY
)

  if [[ "${run_psnr}" != "nan" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${run_psnr}" "${eta1}" "${eta2}" "${k}" "${step}" "${rank}" "${posterior_steps}" "${adapter_lr}" "${factor_lr}" "${adapter_hidden}" >> "${result_file}"

    local better
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

  echo "[SEARCH][inpainting_factor][run ${run_id}/${total_combos}] psnr=${run_psnr} | best_psnr=${best_psnr}" | tee -a "${log_file}"
  rm -f "${run_log}"
  sleep 1
}

run_full_grid() {
  for eta1 in "${eta1_grid[@]}"; do
    for eta2 in "${eta2_grid[@]}"; do
      for k in "${k_grid[@]}"; do
        for step in "${step_grid[@]}"; do
          for rank in "${rank_grid[@]}"; do
            for posterior_steps in "${posterior_steps_grid[@]}"; do
              run_config "${eta1}" "${eta2}" "${k}" "${step}" "${rank}" "${posterior_steps}"
            done
          done
        done
      done
    done
  done
}

total_combos=$(( ${#eta1_grid[@]} * ${#eta2_grid[@]} * ${#k_grid[@]} * ${#step_grid[@]} * ${#rank_grid[@]} * ${#posterior_steps_grid[@]} ))
echo "[SEARCH][inpainting_factor] mode=grid total_combos=${total_combos}"
run_full_grid

echo "[SEARCH][inpainting_factor] search done"
echo "[SEARCH][inpainting_factor] best_psnr=${best_psnr}"
echo "[SEARCH][inpainting_factor] best_cfg=${best_cfg}"

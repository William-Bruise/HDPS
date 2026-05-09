#!/usr/bin/env bash
set -euo pipefail

# Fast and safer random search for inpainting hyperparameters.
# Goals:
#   1) smaller search space (faster)
#   2) skip high-risk OOM configs
#   3) continue on OOM instead of aborting

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RANDOM_TRIALS="${RANDOM_TRIALS:-24}"
RANDOM_SEED="${RANDOM_SEED:-42}"

# data / task
dataroot="data"
data_file="chaos_traffic.mat"
dataname="Salinas"
task="inpainting"
task_params="0.8"
gpu="2"
beta_schedule="exp"

# compact candidate sets (reduced from original huge grid)
eta1_grid=(6 8 12)
eta2_grid=(2 4 6)
k_grid=(4 6)
step_grid=(20 24)
rank_grid=(4 6)
posterior_steps_grid=(1 3 5 10)
adapter_lr_grid=(1e-3 5e-4 1e-4)
factor_lr_grid=(5e-3 1e-3)
adapter_hidden_grid=(16 32 64)

extra_args=("$@")
best_psnr="-inf"
best_cfg=""
run_id=0
log_file="search_inpainting.log"
result_file="search_inpainting_results.tsv"
: > "${log_file}"
: > "${result_file}"
declare -A seen_configs

pick_random() {
  local -n arr_ref="$1"
  local n="${#arr_ref[@]}"
  local idx=$((RANDOM % n))
  echo "${arr_ref[$idx]}"
}

is_oom_risk() {
  local step="$1" rank="$2" posterior_steps="$3" hidden="$4"

  # conservative heuristics: avoid known memory-heavy combinations
  if (( step >= 24 && rank >= 6 && hidden >= 64 )); then
    return 0
  fi
  if (( posterior_steps >= 10 && hidden >= 64 )); then
    return 0
  fi
  if (( step >= 24 && posterior_steps >= 10 )); then
    return 0
  fi
  return 1
}

run_config() {
  local eta1="$1" eta2="$2" k="$3" step="$4" rank="$5" posterior_steps="$6" adapter_lr="$7" factor_lr="$8" adapter_hidden="$9"
  local key="${eta1}|${eta2}|${k}|${step}|${rank}|${posterior_steps}|${adapter_lr}|${factor_lr}|${adapter_hidden}"

  if [[ -n "${seen_configs[$key]:-}" ]]; then
    return 0
  fi
  seen_configs[$key]=1

  if is_oom_risk "${step}" "${rank}" "${posterior_steps}" "${adapter_hidden}"; then
    echo "[SEARCH][inpainting] skip high-risk OOM config: ${key}" | tee -a "${log_file}"
    return 0
  fi

  run_id=$((run_id + 1))
  local run_log=".search_run_${run_id}.log"

  echo "[SEARCH][inpainting][run ${run_id}] eta1=${eta1} eta2=${eta2} k=${k} step=${step} rank=${rank} posterior_steps=${posterior_steps} adapter_lr=${adapter_lr} factor_lr=${factor_lr} adapter_hidden=${adapter_hidden}" | tee -a "${log_file}"

  if python main.py \
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
      echo "[SEARCH][inpainting][run ${run_id}] OOM detected, skip and continue." | tee -a "${log_file}"
      rm -f "${run_log}"
      sleep 1
      return 0
    fi
    echo "[SEARCH][inpainting][run ${run_id}] failed (non-OOM), stop." | tee -a "${log_file}"
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

  echo "[SEARCH][inpainting][run ${run_id}] psnr=${run_psnr} | best_psnr=${best_psnr}" | tee -a "${log_file}"
  rm -f "${run_log}"
  sleep 1
}

run_random_search() {
  RANDOM="${RANDOM_SEED}"
  local attempts=0
  while (( run_id < RANDOM_TRIALS )); do
    attempts=$((attempts + 1))
    if (( attempts > RANDOM_TRIALS * 20 )); then
      echo "[SEARCH][inpainting] reached max attempts while sampling unique/safe configs."
      break
    fi

    eta1="$(pick_random eta1_grid)"
    eta2="$(pick_random eta2_grid)"
    k="$(pick_random k_grid)"
    step="$(pick_random step_grid)"
    rank="$(pick_random rank_grid)"
    posterior_steps="$(pick_random posterior_steps_grid)"
    adapter_lr="$(pick_random adapter_lr_grid)"
    factor_lr="$(pick_random factor_lr_grid)"
    adapter_hidden="$(pick_random adapter_hidden_grid)"

    run_config "${eta1}" "${eta2}" "${k}" "${step}" "${rank}" "${posterior_steps}" "${adapter_lr}" "${factor_lr}" "${adapter_hidden}"
  done
}

echo "[SEARCH][inpainting] mode=random trials=${RANDOM_TRIALS} seed=${RANDOM_SEED}"
run_random_search

echo "[SEARCH][inpainting] search done"
echo "[SEARCH][inpainting] best_psnr=${best_psnr}"
echo "[SEARCH][inpainting] best_cfg=${best_cfg}"

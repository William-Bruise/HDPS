#!/usr/bin/env bash
set -euo pipefail

# Full grid search for inpainting hyperparameters.
# This script exhausts all combinations in the configured grids.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# data / task
dataroot="data"
task="inpainting"
DATA_FILES=(animal_garden.mat car.mat chaos_traffic.mat fruit.mat ironman.mat)
TASK_PARAMS=("0.7" "0.8" "0.9")
inpaint_noise_sigma="0"
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
adapter_mode="inpainting_factor"


run_single_case() {
  local data_file="$1" task_params="$2"
  local dataname="${data_file%.mat}"
  local best_psnr="-inf" best_cfg="" best_mat_path=""
  local run_id=0 total_combos=0
  local log_file="search_${adapter_mode}_${dataname}_tp${task_params}.log"
  local result_file="search_${adapter_mode}_${dataname}_tp${task_params}_results.tsv"
  : > "${log_file}"; : > "${result_file}"
  declare -A seen_configs=()

is_oom_risk() {
  # Pre-filter is intentionally disabled: always try sampled configs.
  return 1
}

  run_config() {
  local eta1="$1" eta2="$2" k="$3" step="$4" rank="$5" posterior_steps="$6"
  local key="${eta1}|${eta2}|${k}|${step}|${rank}|${posterior_steps}|${adapter_lr}|${factor_lr}|${adapter_hidden}"

  # Guard against invalid diffusion schedule shape parameter.
  if [[ "${k}" == "0" || "${k}" == "0.0" ]]; then
    echo "[SEARCH][inpainting_factor] skip invalid config (k must be > 0): ${key}" | tee -a "${log_file}"
    return 0
  fi

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
    --inpaint_noise_sigma "${inpaint_noise_sigma}" \
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
    if grep -qiE "AssertionError|betas > 0|betas <= 1" "${run_log}"; then
      echo "[SEARCH][inpainting_factor][run ${run_id}] invalid beta schedule params, skip and continue." | tee -a "${log_file}"
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
  local run_mat_path
  run_mat_path=$(python - "${run_log}" <<'PY'
import re, sys
text = open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').read()
vals = re.findall(r'\[INFO\]\s+Saved best output mat to:\s*(.+)', text)
print(vals[-1].strip() if vals else "")
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
      best_mat_path="${run_mat_path}"
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
  echo "[SEARCH][inpainting_factor] data=${dataname} task_params=${task_params} mode=grid total_combos=${total_combos}"
  run_full_grid

  echo "[SEARCH][inpainting_factor] search done data=${dataname} task_params=${task_params}"
  echo "[SEARCH][inpainting_factor] best_psnr=${best_psnr}"
  echo "[SEARCH][inpainting_factor] best_cfg=${best_cfg}"
  if [[ -n "${best_mat_path}" && -f "${best_mat_path}" ]]; then
    summary_dir="results/search_best/inpainting_factor"
    mkdir -p "${summary_dir}"
    output_mat="${summary_dir}/${dataname}_${task_params}.mat"
    cp -f "${best_mat_path}" "${output_mat}"
    cat > "${summary_dir}/${dataname}_${task_params}_best_params.txt" <<EOF
best_psnr=${best_psnr}
best_cfg=${best_cfg}
source_mat=${best_mat_path}
output_mat=${output_mat}
task=${task}
task_params=${task_params}
dataname=${dataname}
data_file=${data_file}
eta1_grid=${eta1_grid[*]}
eta2_grid=${eta2_grid[*]}
k_grid=${k_grid[*]}
step_grid=${step_grid[*]}
rank_grid=${rank_grid[*]}
posterior_steps_grid=${posterior_steps_grid[*]}
adapter_lr=${adapter_lr}
factor_lr=${factor_lr}
adapter_hidden=${adapter_hidden}
EOF
    echo "[SEARCH][inpainting_factor] best mat saved to ${output_mat}"
  fi
}

for data_file in "${DATA_FILES[@]}"; do
  for task_params in "${TASK_PARAMS[@]}"; do
    run_single_case "${data_file}" "${task_params}"
  done
done

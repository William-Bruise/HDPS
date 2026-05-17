#!/usr/bin/env bash
set -euo pipefail

# Vanilla HIR-Diff (no adapter + no additive spectral matrix finetune).
# Coarse-to-fine search: coarse grid first, then fine search around coarse top-k.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TOP_K="${TOP_K:-3}"
RANDOM_SEED="${RANDOM_SEED:-42}"

dataroot="data"
task="inpainting"
DATA_FILES=(animal_garden.mat car.mat chaos_traffic.mat fruit.mat ironman.mat)
TASK_PARAMS=("0.7" "0.8" "0.9")
inpaint_noise_sigma="0"
gpu="2"
beta_schedule="exp"

eta1_grid=(4 8 12 16)
eta2_grid=(1 2 4 8)
k_grid=(4 6 8 10)
step_grid=(10 20 30 40)
rank_grid=(3)

extra_args=("$@")

run_single_case() {
  local data_file="$1" task_params="$2"
  local dataname="${DATANAME:-Salinas}"
  local best_psnr="-inf" best_cfg="" best_mat_path=""
  local run_id=0
  local log_file="grid_inpainting_vanilla_${dataname}_tp${task_params}.log"
  local result_file="grid_inpainting_vanilla_${dataname}_tp${task_params}_results.tsv"
  : > "${log_file}"
  : > "${result_file}"
  declare -A seen_configs=()

  run_config() {
    local eta1="$1" eta2="$2" k="$3" step="$4" rank="$5"
    local key="${eta1}|${eta2}|${k}|${step}|${rank}"
    if [[ -n "${seen_configs[$key]:-}" ]]; then return 0; fi
    seen_configs[$key]=1

    run_id=$((run_id + 1))
    local run_log=".grid_vanilla_run_${run_id}.log"
    echo "[VANILLA][inpainting][${dataname}][tp=${task_params}][run ${run_id}] eta1=${eta1} eta2=${eta2} k=${k} step=${step} rank=${rank}"

    if python main.py --vanilla_hirdiff \
      -eta1 "${eta1}" -eta2 "${eta2}" --k "${k}" -step "${step}" \
      -dn "${dataname}" --task "${task}" --task_params "${task_params}" \
      --inpaint_noise_sigma "${inpaint_noise_sigma}" \
      --dataroot "${dataroot}" --data_file "${data_file}" \
      --rank "${rank}" -gpu "${gpu}" --beta_schedule "${beta_schedule}" "${extra_args[@]}" | tee -a "${log_file}" "${run_log}"; then
      :
    else
      if grep -qiE "outofmemoryerror|cuda out of memory" "${run_log}"; then
        echo "[VANILLA][inpainting][run ${run_id}] OOM detected, skip."
        rm -f "${run_log}"; sleep 1; return 0
      fi
      echo "[VANILLA][inpainting][run ${run_id}] failed (non-OOM), stop."
      rm -f "${run_log}"; exit 1
    fi

    local run_psnr run_mat_path
    run_psnr=$(python - "${run_log}" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
vals=re.findall(r'best psnr:\s*([0-9]+(?:\.[0-9]+)?)', text)
print(vals[-1] if vals else 'nan')
PY
)
    run_mat_path=$(python - "${run_log}" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
vals=re.findall(r'\[INFO\]\s+Saved best output mat to:\s*(.+)', text)
print(vals[-1].strip() if vals else '')
PY
)

    if [[ "${run_psnr}" != "nan" ]]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${run_psnr}" "${eta1}" "${eta2}" "${k}" "${step}" "${rank}" >> "${result_file}"
      better=$(python - "${run_psnr}" "${best_psnr}" <<'PY'
import sys
cur=float(sys.argv[1]);best=float('-inf') if sys.argv[2]=='-inf' else float(sys.argv[2])
print(1 if cur>best else 0)
PY
)
      if [[ "${better}" == "1" ]]; then
        best_psnr="${run_psnr}"
        best_cfg="eta1=${eta1}, eta2=${eta2}, k=${k}, step=${step}, rank=${rank}"
        best_mat_path="${run_mat_path}"
      fi
    fi
    rm -f "${run_log}"
  }

  run_coarse_to_fine() {
    mapfile -t coarse_cfgs < <(python - <<PY
import itertools
def coarse(arr):
  return arr if len(arr)<=3 else [arr[i] for i in sorted(set([0,len(arr)//2,len(arr)-1]))]
for c in itertools.product(coarse("${eta1_grid[*]}".split()), coarse("${eta2_grid[*]}".split()), coarse("${k_grid[*]}".split()), coarse("${step_grid[*]}".split()), coarse("${rank_grid[*]}".split())):
  print('\t'.join(c))
PY
)
    for line in "${coarse_cfgs[@]}"; do IFS=$'\t' read -r a b c d e <<<"${line}"; run_config "$a" "$b" "$c" "$d" "$e"; done

    [[ -s "${result_file}" ]] || return
    local top_file=".grid_vanilla_top.tsv"
    sort -t $'\t' -k1,1nr "${result_file}" | head -n "${TOP_K}" > "${top_file}"
    mapfile -t fine_cfgs < <(python - <<PY
import itertools
from pathlib import Path
def parse_arr(s): return s.split()
def nbh(arr,val):
  i=arr.index(val); return [arr[j] for j in [i-1,i,i+1] if 0<=j<len(arr)]
arrs=[parse_arr("${eta1_grid[*]}"),parse_arr("${eta2_grid[*]}"),parse_arr("${k_grid[*]}"),parse_arr("${step_grid[*]}"),parse_arr("${rank_grid[*]}")]
sel=[set() for _ in arrs]
for line in Path('.grid_vanilla_top.tsv').read_text().splitlines():
  for idx,val in enumerate(line.split('\t')[1:]): sel[idx].update(nbh(arrs[idx],val))
final=[sorted(s,key=lambda x:arrs[i].index(x)) if s else arrs[i] for i,s in enumerate(sel)]
for c in itertools.product(*final): print('\t'.join(c))
PY
)
    rm -f "${top_file}"
    for line in "${fine_cfgs[@]}"; do IFS=$'\t' read -r a b c d e <<<"${line}"; run_config "$a" "$b" "$c" "$d" "$e"; done
  }

  run_coarse_to_fine
  if [[ -n "${best_mat_path}" && -f "${best_mat_path}" ]]; then
    summary_dir="results/search_best/inpainting_vanilla"
    mkdir -p "${summary_dir}"
    output_mat="${summary_dir}/${dataname}_${task_params}.mat"
    cp -f "${best_mat_path}" "${output_mat}"
    cat > "${summary_dir}/${dataname}_${task_params}_best_params.txt" <<EOT
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
posterior_steps_grid=
adapter_lr=
factor_lr=
adapter_hidden=
EOT
  fi
}

for data_file in "${DATA_FILES[@]}"; do
  for task_params in "${TASK_PARAMS[@]}"; do
    run_single_case "${data_file}" "${task_params}"
  done
done

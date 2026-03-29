#!/usr/bin/env bash
set -euo pipefail

# Search modes:
#   SEARCH_MODE=grid   -> full Cartesian grid
#   SEARCH_MODE=hybrid -> coarse grid first, then fine grid around coarse best (default)

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEARCH_MODE="${SEARCH_MODE:-hybrid}"
COARSE_RANDOM_N="${COARSE_RANDOM_N:-120}"
TOP_K="${TOP_K:-3}"
RANDOM_SEED="${RANDOM_SEED:-42}"

dataroot="data"
data_file="car.mat"
dataname="Houston"
task="denoise"
task_params="50"
gpu="2"
beta_schedule="exp"

eta1_grid=(8 16 32)
eta2_grid=(2 6 10)
k_grid=(6 8 10)
step_grid=(20 30)
rank_grid=(4 6 8)
posterior_steps_grid=(1 5 10 20 50 100)
adapter_lr_grid=(1e-2 5e-3 1e-3 5e-4 1e-4)
factor_lr_grid=(1e-2 5e-3 1e-3)
adapter_hidden_grid=(32 64 128 256)

extra_args=("$@")
best_psnr="-inf"
best_cfg=""
run_id=0
log_file="grid_denoise.log"
result_file="grid_denoise_results.tsv"
: > "${log_file}"
: > "${result_file}"
declare -A seen_configs

run_config() {
  local eta1="$1" eta2="$2" k="$3" step="$4" rank="$5" posterior_steps="$6" adapter_lr="$7" factor_lr="$8" adapter_hidden="$9"
  local key="${eta1}|${eta2}|${k}|${step}|${rank}|${posterior_steps}|${adapter_lr}|${factor_lr}|${adapter_hidden}"
  if [[ -n "${seen_configs[$key]:-}" ]]; then
    return 0
  fi
  seen_configs[$key]=1

  run_id=$((run_id + 1))
  local run_log=".grid_run_${run_id}.log"

  echo "[GRID][denoise][run ${run_id}] eta1=${eta1} eta2=${eta2} k=${k} step=${step} rank=${rank} posterior_steps=${posterior_steps} adapter_lr=${adapter_lr} factor_lr=${factor_lr} adapter_hidden=${adapter_hidden}"

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
      echo "[GRID][denoise][run ${run_id}] OOM detected, skip this config and continue."
      rm -f "${run_log}"
      sleep 2
      return 0
    fi
    echo "[GRID][denoise][run ${run_id}] failed (non-OOM), stop search."
    rm -f "${run_log}"
    exit 1
  fi

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

  echo "[GRID][denoise][run ${run_id}] psnr=${run_psnr} | best_psnr=${best_psnr}"
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
              for adapter_lr in "${adapter_lr_grid[@]}"; do
                for factor_lr in "${factor_lr_grid[@]}"; do
                  for adapter_hidden in "${adapter_hidden_grid[@]}"; do
                    run_config "${eta1}" "${eta2}" "${k}" "${step}" "${rank}" "${posterior_steps}" "${adapter_lr}" "${factor_lr}" "${adapter_hidden}"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
}

run_hybrid_search() {
  mapfile -t coarse_cfgs < <(python - <<PY
import itertools

def coarse(arr):
    if len(arr) <= 3:
        return arr
    idxs = sorted(set([0, len(arr)//2, len(arr)-1]))
    return [arr[i] for i in idxs]

eta1=coarse("${eta1_grid[*]}".split())
eta2=coarse("${eta2_grid[*]}".split())
k=coarse("${k_grid[*]}".split())
step=coarse("${step_grid[*]}".split())
rank=coarse("${rank_grid[*]}".split())
post=coarse("${posterior_steps_grid[*]}".split())
alr=coarse("${adapter_lr_grid[*]}".split())
flr=coarse("${factor_lr_grid[*]}".split())
ah=coarse("${adapter_hidden_grid[*]}".split())
for c in itertools.product(eta1,eta2,k,step,rank,post,alr,flr,ah):
    print('\t'.join(c))
PY
)

  for line in "${coarse_cfgs[@]}"; do
    IFS=$'\t' read -r a b c d e f g h i <<<"${line}"
    run_config "$a" "$b" "$c" "$d" "$e" "$f" "$g" "$h" "$i"
  done

  if [[ ! -s "${result_file}" ]]; then
    return
  fi

  local top_file=".grid_top.tsv"
  sort -t $'\t' -k1,1nr "${result_file}" | head -n "${TOP_K}" > "${top_file}"

  mapfile -t fine_cfgs < <(python - <<PY
import itertools
from pathlib import Path

def parse_arr(s): return s.split()
eta1=parse_arr("${eta1_grid[*]}")
eta2=parse_arr("${eta2_grid[*]}")
k=parse_arr("${k_grid[*]}")
step=parse_arr("${step_grid[*]}")
rank=parse_arr("${rank_grid[*]}")
post=parse_arr("${posterior_steps_grid[*]}")
alr=parse_arr("${adapter_lr_grid[*]}")
flr=parse_arr("${factor_lr_grid[*]}")
ah=parse_arr("${adapter_hidden_grid[*]}")
arrs=[eta1,eta2,k,step,rank,post,alr,flr,ah]
selected=[set() for _ in arrs]
for line in Path('.grid_top.tsv').read_text().splitlines():
    parts=line.split('\t')[1:]
    for i,val in enumerate(parts):
        arr=arrs[i]
        idx=arr.index(val)
        for j in [idx-1,idx,idx+1]:
            if 0<=j<len(arr):
                selected[i].add(arr[j])
final=[sorted(s,key=lambda x: arrs[i].index(x)) if s else arrs[i] for i,s in enumerate(selected)]
for c in itertools.product(*final):
    print('\t'.join(c))
PY
)

  rm -f "${top_file}"

  for line in "${fine_cfgs[@]}"; do
    IFS=$'\t' read -r a b c d e f g h i <<<"${line}"
    run_config "$a" "$b" "$c" "$d" "$e" "$f" "$g" "$h" "$i"
  done
}

if [[ "${SEARCH_MODE}" == "hybrid" ]]; then
  echo "[GRID][denoise] mode=hybrid(coarse2fine)"
  run_hybrid_search
else
  echo "[GRID][denoise] mode=grid"
  run_full_grid
fi

echo "[GRID][denoise] search done"
echo "[GRID][denoise] best_psnr=${best_psnr}"
echo "[GRID][denoise] best_cfg=${best_cfg}"

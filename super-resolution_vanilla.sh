#!/usr/bin/env bash
set -euo pipefail

# Vanilla HIR-Diff (no adapter + no additive spectral matrix finetune).
# Coarse-to-fine search: coarse grid first, then fine search around coarse top-k.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TOP_K="${TOP_K:-3}"
RANDOM_SEED="${RANDOM_SEED:-42}"

dataroot="data"
data_file="animal_garden.mat"
dataname="WDC"
task="sr"
task_params="0.25"
gpu="2"
beta_schedule="exp"

eta1_grid=(200 400 600 800)
eta2_grid=(2 4 8 12)
k_grid=(6 8 10 12)
step_grid=(10 20 30 40)
rank_grid=(3)

extra_args=("$@")
best_psnr="-inf"
best_cfg=""
run_id=0
log_file="grid_sr_vanilla.log"
result_file="grid_sr_vanilla_results.tsv"
: > "${log_file}"
: > "${result_file}"
declare -A seen_configs

run_config() {
  local eta1="$1" eta2="$2" k="$3" step="$4" rank="$5"
  local key="${eta1}|${eta2}|${k}|${step}|${rank}"
  if [[ -n "${seen_configs[$key]:-}" ]]; then
    return 0
  fi
  seen_configs[$key]=1

  run_id=$((run_id + 1))
  local run_log=".grid_vanilla_run_${run_id}.log"

  echo "[VANILLA][sr][run ${run_id}] eta1=${eta1} eta2=${eta2} k=${k} step=${step} rank=${rank}"

  if python main.py \
    --vanilla_hirdiff \
    -eta1 "${eta1}" -eta2 "${eta2}" --k "${k}" -step "${step}" \
    -dn "${dataname}" --task "${task}" --task_params "${task_params}" \
    --dataroot "${dataroot}" --data_file "${data_file}" \
    --rank "${rank}" \
    -gpu "${gpu}" --beta_schedule "${beta_schedule}" "${extra_args[@]}" | tee -a "${log_file}" "${run_log}"; then
    run_status="ok"
  else
    run_status="failed"
  fi

  if [[ "${run_status}" == "failed" ]]; then
    if grep -qiE "outofmemoryerror|cuda out of memory" "${run_log}"; then
      echo "[VANILLA][sr][run ${run_id}] OOM detected, skip."
      rm -f "${run_log}"
      sleep 2
      return 0
    fi
    echo "[VANILLA][sr][run ${run_id}] failed (non-OOM), stop."
    rm -f "${run_log}"
    exit 1
  fi

  run_psnr=$(python - "${run_log}" <<'PY'
import re, sys
text = open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').read()
vals = re.findall(r'best psnr:\s*([0-9]+(?:\.[0-9]+)?)', text)
print(vals[-1] if vals else 'nan')
PY
)

  if [[ "${run_psnr}" != "nan" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${run_psnr}" "${eta1}" "${eta2}" "${k}" "${step}" "${rank}" >> "${result_file}"
    better=$(python - "${run_psnr}" "${best_psnr}" <<'PY'
import sys
cur=float(sys.argv[1])
best=float('-inf') if sys.argv[2]=='-inf' else float(sys.argv[2])
print('1' if cur>best else '0')
PY
)
    if [[ "${better}" == "1" ]]; then
      best_psnr="${run_psnr}"
      best_cfg="eta1=${eta1}, eta2=${eta2}, k=${k}, step=${step}, rank=${rank}"
    fi
  fi

  echo "[VANILLA][sr][run ${run_id}] psnr=${run_psnr} | best_psnr=${best_psnr}"
  rm -f "${run_log}"
}

run_coarse_to_fine() {
  mapfile -t coarse_cfgs < <(python - <<PY
import itertools

def coarse(arr):
    if len(arr)<=3:
        return arr
    idxs=sorted(set([0, len(arr)//2, len(arr)-1]))
    return [arr[i] for i in idxs]
eta1=coarse("${eta1_grid[*]}".split())
eta2=coarse("${eta2_grid[*]}".split())
k=coarse("${k_grid[*]}".split())
step=coarse("${step_grid[*]}".split())
rank=coarse("${rank_grid[*]}".split())
for c in itertools.product(eta1,eta2,k,step,rank):
    print('\t'.join(c))
PY
)

  for line in "${coarse_cfgs[@]}"; do
    IFS=$'\t' read -r a b c d e <<<"${line}"
    run_config "$a" "$b" "$c" "$d" "$e"
  done

  [[ -s "${result_file}" ]] || return
  local top_file=".grid_vanilla_top.tsv"
  sort -t $'\t' -k1,1nr "${result_file}" | head -n "${TOP_K}" > "${top_file}"

  mapfile -t fine_cfgs < <(python - <<PY
import itertools
from pathlib import Path

def parse_arr(s): return s.split()
def nbh(arr, val):
    i=arr.index(val)
    return [arr[j] for j in [i-1,i,i+1] if 0<=j<len(arr)]
eta1=parse_arr("${eta1_grid[*]}")
eta2=parse_arr("${eta2_grid[*]}")
k=parse_arr("${k_grid[*]}")
step=parse_arr("${step_grid[*]}")
rank=parse_arr("${rank_grid[*]}")
arrs=[eta1,eta2,k,step,rank]
sel=[set() for _ in arrs]
for line in Path('.grid_vanilla_top.tsv').read_text().splitlines():
    parts=line.split('\t')[1:]
    for idx,val in enumerate(parts):
        sel[idx].update(nbh(arrs[idx], val))
final=[sorted(s, key=lambda x: arrs[i].index(x)) if s else arrs[i] for i,s in enumerate(sel)]
for c in itertools.product(*final):
    print('\t'.join(c))
PY
)

  rm -f "${top_file}"
  for line in "${fine_cfgs[@]}"; do
    IFS=$'\t' read -r a b c d e <<<"${line}"
    run_config "$a" "$b" "$c" "$d" "$e"
  done
}

echo "[VANILLA][sr] mode=coarse2fine, top_k=${TOP_K}, seed=${RANDOM_SEED}"
run_coarse_to_fine

echo "[VANILLA][sr] done"
echo "[VANILLA][sr] best_psnr=${best_psnr}"
echo "[VANILLA][sr] best_cfg=${best_cfg}"

#!/bin/bash
# Phase 1 baseline: stock upstream Boltz-2 on the 9 fixed inputs, one GPU process at a time.
# Usage: run_baseline.sh <pass-label> <seed>     e.g. run_baseline.sh A_seed0 0
set -eo pipefail
D="${BOLTZ2_HOME:-$HOME/boltz2-dev}"
PASS="$1"; SEED="$2"
[ -n "$PASS" ] && [ -n "$SEED" ] || { echo "usage: $0 <pass-label> <seed>"; exit 2; }
ENVS="${CONDA_ENVS:-$HOME/miniforge3/envs}"
BOLTZ="${BOLTZ_BIN:-$ENVS/boltzdev/bin/boltz}"
OUT="${OUT_ROOT:-$D/runs/baseline}/$PASS"
mkdir -p "$OUT"
TIMES="$OUT/times.tsv"
[ -f "$TIMES" ] || printf "input\tseed\texit\twall_s\tuser_s\tsys_s\tmax_rss_bytes\tpeak_footprint_bytes\tstarted_utc\n" > "$TIMES"

for Y in "$D"/inputs/yaml/*.yaml; do
  NAME=$(basename "$Y" .yaml)
  if grep -q "^$NAME	" "$TIMES"; then echo "skip $NAME (done)"; continue; fi
  LOG="$OUT/$NAME.log"
  START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  set +e
  # LAUNCHER (optional): a python script that takes `boltz predict` args, e.g. scripts/boltz_nw0_exact.py
  if [ -n "$LAUNCHER" ]; then CMD=("${LAUNCHER_PY:-$ENVS/boltzdev/bin/python}" "$LAUNCHER"); else CMD=("$BOLTZ" predict); fi
  /usr/bin/time -l "${CMD[@]}" "$Y" --out_dir "$OUT" --cache "${CACHE_DIR:-$D/cache}" \
      --diffusion_samples 5 --accelerator "${ACCEL:-gpu}" --no_kernels --seed "$SEED" $EXTRA_ARGS > "$LOG" 2>&1
  RC=$?
  set -e
  WALL=$(grep -E "^ +[0-9.]+ real" "$LOG" | awk '{print $1}')
  USR=$(grep -E "^ +[0-9.]+ real" "$LOG" | awk '{print $3}')
  SYS=$(grep -E "^ +[0-9.]+ real" "$LOG" | awk '{print $5}')
  RSS=$(grep "maximum resident set size" "$LOG" | awk '{print $1}')
  PEAK=$(grep "peak memory footprint" "$LOG" | awk '{print $1}')
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$SEED" "$RC" "$WALL" "$USR" "$SYS" "$RSS" "$PEAK" "$START" >> "$TIMES"
  echo "$NAME seed=$SEED exit=$RC wall=${WALL}s"
done

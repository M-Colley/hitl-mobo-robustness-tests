#!/usr/bin/env bash
# The three experiments the mock TMLR reviews of 2026-11-07 asked for; the analysis is frozen in
# configs/review_round_plan.json (sha256 recorded before this script first ran). LogEI and qNEI, T = 50,
# five initial designs, the two headline cells, each cell in its own directory (a clean twin's resume
# fingerprint holds the error settings, so two cells cannot share one directory's clean twins):
#   E1  seeds 27-36 under bias, drift and ar1 (gaussian is output-boba-confirmatory's), default GP;
#   E2  seeds 7-16 under drift_down, the drift of the other sign, paired with the main sweep's drift runs;
#   E3  seeds 7-16 under the four processes with --gp-prior legacy (BoTorch's default before 0.12).
# Usage: bash run_review_experiments.sh [n_jobs]
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
JOBS="${1:-8}"
LOG=run_review_experiments.log
COMMON=(--acq-list logei,qnei --iterations 50 --initial-samples 5 --n-jobs "$JOBS" --resume)
run() {  # run <output dir> <driver args...>
    local out="$1"; shift
    if [ -f "$out/ARM_COMPLETE" ]; then echo "[skip] $out complete" | tee -a "$LOG"; return; fi
    echo "[run] $out $* $(date -Iseconds)" | tee -a "$LOG"
    "$PY" scripts/bo_synthetic_error_simulation.py "${COMMON[@]}" --output-dir "$out" "$@" >> "$LOG" 2>&1 \
        || { echo "[FAIL] $out $* exit $?" | tee -a "$LOG"; exit 1; }
    for d in "$out"/*/; do
        case "$d" in *analysis*|*evaluation*) continue ;; esac
        "$PY" scripts/evaluate_research_question.py --input-dir "$d" --output-dir "${d}evaluation" >> "$LOG" 2>&1 \
            || { echo "[FAIL] eval $d" | tee -a "$LOG"; exit 1; }
    done
    touch "$out/ARM_COMPLETE"
}
A=(--jitter-stds 1.0 --jitter-iterations 0)
B=(--jitter-stds 5.0 --jitter-iterations 20)
S_FRESH=27,28,29,30,31,32,33,34,35,36
S_MAIN=7,8,9,10,11,12,13,14,15,16
# E1
run output-boba-confirm4p-1s-t1  --error-models bias,drift,ar1 --seeds "$S_FRESH" "${A[@]}"
run output-boba-confirm4p-5s-t21 --error-models bias,drift,ar1 --seeds "$S_FRESH" "${B[@]}"
# E2
run output-boba-driftdown-1s-t1  --error-models drift_down --seeds "$S_MAIN" "${A[@]}"
run output-boba-driftdown-5s-t21 --error-models drift_down --seeds "$S_MAIN" "${B[@]}"
# E3
run output-boba-gplegacy-1s-t1  --error-models gaussian,bias,drift,ar1 --seeds "$S_MAIN" --gp-prior legacy "${A[@]}"
run output-boba-gplegacy-5s-t21 --error-models gaussian,bias,drift,ar1 --seeds "$S_MAIN" --gp-prior legacy "${B[@]}"
echo "[done] $(date -Iseconds)" | tee -a "$LOG"

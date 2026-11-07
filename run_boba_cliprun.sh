#!/usr/bin/env bash
# The running response clip (--response-clip running2): the version of clipping a study can run.
# One margin, fixed before any run: 2 robust SDs of the ratings recorded so far. It acts on the clean twin too,
# so the arm has its own directories and is scored against the standard process (analyse_boba_adaptations):
#   - the four gross-fault settings of output-boba-spike at 0.25 sigma from the first rating, seeds 7-11;
#   - the headline gross-fault setting (15% of trials, SD 20) also on the held-out seeds 12-16;
#   - gaussian error at 1 sigma from the first rating, seeds 7-16 (no gross fault: what the clip costs a study
#     whose rater has none), against the main sweep.
# Every setting has its own directory. A clean twin's resume fingerprint holds the whole run configuration,
# the error settings included, so two spike sizes (or spike and gaussian) cannot share one directory's clean
# twins under --resume; output-boba-spike predates the fingerprint and could.
# Usage: bash run_boba_cliprun.sh [n_jobs]
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
JOBS="${1:-4}"
STEM=output-boba-cliprun2
LOG=run_boba_cliprun.log
COMMON=(--acq-list logei,qnei --iterations 50 --initial-samples 5 --jitter-iterations 0
        --n-jobs "$JOBS" --resume --response-clip running2)
run() {  # run <output dir> <driver args...>
    local out="$1"; shift
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
for spike in "0.05 5" "0.15 5" "0.05 20"; do
    set -- $spike
    run "$STEM-sp$1-$2" --error-models spike --jitter-stds 0.25 --seeds 7,8,9,10,11 \
        --error-spike-std-mode fixed --error-spike-prob "$1" --error-spike-std "$2"
done
run "$STEM-sp0.15-20" --error-models spike --jitter-stds 0.25 --seeds 7,8,9,10,11,12,13,14,15,16 \
    --error-spike-std-mode fixed --error-spike-prob 0.15 --error-spike-std 20
run "$STEM-gaussian" --error-models gaussian --jitter-stds 1.0 --seeds 7,8,9,10,11,12,13,14,15,16
echo "[done] $(date -Iseconds)" | tee -a "$LOG"

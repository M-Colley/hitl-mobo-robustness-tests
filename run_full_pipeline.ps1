# Full evaluation pipeline driver (resumable, sleep-suppressing).
#
# Runs the whole study end-to-end and is safe to re-run after any interruption
# (reboot, sleep, closed editor): every stage uses --resume / idempotent
# skips, so completed per-run CSVs are reused.
#
# Run it from a NORMAL PowerShell terminal (not inside an editor task pane) so
# it survives the editor/session closing:
#
#     powershell -ExecutionPolicy Bypass -File run_full_pipeline.ps1
#
# Stages: oracle selection (if missing) -> composite sweep -> multi-objective
# sweep with the memory-safe qLog acquisitions -> evaluation -> figures.
# On success it writes output\PIPELINE_COMPLETE.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

if (-not (Test-Path output)) { New-Item -ItemType Directory output | Out-Null }
$log = "output\pipeline.log"
if (Test-Path output\PIPELINE_COMPLETE) {
    "PIPELINE_COMPLETE marker present; nothing to do." | Tee-Object -FilePath $log -Append
    exit 0
}
"=== Pipeline (re)started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

# --- Keep the machine awake for the duration (released when the script ends) ---
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class Power {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
'@
# ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x1)
[void][Power]::SetThreadExecutionState([uint32]"0x80000001")

if (-not (Test-Path "PYTHON")) { $PYTHON = "python" }
$SEEDS = "7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26"
$ORACLE = "output\best_oracle_models.json"

function Fail($stage) {
    "FAILED: $stage (exit $LASTEXITCODE) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    [void][Power]::SetThreadExecutionState([uint32]"0x80000000")  # ES_CONTINUOUS only
    exit 1
}

# --- 1. Oracle selection (skip if already chosen) ---
if (-not (Test-Path $ORACLE)) {
    "[1/5] Selecting oracle models..." | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\select_best_oracle_model.py `
        --oracle-models xgboost,lightgbm,catboost,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting `
        --cv-folds 5 --output-path $ORACLE *>> $log
    if ($LASTEXITCODE -ne 0) { Fail "oracle selection" }
} else {
    "[1/5] Oracle selection: reusing $ORACLE" | Tee-Object -FilePath $log -Append
}

# --- 2. Composite sweep (single-objective acquisitions + model-free floors) ---
"[2/5] Composite sweep (resume)..." | Tee-Object -FilePath $log -Append
& $PYTHON scripts\bo_sensor_error_simulation.py `
    --objective composite --acq all `
    --oracle-model auto --oracle-selection-path $ORACLE `
    --seeds $SEEDS --output-dir output --parallel --n-jobs 20 --resume *>> $log
if ($LASTEXITCODE -ne 0) { Fail "composite sweep" }

# --- 3. Multi-objective sweep with the numerically-stable, memory-safe qLog
#        acquisitions. The plain qEHVI/qNEHVI exhaust RAM at 5 objectives
#        (~22 GB/run); the log variants with reduced sampling stay ~1-3 GB, so
#        we can run several in parallel. random/sobol are the model-free floors.
"[3/5] Multi-objective sweep (qLogEHVI/qLogNEHVI, reduced sampling, resume)..." | Tee-Object -FilePath $log -Append
& $PYTHON scripts\bo_sensor_error_simulation.py `
    --objective multi_objective --acq-list qlogehvi,qlognehvi,random,sobol `
    --oracle-model auto --oracle-selection-path $ORACLE `
    --seeds $SEEDS --output-dir output --parallel --n-jobs 10 `
    --acq-raw-samples 128 --acq-mc-samples 64 --acq-num-restarts 5 --resume *>> $log
if ($LASTEXITCODE -ne 0) { Fail "multi-objective sweep" }

# --- 4. Evaluation ---
"[4/5] Evaluating..." | Tee-Object -FilePath $log -Append
& $PYTHON scripts\evaluate_research_question.py `
    --input-dir output --output-dir output\evaluation *>> $log
if ($LASTEXITCODE -ne 0) { Fail "evaluation" }

# --- 5. Figures ---
"[5/5] Building figures..." | Tee-Object -FilePath $log -Append
& $PYTHON scripts\plot_combined_aspects.py `
    --evaluation-dir output\evaluation --output-dir output\evaluation\figures `
    --shortlist pi,logpi,qucb *>> $log
if ($LASTEXITCODE -ne 0) { Fail "figures" }

"=== PIPELINE_COMPLETE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
"complete" | Out-File output\PIPELINE_COMPLETE -Encoding utf8
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")  # release sleep lock
exit 0

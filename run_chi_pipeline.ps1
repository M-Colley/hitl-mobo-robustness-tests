# CHI publication pipeline driver (resumable, sleep-suppressing).
#
# Differences from run_full_pipeline.ps1 (the canonical driver):
#   1. FRESH output directory (output-chi\): the old output\ tree was generated
#      with the pre-review simulator (ramp-only drift, AR(1) cold start, GP
#      fit outside the fallback try). Resuming over it would silently mix old
#      and new behavior, so this run starts clean.
#   2. EXTENDED, human-plausible error design (the README-recommended sweep):
#        --error-models gaussian,bias,drift,ar1
#        --jitter-iterations 0,10,20,40      (0 = noisy from the 1st observation)
#        --response-clip auto                 (ratings stay on the instrument scale)
#      This is ~2.6x the jittered runs of the default design; expect ~30-45 h
#      total on this machine. Both sweeps are fully resumable - re-run this
#      script after any interruption and completed runs are reused.
#   3. Stops after evaluation. The confirmatory follow-up (fresh seeds 27-46)
#      is run separately AFTER inspecting the screening rankings, so the
#      confirmed hypothesis is explicitly selected from screening only.
#
#     powershell -ExecutionPolicy Bypass -File run_chi_pipeline.ps1
#
# On success it writes output-chi\SWEEP_COMPLETE.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
if (-not (Test-Path output-chi)) { New-Item -ItemType Directory output-chi | Out-Null }
$log = "output-chi\pipeline.log"
if (Test-Path output-chi\SWEEP_COMPLETE) {
    "SWEEP_COMPLETE marker present; nothing to do." | Tee-Object -FilePath $log -Append
    exit 0
}
"=== CHI pipeline (re)started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

# --- Keep the machine awake for the duration (released when the script ends) ---
Add-Type @'
using System; using System.Runtime.InteropServices;
public static class Power { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags); }
'@
[void][Power]::SetThreadExecutionState([uint32]"0x80000001")  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$PYTHON = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$SEEDS = "7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26"
$ORACLE = "output-chi\best_oracle_models.json"
$ERRORS = "gaussian,bias,drift,ar1"
$JITS   = "0,10,20,40"

function Fail($stage) {
    "FAILED: $stage (exit $LASTEXITCODE) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    [void][Power]::SetThreadExecutionState([uint32]"0x80000000")  # release
    exit 1
}

# --- 1. Oracle selection (reuse the current ehmi=mean selection) ---------------
if (-not (Test-Path $ORACLE)) {
    if (Test-Path "output\best_oracle_models.json") {
        "[1/4] Reusing existing oracle selection (ehmi = mean target)..." | Tee-Object -FilePath $log -Append
        Copy-Item "output\best_oracle_models.json" $ORACLE
    } else {
        "[1/4] Selecting oracle models..." | Tee-Object -FilePath $log -Append
        & $PYTHON scripts\select_best_oracle_model.py `
            --oracle-models xgboost,lightgbm,catboost,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting `
            --cv-folds 5 --output-path $ORACLE *>> $log
        if ($LASTEXITCODE -ne 0) { Fail "oracle selection" }
    }
} else {
    "[1/4] Oracle selection present: $ORACLE" | Tee-Object -FilePath $log -Append
}

# --- 2. Composite sweep (single-objective acquisitions + model-free floors) ---
"[2/4] Composite sweep (extended error design, resume)... $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
& $PYTHON scripts\bo_sensor_error_simulation.py `
    --objective composite --acq all `
    --oracle-model auto --oracle-selection-path $ORACLE `
    --error-models $ERRORS --jitter-iterations $JITS --response-clip auto `
    --seeds $SEEDS --output-dir output-chi --parallel --n-jobs 20 --resume *>> $log
if ($LASTEXITCODE -ne 0) { Fail "composite sweep" }

# --- 3. Multi-objective sweep --------------------------------------------------
# Plain qEHVI/qNEHVI at REDUCED sampling (~6 GB/run at 5 objectives) so 8
# workers fit in memory; random/sobol are the model-free floors.
"[3/4] Multi-objective sweep (qEHVI/qNEHVI, reduced sampling, resume)... $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
& $PYTHON scripts\bo_sensor_error_simulation.py `
    --objective multi_objective --acq-list qehvi,qnehvi,random,sobol `
    --oracle-model auto --oracle-selection-path $ORACLE `
    --error-models $ERRORS --jitter-iterations $JITS --response-clip auto `
    --seeds $SEEDS --output-dir output-chi --parallel --n-jobs 8 `
    --acq-raw-samples 128 --acq-mc-samples 64 --acq-num-restarts 5 --resume *>> $log
if ($LASTEXITCODE -ne 0) { Fail "multi-objective sweep" }

# --- 4. Evaluation -------------------------------------------------------------
"[4/4] Evaluating... $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
& $PYTHON scripts\evaluate_research_question.py `
    --input-dir output-chi --output-dir output-chi\evaluation *>> $log
if ($LASTEXITCODE -ne 0) { Fail "evaluation" }

"=== SWEEP_COMPLETE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
"complete" | Out-File output-chi\SWEEP_COMPLETE -Encoding utf8
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")  # release sleep lock
exit 0

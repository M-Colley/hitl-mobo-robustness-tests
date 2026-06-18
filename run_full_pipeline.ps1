# Full evaluation pipeline driver (resumable, sleep-suppressing).
#
# Runs the whole study end-to-end and is safe to re-run after any interruption
# (reboot, sleep, closed editor): every stage uses --resume / idempotent skips,
# so completed per-run CSVs are reused.
#
# Run it from a NORMAL PowerShell terminal (not an editor task pane) so it
# survives the editor/session closing:
#
#     powershell -ExecutionPolicy Bypass -File run_full_pipeline.ps1
#
# Stages: oracle selection -> composite sweep -> multi-objective sweep (plain
# qEHVI/qNEHVI at reduced sampling, ~6 GB/run) -> evaluation -> figures.
# On success it writes output\PIPELINE_COMPLETE.
#
# Oracle target: ehmi uses the per-design "mean" oracle (see datasets.json),
# which raises held-out R^2 from ~0.14 to ~0.55. Because ehmi's best model name
# is unchanged, this driver invalidates any ehmi composite CSVs produced with
# the older individual-row oracle before re-running, so --resume cannot reuse
# stale results.

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
using System; using System.Runtime.InteropServices;
public static class Power { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags); }
'@
[void][Power]::SetThreadExecutionState([uint32]"0x80000001")  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$PYTHON = "python"
$SEEDS = "7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26"
$ORACLE = "output\best_oracle_models.json"

function Fail($stage) {
    "FAILED: $stage (exit $LASTEXITCODE) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    [void][Power]::SetThreadExecutionState([uint32]"0x80000000")  # release
    exit 1
}

# Is the oracle selection already the current (ehmi = mean) version?
function Oracle-IsCurrent {
    if (-not (Test-Path $ORACLE)) { return $false }
    try {
        $j = Get-Content $ORACLE -Raw | ConvertFrom-Json
        foreach ($d in $j.datasets) {
            if ($d.name -eq "ehmi") {
                foreach ($p in $d.objectives.PSObject.Properties) {
                    if ($p.Value.oracle_target -ne "mean") { return $false }
                }
                return $true
            }
        }
    } catch { return $false }
    return $false
}

# --- 1. Oracle selection -------------------------------------------------------
if (-not (Oracle-IsCurrent)) {
    "[1/5] (Re)selecting oracle models (ehmi -> mean target)..." | Tee-Object -FilePath $log -Append
    # Invalidate ehmi composite CSVs from any older individual-row oracle FIRST,
    # so an interruption mid-selection can't leave stale results reachable by
    # --resume (ehmi's best model name is unchanged, so filenames collide).
    Get-ChildItem output -Filter "bo_sensor_error_ehmi_composite_*.csv" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    & $PYTHON scripts\select_best_oracle_model.py `
        --oracle-models xgboost,lightgbm,catboost,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting `
        --cv-folds 5 --output-path $ORACLE *>> $log
    if ($LASTEXITCODE -ne 0) { Fail "oracle selection" }
} else {
    "[1/5] Oracle selection: reusing current $ORACLE (ehmi = mean)." | Tee-Object -FilePath $log -Append
}

# --- 2. Composite sweep (single-objective acquisitions + model-free floors) ---
"[2/5] Composite sweep (resume)..." | Tee-Object -FilePath $log -Append
& $PYTHON scripts\bo_sensor_error_simulation.py `
    --objective composite --acq all `
    --oracle-model auto --oracle-selection-path $ORACLE `
    --seeds $SEEDS --output-dir output --parallel --n-jobs 20 --resume *>> $log
if ($LASTEXITCODE -ne 0) { Fail "composite sweep" }

# --- 3. Multi-objective sweep --------------------------------------------------
# Plain qEHVI/qNEHVI at REDUCED sampling: ~6 GB and ~140 s per run at 5
# objectives (vs ~22 GB at full sampling), so ~8 workers fit in 64 GB without
# the BrokenProcessPool OOM. random/sobol are the model-free floors.
"[3/5] Multi-objective sweep (qEHVI/qNEHVI, reduced sampling, resume)..." | Tee-Object -FilePath $log -Append
& $PYTHON scripts\bo_sensor_error_simulation.py `
    --objective multi_objective --acq-list qehvi,qnehvi,random,sobol `
    --oracle-model auto --oracle-selection-path $ORACLE `
    --seeds $SEEDS --output-dir output --parallel --n-jobs 8 `
    --acq-raw-samples 128 --acq-mc-samples 64 --acq-num-restarts 5 --resume *>> $log
if ($LASTEXITCODE -ne 0) { Fail "multi-objective sweep" }

# --- 4. Evaluation -------------------------------------------------------------
"[4/5] Evaluating..." | Tee-Object -FilePath $log -Append
& $PYTHON scripts\evaluate_research_question.py `
    --input-dir output --output-dir output\evaluation *>> $log
if ($LASTEXITCODE -ne 0) { Fail "evaluation" }

# --- 5. Figures ----------------------------------------------------------------
"[5/5] Building figures..." | Tee-Object -FilePath $log -Append
& $PYTHON scripts\plot_combined_aspects.py `
    --evaluation-dir output\evaluation --output-dir output\evaluation\figures `
    --shortlist pi,logpi,qucb *>> $log
if ($LASTEXITCODE -ne 0) { Fail "figures" }

"=== PIPELINE_COMPLETE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
"complete" | Out-File output\PIPELINE_COMPLETE -Encoding utf8
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")  # release sleep lock
exit 0

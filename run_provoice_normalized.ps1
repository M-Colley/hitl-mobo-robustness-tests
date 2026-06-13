# Provoice normalized-composite rerun (fix for the Mental-Demand scale-dominance bug:
# raw averaging makes -MentalDemand 81% of composite variance; --normalize-objective
# rescales constructs to [0,1] first).
# Waits for the main sweep (output\regeneration.log) to finish, then runs
# selection -> simulation -> evaluation into output\provoice_normalized\.
# jitter-stds are empirically calibrated for the normalized scale
# (output\noise_calibration_provoice_normalized.csv: sd_hat=0.065 -> 0.5/1/2/4x grid).
# Idempotent: safe to relaunch after a reboot (--resume).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$log = "output\provoice_normalized.log"

if ((Test-Path $log) -and (Select-String -Path $log -Pattern "PROVOICE_NORMALIZED_COMPLETE" -Quiet)) { exit 0 }
New-Item -ItemType Directory -Force output\provoice_normalized | Out-Null

"=== provoice-normalized driver (re)started $(Get-Date -Format o); waiting for main sweep ===" | Out-File $log -Append
while ($true) {
    if (Test-Path output\regeneration.log) {
        $content = Get-Content output\regeneration.log -Raw
        if ($content -match "REGENERATION_COMPLETE") { break }
        if ($content -match "FAILED:") { "ABORTED: main sweep failed; not starting provoice rerun." | Out-File $log -Append; exit 1 }
    }
    Start-Sleep -Seconds 300
}
"Main sweep complete; starting provoice normalized rerun $(Get-Date -Format o)" | Out-File $log -Append

if (-not (Test-Path output\provoice_normalized\best_oracle_models.json)) {
    python scripts\select_best_oracle_model.py --dataset-config datasets-provoice.json --normalize-objective --oracle-models xgboost,lightgbm,catboost,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting --cv-folds 5 --output-path output\provoice_normalized\best_oracle_models.json *>> $log
    if ($LASTEXITCODE -ne 0) { "FAILED: oracle selection (exit $LASTEXITCODE)" | Out-File $log -Append; exit 1 }
}

python scripts\bo_sensor_error_simulation.py --dataset-config datasets-provoice.json --normalize-objective --acq all --oracle-model auto --oracle-selection-path output\provoice_normalized\best_oracle_models.json --seeds 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 --jitter-stds 0.033,0.065,0.13,0.26 --output-dir output\provoice_normalized --parallel --n-jobs 16 --resume *>> $log
if ($LASTEXITCODE -ne 0) { "FAILED: simulation (exit $LASTEXITCODE)" | Out-File $log -Append; exit 1 }

python scripts\evaluate_research_question.py --input-dir output\provoice_normalized --output-dir output\provoice_normalized\evaluation *>> $log
if ($LASTEXITCODE -ne 0) { "FAILED: evaluation (exit $LASTEXITCODE)" | Out-File $log -Append; exit 1 }

"=== PROVOICE_NORMALIZED_COMPLETE $(Get-Date -Format o) ===" | Out-File $log -Append

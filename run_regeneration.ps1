# Regeneration driver: oracle selection -> 20-seed broad sweep -> evaluation -> figures.
# Launched detached so it survives editor/session restarts. Log: output\regeneration.log
# Idempotent: re-running after an interruption (e.g. reboot) reuses the completed
# oracle selection and all per-run CSVs already on disk (--resume).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

if ((Test-Path output\regeneration.log) -and (Select-String -Path output\regeneration.log -Pattern "REGENERATION_COMPLETE" -Quiet)) {
    exit 0
}

"=== Regeneration (re)started $(Get-Date -Format o) ===" | Out-File output\regeneration.log -Append

if (-not (Test-Path output\best_oracle_models.json)) {
    python scripts\select_best_oracle_model.py --oracle-models xgboost,lightgbm,catboost,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting --cv-folds 5 --output-path output\best_oracle_models.json *>> output\regeneration.log
    if ($LASTEXITCODE -ne 0) { "FAILED: oracle selection (exit $LASTEXITCODE)" | Out-File output\regeneration.log -Append; exit 1 }
} else {
    "Skipping oracle selection: output\best_oracle_models.json exists." | Out-File output\regeneration.log -Append
}

python scripts\bo_sensor_error_simulation.py --acq all --oracle-model auto --oracle-selection-path output\best_oracle_models.json --seeds 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 --output-dir output --parallel --n-jobs 16 --resume *>> output\regeneration.log
if ($LASTEXITCODE -ne 0) { "FAILED: simulation (exit $LASTEXITCODE)" | Out-File output\regeneration.log -Append; exit 1 }

python scripts\evaluate_research_question.py --input-dir output --output-dir output\evaluation *>> output\regeneration.log
if ($LASTEXITCODE -ne 0) { "FAILED: evaluation (exit $LASTEXITCODE)" | Out-File output\regeneration.log -Append; exit 1 }

python scripts\plot_combined_aspects.py --evaluation-dir output\evaluation --output-dir output\evaluation\figures --shortlist pi,logpi,qucb *>> output\regeneration.log
if ($LASTEXITCODE -ne 0) { "FAILED: figures (exit $LASTEXITCODE)" | Out-File output\regeneration.log -Append; exit 1 }

"=== REGENERATION_COMPLETE $(Get-Date -Format o) ===" | Out-File output\regeneration.log -Append

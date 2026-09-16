# E6: the early-replication bundle on the fitted-oracle companion arm.
#
# Same design as output-fitted (three archival datasets, gaussian error at the
# dataset's own sigma_f multiples, onsets 0 and 20, seeds 7-16) with LogEI and
# qNEI and the first ten proposals rated twice, so it pairs cell for cell with
# output-fitted on those two acquisitions. Output: output-fitted-adapt-rep10\<dataset>.
#
#     powershell -ExecutionPolicy Bypass -File run_fitted_adapt.ps1 -Jobs 3

param([int]$Jobs = 3)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$log = "output-boba\adapt-fitted.log"
"=== fitted bundle arm started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
$PYTHON = if ($env:PYTHON) { $env:PYTHON } else { "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" }
$entries = Get-Content "output\per_dataset\manifest.json" | ConvertFrom-Json
foreach ($name in $entries.PSObject.Properties.Name) {
    $e = $entries.$name
    $outdir = "output-fitted-adapt-rep10\$name"
    New-Item -ItemType Directory -Force $outdir | Out-Null
    "[$name] starting $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\bo_sensor_error_simulation.py `
        --dataset-config $e.config --objective composite --acq-list logei,qnei `
        --oracle-model auto --oracle-selection-path output\best_oracle_models.json `
        --iterations 50 --initial-samples 5 `
        --error-models gaussian --jitter-stds $e.jitter_stds --jitter-iterations 0,20 `
        --replicate-first 10 `
        --seeds 7,8,9,10,11,12,13,14,15,16 --output-dir $outdir --parallel --n-jobs $Jobs --resume *>> $log
    if ($LASTEXITCODE -ne 0) { "FAILED: $name" | Tee-Object -FilePath $log -Append; exit 1 }
    & $PYTHON scripts\evaluate_research_question.py --input-dir $outdir --output-dir "$outdir\evaluation" *>> $log
    if ($LASTEXITCODE -ne 0) { "FAILED: eval $name" | Tee-Object -FilePath $log -Append; exit 1 }
}
"done $(Get-Date -Format o)" | Set-Content -Path "output-boba\adapt-fitted.DONE" -Encoding ascii
"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

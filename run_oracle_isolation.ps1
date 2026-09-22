# The oracle-isolation experiment: the fitted-oracle pipeline on synthetic
# archival datasets drawn from each analytic landscape, so the landscape is held
# fixed and only the oracle changes. scripts/oracle_isolation.py builds the data,
# selects each oracle by the companion arm's own grouped cross-validation, and
# writes output-oracle-iso/manifest.csv with the error grid scaled to the fitted
# oracle's sigma_f. This script runs the simulator on each and evaluates it.

$ErrorActionPreference = "Continue"
$PYTHON = "python"
$WORKERS = 8
$log = "run_oracle_isolation.log"
"=== START $(Get-Date -Format o) ===" | Tee-Object -FilePath $log

foreach ($row in (Import-Csv "output-oracle-iso\manifest.csv")) {
    $name = $row.dataset
    $out = "output-oracle-iso\runs\$name"
    $marker = Join-Path $out ".done"
    if (Test-Path $marker) { "[skip] $name" | Tee-Object -FilePath $log -Append; continue }
    New-Item -ItemType Directory -Force $out | Out-Null
    "[run] $name  stds=$($row.jitter_stds)  oracle=$($row.oracle_model)" | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\bo_sensor_error_simulation.py `
        --dataset-config "output-oracle-iso\configs\datasets-$name.json" --objective composite `
        --acq-list logei,qnei,ucb,ei,random,sobol `
        --oracle-model auto --oracle-selection-path "output-oracle-iso\selection\$name.json" `
        --iterations 50 --initial-samples 5 `
        --error-models gaussian --jitter-stds $row.jitter_stds --jitter-iterations 0 `
        --seeds 7,8,9,10,11 --output-dir $out --parallel --n-jobs $WORKERS --resume *>> $log
    if ($LASTEXITCODE -ne 0) { "[FAIL] $name exited $LASTEXITCODE" | Tee-Object -FilePath $log -Append; continue }
    & $PYTHON scripts\evaluate_research_question.py --input-dir $out --output-dir "$out\evaluation" *>> $log
    if ($LASTEXITCODE -ne 0) { "[FAIL] eval $name" | Tee-Object -FilePath $log -Append; continue }
    New-Item -ItemType File $marker -Force | Out-Null
    "[done] $name $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
}
"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

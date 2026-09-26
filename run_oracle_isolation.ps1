# The oracle-isolation experiment: the fitted-oracle pipeline on synthetic
# archival datasets drawn from each analytic landscape, so the landscape is held
# fixed and only the oracle changes. scripts/oracle_isolation.py builds the data,
# selects each oracle by the companion arm's own grouped cross-validation, and
# writes output-oracle-iso/manifest.csv with the error grid scaled to the fitted
# oracle's sigma_f. This script runs the simulator on each and evaluates it.
#
# -Family gaussian_process or -Family mlp runs the same datasets against one
# forced smooth oracle family, without jitter augmentation, in
# output-oracle-iso-<family> (run `oracle_isolation.py select/calibrate
# --family <family>` first).

param([string]$Family = "", [int]$Workers = 8)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$PYTHON = if ($env:HITL_PYTHON) { $env:HITL_PYTHON } else { "python" }
if ($Family) {
    $root = "output-oracle-iso-$Family"
    $augmentation = "none"
} else {
    $root = "output-oracle-iso"
    $augmentation = "jitter"
}
$log = "run_oracle_isolation$(if ($Family) { "-$Family" }).log"
"=== START $(Get-Date -Format o) ===" | Tee-Object -FilePath $log

foreach ($row in (Import-Csv "$root\manifest.csv")) {
    $name = $row.dataset
    $out = "$root\runs\$name"
    $marker = Join-Path $out ".done"
    if (Test-Path $marker) { "[skip] $name" | Tee-Object -FilePath $log -Append; continue }
    New-Item -ItemType Directory -Force $out | Out-Null
    "[run] $name  stds=$($row.jitter_stds)  oracle=$($row.oracle_model)" | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\bo_sensor_error_simulation.py `
        --dataset-config "output-oracle-iso\configs\datasets-$name.json" --objective composite `
        --acq-list logei,qnei,ucb,ei,random,sobol `
        --oracle-model auto --oracle-selection-path "$root\selection\$name.json" `
        --oracle-augmentation $augmentation `
        --iterations 50 --initial-samples 5 `
        --error-models gaussian --jitter-stds $row.jitter_stds --jitter-iterations 0 `
        --seeds 7,8,9,10,11 --output-dir $out --parallel --n-jobs $Workers --resume *>> $log
    if ($LASTEXITCODE -ne 0) { "[FAIL] $name exited $LASTEXITCODE" | Tee-Object -FilePath $log -Append; continue }
    & $PYTHON scripts\evaluate_research_question.py --input-dir $out --output-dir "$out\evaluation" *>> $log
    if ($LASTEXITCODE -ne 0) { "[FAIL] eval $name" | Tee-Object -FilePath $log -Append; continue }
    New-Item -ItemType File $marker -Force | Out-Null
    "[done] $name $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
}
"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

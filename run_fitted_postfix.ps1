# Rerun the fitted-oracle companion arms on opticarvis and provoice after the
# two archival-data fixes of 2026-09-23 (register R-D1, R-D2): opticarvis drops
# the 40 ratings logged on the raw instrument scales (column_ranges in
# datasets.json), provoice enters Predictability with a minus sign. ehmi is
# unchanged and keeps its runs.
#
# Prerequisites, in order: output\best_oracle_models.json from
# select_best_oracle_model.py on the fixed datasets.json, output\noise_anchor.csv
# from calibrate_noise_from_data.py + anchor_noise_scale.py, and
# output\per_dataset\manifest.json from make_per_dataset_configs.py.
#
# The pre-fix runs are moved to <arm>-prefix\<dataset> rather than deleted, so
# the before and after can be compared. Same design as the original arms
# (run_boba_gaps_rest.ps1 section F, run_boba_gaps2.ps1 section N,
# run_fitted_adapt.ps1).

param([int]$Jobs = 18, [string]$Arm = "", [string]$Dataset = "")
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$PYTHON = "python"
$log = if ($Arm) { "run_fitted_postfix-$Arm-$Dataset.log" } else { "run_fitted_postfix.log" }
$SEEDS10 = "7,8,9,10,11,12,13,14,15,16"
"=== START $(Get-Date -Format o) ===" | Tee-Object -FilePath $log

$entries = Get-Content "output\per_dataset\manifest.json" | ConvertFrom-Json
$ARMS = @(
    @{ Arm = "output-fitted";             Extra = @("--acq", "all") },
    @{ Arm = "output-fitted-noaug";       Extra = @("--acq", "all", "--oracle-augmentation", "none") },
    @{ Arm = "output-fitted-adapt-rep10"; Extra = @("--acq-list", "logei,qnei", "--replicate-first", "10") }
)
foreach ($a in $ARMS) {
    if ($Arm -and $a.Arm -ne $Arm) { continue }
    foreach ($name in @("opticarvis", "provoice")) {
        if ($Dataset -and $name -ne $Dataset) { continue }
        $e = $entries.$name
        $outdir = "$($a.Arm)\$name"
        $marker = "$outdir\.postfix-done"
        if (Test-Path $marker) { "[skip] $outdir" | Tee-Object -FilePath $log -Append; continue }
        $old = "$($a.Arm)-prefix\$name"
        if ((Test-Path $outdir) -and -not (Test-Path $old)) {
            New-Item -ItemType Directory -Force "$($a.Arm)-prefix" | Out-Null
            Move-Item $outdir $old
            "[moved] $outdir -> $old" | Tee-Object -FilePath $log -Append
        }
        New-Item -ItemType Directory -Force $outdir | Out-Null
        "[run] $outdir sigma_f=$($e.sigma_f) stds=$($e.jitter_stds) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
        & $PYTHON scripts\bo_sensor_error_simulation.py `
            --dataset-config $e.config --objective composite @($a.Extra) `
            --oracle-model auto --oracle-selection-path output\best_oracle_models.json `
            --iterations 50 --initial-samples 5 `
            --error-models gaussian --jitter-stds $e.jitter_stds --jitter-iterations 0,20 `
            --seeds $SEEDS10 --output-dir $outdir --parallel --n-jobs $Jobs --resume *>> $log
        if ($LASTEXITCODE -ne 0) { "[FAIL] $outdir exited $LASTEXITCODE" | Tee-Object -FilePath $log -Append; continue }
        & $PYTHON scripts\evaluate_research_question.py --input-dir $outdir --output-dir "$outdir\evaluation" *>> $log
        if ($LASTEXITCODE -ne 0) { "[FAIL] eval $outdir" | Tee-Object -FilePath $log -Append; continue }
        New-Item -ItemType File $marker -Force | Out-Null
        "[done] $outdir $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    }
}
"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

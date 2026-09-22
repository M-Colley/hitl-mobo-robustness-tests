# Calibration of the self-report arm (2026-09-22 review), run beside the held-out
# sweep to finish the same evening. The earlier model gave the rater a rank
# correlation of about 0.96 with their own realised squared error; these runs
# name the correlation instead, at 0, 0.3 and 0.6, through a gaussian copula that
# keeps the error's own marginal (--confidence-corr). Restricted to the two
# cells where the uncalibrated arm helped most, 0.25 and 1 sigma from the first
# rating, so the comparison with that arm is cell for cell.

param([string]$Only = "", [int]$Workers = 6)
$ErrorActionPreference = "Continue"
$PYTHON = "python"
$WORKERS = $Workers
$log = if ($Only) { "run_boba_review_calibration-$Only.log" } else { "run_boba_review_calibration.log" }

$RUNS = @(
    @{ Tag = "selfreport-r0.6"; Dir = "output-boba-idea-selfreport-r0.6"; Corr = "0.6" },
    @{ Tag = "selfreport-r0.3"; Dir = "output-boba-idea-selfreport-r0.3"; Corr = "0.3" },
    @{ Tag = "selfreport-r0";   Dir = "output-boba-idea-selfreport-r0";   Corr = "0" }
)

"=== START $(Get-Date -Format o) ===" | Tee-Object -FilePath $log
foreach ($r in $RUNS) {
    if ($Only -and $r.Tag -ne $Only) { continue }
    $marker = Join-Path $r.Dir ".done-$($r.Tag)"
    if (Test-Path $marker) { "[skip] $($r.Tag)" | Tee-Object -FilePath $log -Append; continue }
    New-Item -ItemType Directory -Force $r.Dir | Out-Null
    "[run] $($r.Tag) -> $($r.Dir)" | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\bo_synthetic_error_simulation.py `
        --functions all --acq-list logei,qnei --seeds 7,8,9,10,11 `
        --jitter-stds 0.25,1.0 --jitter-iterations 0 `
        --error-models gaussian --n-jobs $WORKERS `
        --output-dir $r.Dir --resume --observation-noise self_report --confidence-corr $r.Corr *>> $log
    if ($LASTEXITCODE -ne 0) { "[FAIL] $($r.Tag) exited $LASTEXITCODE" | Tee-Object -FilePath $log -Append; continue }
    foreach ($d in Get-ChildItem -Path $r.Dir -Directory) {
        if ($d.Name -in @("evaluation", "analysis")) { continue }
        & $PYTHON scripts\evaluate_research_question.py --input-dir $d.FullName `
            --output-dir (Join-Path $d.FullName "evaluation") *>> $log
        if ($LASTEXITCODE -ne 0) { "[FAIL] eval $($d.Name) in $($r.Dir)" | Tee-Object -FilePath $log -Append }
    }
    New-Item -ItemType File $marker -Force | Out-Null
    "[done] $($r.Tag) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
}
"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

# Runs requested by the 2026-09-22 review, one script owning the order.
#
#   held-out seeds 12-16 for the three remedies that were selected on seeds
#   7-11: the rank rule (spike and capped-scale arms) and the self-report arm.
#   Everything else the review asks to validate on held-out seeds already has
#   seeds 7-16 in the replay data and needs no new runs.
#
#   (the calibration of the self-report arm moved to run_boba_review_calibration.ps1)
#   calibration of the self-report arm: the earlier model gave the rater a
#   rank correlation of 0.96 with their own realised error. These runs name the
#   correlation instead, at 0, 0.3 and 0.6, through a gaussian copula that keeps
#   the error's own marginal (--confidence-corr).
#
# Ten workers, leaving room for the oracle-isolation sweep and the analyses.

$ErrorActionPreference = "Continue"
$PYTHON = "python"
$WORKERS = 8
$log = "run_boba_review_heldout.log"

$RUNS = @(
    @{ Tag = "spike-heldout";     Dir = "output-boba-spike";   Err = "spike";    Stds = "0.25";          Onsets = "0";    Seeds = "12,13,14,15,16";
       Flags = @("--error-spike-std-mode", "fixed", "--error-spike-prob", "0.15", "--error-spike-std", "20") },
    @{ Tag = "ceiling-heldout";   Dir = "output-boba-ceiling"; Err = "gaussian"; Stds = "0.25,1";        Onsets = "0";    Seeds = "12,13,14,15,16";
       Flags = @("--response-ceiling", "0.9", "--ceiling-mode", "fixed") },
    @{ Tag = "selfreport-heldout"; Dir = "output-boba-idea-selfreport"; Err = "gaussian"; Stds = "0.05,0.25,1.0,5.0"; Onsets = "0,20"; Seeds = "12,13,14,15,16";
       Flags = @("--observation-noise", "self_report", "--confidence-noise", "0.5") }
)

"=== START $(Get-Date -Format o) ===" | Tee-Object -FilePath $log

foreach ($r in $RUNS) {
    $marker = Join-Path $r.Dir ".done-$($r.Tag)"
    if (Test-Path $marker) { "[skip] $($r.Tag)" | Tee-Object -FilePath $log -Append; continue }
    New-Item -ItemType Directory -Force $r.Dir | Out-Null
    "[run] $($r.Tag) -> $($r.Dir)" | Tee-Object -FilePath $log -Append
    $acq = "logei,qnei"
    & $PYTHON scripts\bo_synthetic_error_simulation.py `
        --functions all --acq-list $acq --seeds $r.Seeds `
        --jitter-stds $r.Stds --jitter-iterations $r.Onsets `
        --error-models $r.Err --n-jobs $WORKERS `
        --output-dir $r.Dir --resume @($r.Flags) *>> $log
    if ($LASTEXITCODE -ne 0) { "[FAIL] $($r.Tag) exited $LASTEXITCODE" | Tee-Object -FilePath $log -Append; continue }

    # The driver writes one directory per landscape, so evaluate per landscape.
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

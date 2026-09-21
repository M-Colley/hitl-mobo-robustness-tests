# The five simulator arms of the 2026-09-21 idea round, in one place so that one
# script owns the ordering. Every arm keeps the number of human trials equal to
# the standard process or lowers it, and each is scored against the standard
# process, never against its own clean twin.
#
#   anchor      the proposal is judged beside the incumbent, so the error shared
#               by the pair cancels and the fresh part is differenced. Run under
#               bias and drift as well as gaussian, because a shared fault is
#               exactly what it should remove.
#   selfreport  the rater reports how sure they were and the GP takes it as a
#               per-trial observation variance.
#   anchors     every fifth trial rates a fixed anchor instead of a proposal;
#               the anchors identify the rater's drift, which is then removed.
#   hold        the first five model-based proposals are held back and rated
#               late, to decouple "informative design" from "early".
#   shiplcb     an acquisition that values a rating by what it does to the
#               cautious ship rule rather than to the posterior maximum.
#
# Sixteen workers, not more: thirty on twenty cores OOM-killed an arm once.

# NOT "Stop": Windows PowerShell wraps a native program's stderr in an
# ErrorRecord, and this driver writes its progress bar there, so "Stop" aborts
# a healthy run on its first line of output. $LASTEXITCODE is the real verdict.
$ErrorActionPreference = "Continue"
$PYTHON = "python"
$WORKERS = 16
$SEEDS = "7,8,9,10,11"
$STDS = "0.05,0.25,1.0,5.0"
$ONSETS = "0,20"
$log = "run_boba_hitl_ideas.log"

$ARMS = @(
    @{ Dir = "output-boba-idea-anchor";     Err = "gaussian,bias,drift"; Acq = "logei,qnei"; Flags = @("--anchor-rating") },
    @{ Dir = "output-boba-idea-selfreport"; Err = "gaussian";            Acq = "logei,qnei"; Flags = @("--observation-noise", "self_report", "--confidence-noise", "0.5") },
    @{ Dir = "output-boba-idea-anchors";    Err = "gaussian,drift";      Acq = "logei,qnei"; Flags = @("--anchor-every", "5", "--anchor-set", "3", "--anchor-model", "detrend") },
    @{ Dir = "output-boba-idea-hold";       Err = "gaussian";            Acq = "logei,qnei"; Flags = @("--hold-early", "5", "--hold-until", "0.6") },
    @{ Dir = "output-boba-idea-shiplcb";    Err = "gaussian";            Acq = "shiplcb";    Flags = @() }
)

"=== START $(Get-Date -Format o) ===" | Tee-Object -FilePath $log

foreach ($arm in $ARMS) {
    $dir = $arm.Dir
    if (Test-Path (Join-Path $dir "ARM_COMPLETE")) {
        "[skip] $dir already complete" | Tee-Object -FilePath $log -Append
        continue
    }
    "[arm] $dir  err=$($arm.Err)  acq=$($arm.Acq)  $($arm.Flags -join ' ')" | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\bo_synthetic_error_simulation.py `
        --functions all --acq-list $arm.Acq --seeds $SEEDS `
        --jitter-stds $STDS --jitter-iterations $ONSETS `
        --error-models $arm.Err --n-jobs $WORKERS `
        --output-dir $dir --resume @($arm.Flags) *>> $log
    if ($LASTEXITCODE -ne 0) { "[FAIL] $dir exited $LASTEXITCODE" | Tee-Object -FilePath $log -Append; continue }

    # The driver writes one directory per landscape, so the evaluator runs per
    # landscape too; pointed at the arm root it finds no per-iteration logs.
    "[eval] $dir" | Tee-Object -FilePath $log -Append
    foreach ($d in Get-ChildItem -Path $dir -Directory) {
        if ($d.Name -in @("evaluation", "analysis")) { continue }
        & $PYTHON scripts\evaluate_research_question.py --input-dir $d.FullName `
            --output-dir (Join-Path $d.FullName "evaluation") *>> $log
        if ($LASTEXITCODE -ne 0) { "[FAIL] eval $($d.Name) in $dir" | Tee-Object -FilePath $log -Append }
    }
    New-Item -ItemType File (Join-Path $dir "ARM_COMPLETE") -Force | Out-Null
}

"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

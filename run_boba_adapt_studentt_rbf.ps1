# Register item B2: the published Student-t arm (output-boba-adapt-studentt)
# changed the kernel to Matern-5/2 as well as the likelihood. This reruns the
# same design with the RBF kernel of the standard GP (--likelihood
# student_t_rbf), so that the likelihood is the only thing that differs.

param([int]$Jobs = 14)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$PYTHON = "python"
$dir = "output-boba-adapt-studentt-rbf"
$log = "run_boba_adapt_studentt_rbf.log"
"=== START $(Get-Date -Format o) ===" | Tee-Object -FilePath $log
& $PYTHON scripts\bo_synthetic_error_simulation.py `
    --functions all --acq-list logei,qnei `
    --iterations 50 --initial-samples 5 `
    --error-models gaussian --jitter-stds 0.01,0.05,0.15,0.4 --jitter-iterations 0,20 `
    --seeds 7,8,9,10,11 --output-dir $dir --n-jobs $Jobs --resume `
    --input-error misclick --input-error-from-sweep --likelihood student_t_rbf *>> $log
if ($LASTEXITCODE -ne 0) { "[FAIL] sweep exited $LASTEXITCODE" | Tee-Object -FilePath $log -Append; exit 1 }
foreach ($d in Get-ChildItem -Path $dir -Directory) {
    if ($d.Name -in @("evaluation", "analysis")) { continue }
    & $PYTHON scripts\evaluate_research_question.py --input-dir $d.FullName --output-dir (Join-Path $d.FullName "evaluation") *>> $log
    if ($LASTEXITCODE -ne 0) { "[FAIL] eval $($d.Name)" | Tee-Object -FilePath $log -Append }
}
"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

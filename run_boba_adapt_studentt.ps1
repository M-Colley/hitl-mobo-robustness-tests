# E2: the Student-t surrogate under misclicks, then its evaluation and analysis.
#
# run_boba_adapt.ps1 skipped this arm in the first queue because the driver had
# no --likelihood yet. This runs it, evaluates only its own directory, and
# writes output-boba\adapt-studentt.DONE.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_adapt_studentt.ps1 -Jobs 20

param([int]$Jobs = 20)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$log = "output-boba\adapt-studentt-finish.log"
"=== studentt arm started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
"$PID" | Set-Content -Path "output-boba\adapt-studentt.PID" -Encoding ascii
$PYTHON = if ($env:PYTHON) { $env:PYTHON } else { "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" }
$dir = "output-boba-adapt-studentt"

& powershell -NoProfile -ExecutionPolicy Bypass -File run_boba_adapt.ps1 -Arm studentt -Jobs $Jobs
if (-not (Test-Path "$dir\SWEEP_COMPLETE")) { "FAILED: sweep did not complete" | Tee-Object -FilePath $log -Append; exit 1 }

foreach ($d in Get-ChildItem -Path $dir -Directory) {
    if ($d.Name -in @("evaluation", "analysis")) { continue }
    & $PYTHON scripts\evaluate_research_question.py --input-dir $d.FullName --output-dir (Join-Path $d.FullName "evaluation") *>> $log
    if ($LASTEXITCODE -ne 0) { "FAILED: eval $($d.Name)" | Tee-Object -FilePath $log -Append; exit 1 }
}
& $PYTHON scripts\analyse_boba_robustness.py --input-dir $dir --output-dir "$dir\analysis" *>> $log
if ($LASTEXITCODE -ne 0) { "FAILED: analyse_boba_robustness" | Tee-Object -FilePath $log -Append; exit 1 }
"done $(Get-Date -Format o)" | Set-Content -Path "output-boba\adapt-studentt.DONE" -Encoding ascii
"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

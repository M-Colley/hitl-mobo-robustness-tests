# Runs the adaptation arms three at a time, then evaluates and analyses them.
#
#   batch 1: rep10, rep10-obs, rerate          (gaussian error)
#   batch 2: rerate-slip, nigp, studentt        (input error; studentt only if wired)
#
# Each arm is its own run_boba_adapt.ps1 process (resume-safe), so a crash in
# one does not take the others down. Writes output-boba\adapt-queue.DONE at the
# end; the PID is in output-boba\adapt-queue.PID.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_adapt_queue.ps1

param([int]$Jobs = 8)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$log = "output-boba\adapt-queue.log"
"=== adaptation queue started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
"$PID" | Set-Content -Path "output-boba\adapt-queue.PID" -Encoding ascii

Add-Type @'
using System; using System.Runtime.InteropServices;
public static class PowerQueue { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags); }
'@
[void][PowerQueue]::SetThreadExecutionState([uint32]"0x80000001")

$PYTHON = if ($env:PYTHON) { $env:PYTHON } else { "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" }
$DIRS = @{
    "rep10" = "output-boba-adapt-rep10"; "rep10-obs" = "output-boba-adapt-rep10-obs"; "rerate" = "output-boba-adapt-rerate"
    "rerate-slip" = "output-boba-adapt-rerate-slip"; "nigp" = "output-boba-adapt-nigp"; "studentt" = "output-boba-adapt-studentt"
}
function Say($m) { "$(Get-Date -Format o)  $m" | Tee-Object -FilePath $log -Append }

function Run-Batch($names) {
    $procs = @()
    foreach ($n in $names) {
        if (Test-Path "$($DIRS[$n])\SWEEP_COMPLETE") { Say "$n already complete"; continue }
        Say "starting $n"
        $procs += Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','run_boba_adapt.ps1','-Arm',$n,'-Jobs',"$Jobs" -WindowStyle Hidden -PassThru
    }
    if ($procs.Count -gt 0) { $procs | Wait-Process }
    foreach ($n in $names) {
        if (Test-Path "$($DIRS[$n])\SWEEP_COMPLETE") { Say "$n complete" }
        else { Say "WARNING: $n has no SWEEP_COMPLETE (studentt is skipped by design if the driver lacks --likelihood)" }
    }
}

Run-Batch @("rep10", "rep10-obs", "rerate")
Run-Batch @("rerate-slip", "nigp", "studentt")

Say "evaluating"
& powershell -NoProfile -ExecutionPolicy Bypass -File run_boba_adapt.ps1 -Arm eval
if ($LASTEXITCODE -ne 0) { Say "FAILED: evaluation"; exit 1 }

foreach ($n in $DIRS.Keys) {
    $d = $DIRS[$n]
    if (-not (Test-Path "$d\SWEEP_COMPLETE")) { continue }
    Say "analysis: $d"
    & $PYTHON scripts\analyse_boba_robustness.py --input-dir $d --output-dir "$d\analysis" *>> $log
    if ($LASTEXITCODE -ne 0) { Say "FAILED: analyse_boba_robustness $d"; exit 1 }
    & $PYTHON scripts\analyse_extra_runs.py --input-dir $d --k 10,25 --tolerance 0,0.01 *>> $log
    if ($LASTEXITCODE -ne 0) { Say "FAILED: analyse_extra_runs $d"; exit 1 }
}
if (Test-Path "scripts\analyse_boba_adaptations.py") {
    Say "adaptation analysis"
    & $PYTHON scripts\analyse_boba_adaptations.py *>> $log
    if ($LASTEXITCODE -ne 0) { Say "FAILED: analyse_boba_adaptations"; exit 1 }
}
"done $(Get-Date -Format o)" | Set-Content -Path "output-boba\adapt-queue.DONE" -Encoding ascii
Say "=== DONE ==="
[void][PowerQueue]::SetThreadExecutionState([uint32]"0x80000000")

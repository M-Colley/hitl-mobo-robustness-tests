# Finishes the input-error arm without anyone having to be watching.
#
# Two jobs, both timing-dependent and easy to miss by hand:
#
#   1. REBALANCE. The three variants run as three processes of 6 workers. slip
#      finishes first -- it resumed past its slowest tasks -- and its 6 cores
#      would then sit idle while misclick and slip-actual run on at 6 each. When
#      slip's SWEEP_COMPLETE appears, each of the other two is restarted at 9.
#      A restart costs only the runs in flight: --resume keeps every finished
#      CSV, and the worker count cannot change a result.
#
#   2. FINISH. When all three markers exist: evaluate every benchmark
#      directory, run the cross-benchmark analysis on each arm, run the
#      input-error analysis, regenerate the paper's tables, and only then write
#      output-boba\inputerror-finish.DONE. The status script's ALLDONE waits for
#      that marker, so the notification arrives with the numbers ready rather
#      than with an hour of processing still to go.
#
# The script records its own PID so the status script can tell "working" from
# "died" by process ID, not by matching command lines -- a pattern match can
# match the very process doing the matching, which has bitten this project.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_inputerror_finish.ps1

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
if (-not (Test-Path "output-boba")) { New-Item -ItemType Directory "output-boba" | Out-Null }

$log = "output-boba\inputerror-finish.log"
$marker = "output-boba\inputerror-finish.DONE"
"=== input-error finisher (re)started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
"$PID" | Set-Content -Path "output-boba\inputerror-finish.PID" -Encoding ascii

Add-Type @'
using System; using System.Runtime.InteropServices;
public static class Power { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags); }
'@
[void][Power]::SetThreadExecutionState([uint32]"0x80000001")

$PYTHON = if ($env:PYTHON) { $env:PYTHON } else {
    $c = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    & $c -c "import torch, botorch" 2>$null
    if ($LASTEXITCODE -eq 0) { $c } else { "python" }
}

$ARMS = [ordered]@{
    "slip"        = "output-boba-slip"
    "misclick"    = "output-boba-misclick"
    "slip-actual" = "output-boba-slip-actual"
}

function Say($m) { "$(Get-Date -Format o)  $m" | Tee-Object -FilePath $log -Append }

function Fail($stage) {
    "FAILED: $stage $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    [void][Power]::SetThreadExecutionState([uint32]"0x80000000")
    exit 1
}

function Complete($name) { Test-Path "$($ARMS[$name])\SWEEP_COMPLETE" }

# The runner for one arm, matched on "-File <script> -Arm <name>" followed by a
# space or the end, so "slip" cannot match "slip-actual" and a -Command wrapper
# that merely mentions the script cannot match at all.
function Arm-Runner($name) {
    $pattern = "-File\s+\S*run_boba_inputerror\.ps1\s+-Arm\s+" + [regex]::Escape($name) + "(\s|$)"
    @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object { $_.CommandLine -match $pattern })
}

# An arm with no runner and no marker has died. Allow a moment first: a runner
# can be between its last write and its exit.
function Assert-Alive($name) {
    if ((Complete $name) -or (Arm-Runner $name).Count -gt 0) { return }
    Start-Sleep -Seconds 30
    if (-not (Complete $name) -and (Arm-Runner $name).Count -eq 0) {
        Fail "$name has no runner and no SWEEP_COMPLETE"
    }
}

# --- 1. rebalance when slip finishes -----------------------------------------
if (-not (Complete "slip")) {
    Say "waiting for slip to finish before rebalancing ..."
    while (-not (Complete "slip")) {
        Assert-Alive "slip"
        Start-Sleep -Seconds 60
    }
}
Say "slip is complete."
foreach ($name in @("misclick", "slip-actual")) {
    if (Complete $name) { Say "$name already complete; nothing to rebalance."; continue }
    $r = Arm-Runner $name
    if ($r.Count -ne 1) {
        Say "WARNING: expected one $name runner, found $($r.Count); leaving it as it is."
        continue
    }
    if ($r[0].CommandLine -match "-Jobs\s+9") { Say "$name already runs 9 workers."; continue }
    Say "restarting $name at 9 workers (in-flight runs are redone; --resume keeps the rest)"
    taskkill /PID $r[0].ProcessId /T /F | Out-Null
    Start-Sleep -Seconds 8
    Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','run_boba_inputerror.ps1','-Arm',$name,'-Jobs','9' -WindowStyle Hidden
}
Start-Sleep -Seconds 60

# --- 2. wait for everything, then finish ------------------------------------
Say "waiting for misclick and slip-actual ..."
while (-not ((Complete "misclick") -and (Complete "slip-actual"))) {
    Assert-Alive "misclick"
    Assert-Alive "slip-actual"
    Start-Sleep -Seconds 120
}
Say "all three input-error arms are complete."

Say "evaluating every benchmark directory"
& powershell -NoProfile -ExecutionPolicy Bypass -File run_boba_inputerror.ps1 -Arm eval
if ($LASTEXITCODE -ne 0) { Fail "evaluation (see output-boba\inputerror-eval.log)" }

foreach ($name in $ARMS.Keys) {
    $dir = $ARMS[$name]
    Say "cross-benchmark analysis: $dir"
    & $PYTHON scripts\analyse_boba_robustness.py --input-dir $dir --output-dir "$dir\analysis" *>> $log
    if ($LASTEXITCODE -ne 0) { Fail "analyse_boba_robustness $dir" }
}

Say "input-error analysis"
& $PYTHON scripts\analyse_boba_inputerror.py *>> $log
if ($LASTEXITCODE -ne 0) { Fail "analyse_boba_inputerror" }

Say "regenerating the paper's tables"
& $PYTHON scripts\make_boba_paper_tables.py *>> $log
if ($LASTEXITCODE -ne 0) { Fail "make_boba_paper_tables" }

"done $(Get-Date -Format o)" | Set-Content -Path $marker -Encoding ascii
Say "=== DONE -- marker written ==="
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")

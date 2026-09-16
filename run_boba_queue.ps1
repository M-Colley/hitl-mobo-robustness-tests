# Sequences the remaining arms so the machine is never oversubscribed.
#
# 2026-09-09: the multi-objective arm was killed by the OS mid-run because three
# runners were started independently and together asked for 30 workers on a 20
# core / 64 GB machine. Nothing decides the total worker count when each runner
# picks its own, so this script owns the ordering instead.
#
#   already running   run_boba_gaps_rest.ps1  (10 workers) budget100 -> confirmatory -> fitted
#   already running   run_boba_gaps_mo.ps1    ( 8 workers) multi-objective
#   stage 1           run_boba_gaps2.ps1      (10 workers) instrument -> robust -> no-aug
#                     starts when the MO arm exits, taking over its workers
#   stage 2           run_chi_qnehvi_repair.ps1 (8 workers) after everything above
#                     scoped to ehmi on 2026-09-09: 83 runs, ~4 h. All three
#                     datasets would have been 2,683 runs and ~140 h.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_queue.ps1

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$log = "output-boba\queue.log"
function Say($m) { "$(Get-Date -Format o)  $m" | Tee-Object -FilePath $log -Append }
Say "=== queue started ==="

Add-Type @'
using System; using System.Runtime.InteropServices;
public static class Power { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags); }
'@
[void][Power]::SetThreadExecutionState([uint32]"0x80000001")

# Match only "-File <script>", never a bare -Command whose TEXT happens to name
# the script. An ad-hoc `powershell -Command "... run_boba_queue ..."` otherwise
# matches itself, and a Stop-Process over that set kills the caller mid-script.
function Runner-Alive($script) {
    $me = $PID
    @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object { $_.ProcessId -ne $me -and $_.CommandLine -match "-File\s+\S*$script" }).Count -gt 0
}

function Wait-Runner($pattern, $label) {
    if (-not (Runner-Alive $pattern)) { Say "$label is not running; continuing."; return }
    Say "waiting for $label ..."
    while (Runner-Alive $pattern) { Start-Sleep -Seconds 120 }
    Say "$label has exited."
}

# --- stage 1: the last three synthetic arms, on the workers the MO arm frees ---
Wait-Runner 'run_boba_gaps_mo' 'the multi-objective arm'
$mo = @(Get-ChildItem -Path "output-boba-mo" -Recurse -Filter *.csv -ErrorAction SilentlyContinue).Count
Say "multi-objective arm finished at $mo/1890 runs."
if ($mo -lt 1890) { Say "WARNING: the MO arm is short of its target; check output-boba\gaps-mo.log." }

Say "starting run_boba_gaps2.ps1 (instrument, robust, no-aug)"
& powershell -NoProfile -ExecutionPolicy Bypass -File run_boba_gaps2.ps1
Say "run_boba_gaps2.ps1 exited $LASTEXITCODE"

# --- stage 2: the CHI repair, alone ------------------------------------------
Wait-Runner 'run_boba_gaps_rest' 'the confirmatory/fitted arm'
Say "starting run_chi_qnehvi_repair.ps1 [ehmi only] -- 83 runs at a measured 19.1 runs/h, ~4 h"
& powershell -NoProfile -ExecutionPolicy Bypass -File run_chi_qnehvi_repair.ps1
Say "run_chi_qnehvi_repair.ps1 exited $LASTEXITCODE"

Say "=== queue done ==="
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")

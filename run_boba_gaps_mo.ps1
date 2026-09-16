# M. MULTI-OBJECTIVE arm of the known-function study.
#
# The applied setting is multi-objective and the BOBA suite is scalar. Seven
# BoTorch problems with a PUBLISHED maximum hypervolume, standardised per
# objective; the standardised optimum follows in closed form because
# hypervolume scales by prod(sigma) under the per-objective affine map.
#
# WORKER COUNT IS A MEMORY DECISION, NOT A SPEED ONE.
# 2026-09-09: this arm died at 1,710/6,930 with BrokenProcessPool on seven
# in-flight tasks -- five carsideimpact, one penicillin, one vehiclesafety. No
# MemoryError and no Windows Application Error event, which is what an OS-level
# kill looks like: the worker vanishes and the parent only sees the pipe close.
# It was running 12 workers while the qNEHVI repair (8) and the budget arm (10)
# were also running -- 30 workers on a 64 GB machine, with the hypervolume box
# decomposition on a four-objective problem as the memory-heavy part.
#
# So: 8 workers, and a retry ladder. Every run is fully determined by its seed
# and configuration and --resume validates each CSV's completeness, so a retry
# costs only the runs that were in flight. Dropping the worker count on retry is
# free scientifically -- distribution across workers cannot change a result.
#
# COST, measured rather than guessed (fit_time_sec summed per run, calibrated
# against wall x workers): 253 core-hours remain. carsideimpact is 504 s/run and
# 53% of the total. Parallelism is per (function, seed), so the five
# carsideimpact seed-tasks are the floor: ~24 h wall however many workers run.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_gaps_mo.ps1
#
# Resumable.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$log = "output-boba\gaps-mo.log"
if (-not (Test-Path "output-boba")) { New-Item -ItemType Directory "output-boba" | Out-Null }
"=== BOBA gap arm [multi-objective] (re)started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

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
"Interpreter: $PYTHON" | Tee-Object -FilePath $log -Append

$SEEDS5 = "7,8,9,10,11"
$STDS   = "0.05,0.25,1.0,5.0"

$common = @(
    "scripts\bo_synthetic_error_simulation.py", "--output-dir", "output-boba-mo",
    "--multi-objective", "--functions", "all", "--acq", "all",
    "--iterations", "50", "--initial-samples", "5", "--acq-mc-samples", "64",
    "--error-models", "gaussian", "--jitter-stds", $STDS,
    "--jitter-iterations", "0,20", "--seeds", $SEEDS5, "--resume"
)

if (Test-Path "output-boba-mo\SWEEP_COMPLETE") {
    "[M multi-objective] already complete." | Tee-Object -FilePath $log -Append
} else {
    $ok = $false
    foreach ($jobs in @(8, 5, 3)) {
        "[M multi-objective] starting with $jobs workers $(Get-Date -Format o)" |
            Tee-Object -FilePath $log -Append
        & $PYTHON @common --n-jobs $jobs *>> $log
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        "[M multi-objective] exit $LASTEXITCODE at $jobs workers; retrying lower." |
            Tee-Object -FilePath $log -Append
        Start-Sleep -Seconds 30
    }
    if (-not $ok) {
        "FAILED: M multi-objective (exhausted retry ladder) $(Get-Date -Format o)" |
            Tee-Object -FilePath $log -Append
        [void][Power]::SetThreadExecutionState([uint32]"0x80000000")
        exit 1
    }
}

"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")

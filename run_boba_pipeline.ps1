# Known-function robustness pipeline: the BOBA benchmark suite (resumable).
#
# Why this exists
# ---------------
# The data-driven sweeps (run_full_pipeline.ps1, run_chi_pipeline.ps1) measure
# the cost of noisy human feedback against a REGRESSION ORACLE fitted to
# archival ratings. That oracle's cross-validated R^2 is 0.55 at best, so every
# result is entangled with how badly the surrogate human is mis-specified, and
# the reference optimum it is scored against is only a random-search estimate.
#
# This pipeline replaces the fitted human with the 20 analytic benchmarks of the
# BOBA suite. The objective is exact, the optimum is a verified supremum, and the
# landscape geometry is measured up front (boba_landscape_stats.json) so it can
# be used as a predictor of which landscapes noise actually damages.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_pipeline.ps1
#
# Design (stage 1, ~20 h on 24 workers of this 20-core/28-thread machine):
#   20 benchmarks x 12 acquisitions x 10 seeds
#     x [ 1 noise-free baseline
#         + 4 error models {gaussian, bias, drift, ar1}
#           x 4 magnitudes {0.05, 0.25, 1.0, 5.0} landscape SDs
#           x 2 onsets {0, 20} ]
#   = 79,200 runs of 50 BO iterations.
#
# Magnitudes are in units of each landscape's own standard deviation, so one
# level means the same thing on hartmann_6 (raw range ~3) and powell (~6e4). The
# endpoints 0.05 and 5.0 match the data-driven sweep's grid once its 1-7 rating
# scale is expressed in SDs.
#
# Onset 0 is the human-plausible "noisy from the first rating" condition; onset
# 20 lets BO converge first, which separates "noise while exploring" from "noise
# while exploiting".
#
# 'bias' uses --error-bias-mode scaled, so the systematic offset equals the swept
# magnitude. With the data-driven arm's fixed 0.2 offset the bias condition
# becomes indistinguishable from plain gaussian at the top of the sweep, which
# makes the systematic-vs-random contrast untestable.
#
# Stage 2 (dropout + spike) is a separate, optional sweep into its own directory;
# see the bottom of this file.
#
# Everything is resumable: re-run after any interruption and completed per-run
# CSVs are reused. On success it writes output-boba\SWEEP_COMPLETE.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$OUT = "output-boba"
if (-not (Test-Path $OUT)) { New-Item -ItemType Directory $OUT | Out-Null }
$log = "$OUT\pipeline.log"

"=== BOBA known-function pipeline (re)started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

# --- Keep the machine awake for the duration (released when the script ends) ---
Add-Type @'
using System; using System.Runtime.InteropServices;
public static class Power { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags); }
'@
[void][Power]::SetThreadExecutionState([uint32]"0x80000001")  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

# Bare "python" on this machine is 3.11 and has no torch; resolve an
# interpreter that can actually import the stack before spending 17 hours.
$PYTHON = if ($env:PYTHON) { $env:PYTHON } else {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "python"
    )
    $found = $null
    foreach ($c in $candidates) {
        & $c -c "import torch, botorch" 2>$null
        if ($LASTEXITCODE -eq 0) { $found = $c; break }
    }
    if (-not $found) {
        "No Python interpreter with torch+botorch found. Set `$env:PYTHON." | Tee-Object -FilePath $log -Append
        exit 1
    }
    $found
}
"Interpreter: $PYTHON" | Tee-Object -FilePath $log -Append
# Ten screening seeds. Unlike the data-driven arm -- where seeds are the only
# replication and 5 of them cannot reach p<0.05 on a Wilcoxon -- the primary
# tests here block on the twenty BENCHMARKS, so seeds only sharpen each cell
# mean. Ten keeps the sweep inside a day on this machine; extend to the usual
# twenty at any time by widening $SEEDS and re-running (completed runs are
# reused). Seeds 27+ stay reserved for the confirmatory follow-up.
$SEEDS  = "7,8,9,10,11,12,13,14,15,16"
$ERRORS = "gaussian,bias,drift,ar1"
$STDS   = "0.05,0.25,1.0,5.0"
$JITS   = "0,20"
$JOBS   = 24

function Fail($stage) {
    "FAILED: $stage (exit $LASTEXITCODE) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    [void][Power]::SetThreadExecutionState([uint32]"0x80000000")  # release
    exit 1
}

# --- 1. Landscape statistics ---------------------------------------------------
# The standardisation constants and the descriptor table. Must exist before the
# sweep: they define the objective scale every error magnitude is relative to.
if (-not (Test-Path "boba_landscape_stats.json")) {
    "[1/4] Computing landscape statistics..." | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\boba_benchmarks.py --output boba_landscape_stats.json *>> $log
    if ($LASTEXITCODE -ne 0) { Fail "landscape statistics" }
} else {
    "[1/4] Landscape statistics present." | Tee-Object -FilePath $log -Append
}

# --- 2. Correctness gate -------------------------------------------------------
# The vendored functions ARE the ground truth here, so the parity and
# known-optimum tests run before any compute is spent.
"[2/4] Correctness gate (benchmark parity, known optima, driver schema)..." | Tee-Object -FilePath $log -Append
& $PYTHON -m pytest tests\test_boba_benchmarks.py -q *>> $log
if ($LASTEXITCODE -ne 0) { Fail "correctness gate" }

# --- 3. The sweep --------------------------------------------------------------
if (Test-Path "$OUT\SWEEP_COMPLETE") {
    "[3/4] SWEEP_COMPLETE marker present; skipping the sweep." | Tee-Object -FilePath $log -Append
} else {
    "[3/4] Sweep (resume)... $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\bo_synthetic_error_simulation.py `
        --functions all --acq all `
        --iterations 50 --initial-samples 5 `
        --error-models $ERRORS --jitter-stds $STDS --jitter-iterations $JITS `
        --error-bias-mode scaled `
        --seeds $SEEDS --output-dir $OUT --n-jobs $JOBS --resume *>> $log
    if ($LASTEXITCODE -ne 0) { Fail "sweep" }
}

# --- 4. Evaluation -------------------------------------------------------------
# Per benchmark, so no single process has to hold 8M log rows, and so the 20
# evaluations run concurrently.
"[4/4] Per-benchmark evaluation... $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
$dirs = Get-ChildItem -Path $OUT -Directory | Where-Object { Test-Path "$($_.FullName)\*.csv" }
$jobs = @()
foreach ($d in $dirs) {
    $jobs += Start-Job -ScriptBlock {
        param($py, $root, $name)
        Set-Location $root
        & $py scripts\evaluate_research_question.py `
            --input-dir "output-boba\$name" --output-dir "output-boba\$name\evaluation" 2>&1
        $LASTEXITCODE
    } -ArgumentList $PYTHON, $PSScriptRoot, $d.Name
    while ((Get-Job -State Running).Count -ge 6) { Start-Sleep -Seconds 5 }
}
$jobs | Wait-Job | Out-Null
$jobs | ForEach-Object { Receive-Job $_ *>> $log }
$jobs | Remove-Job

"[4/4] GP noise diagnostic..." | Tee-Object -FilePath $log -Append
& $PYTHON scripts\diagnose_gp_noise.py --input-dir $OUT --per-cell 3 --at-iterations 8,15,25,50 *>> $log

"[4/4] Cross-benchmark synthesis..." | Tee-Object -FilePath $log -Append
& $PYTHON scripts\analyse_boba_robustness.py --input-dir $OUT --output-dir "$OUT\analysis" *>> $log
if ($LASTEXITCODE -ne 0) { Fail "synthesis" }

"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")  # release

# --- Stage 2 (optional): the two purpose-built extension families --------------
# Neither is part of BOBA. Both exist to break a confound the real suite cannot:
#   levy_4d / levy_7d  -- the same Levy function at d=4 and d=7, which together
#     with levy_10 (d=11) turns dimension from a single-point covariate into a
#     three-point manipulation inside one landscape family.
#   bump_a{4,16}_w{0.05,0.15} -- a cos-field background plus one narrow spike.
#     Amplitude moves opt_z ~4x at fixed width; width moves sparsity at roughly
#     fixed opt_z. Across the real suite those two are correlated at ~0.9 and
#     cannot be separated, so any claim that one of them predicts fragility is
#     untestable there. 6 benchmarks x the stage-1 grid = 23,760 runs, ~6 h.
#
#   python scriptso_synthetic_error_simulation.py --functions extensions --acq all `
#     --error-models gaussian,bias,drift,ar1 --jitter-stds 0.05,0.25,1.0,5.0 `
#     --jitter-iterations 0,20 --error-bias-mode scaled --seeds $SEEDS `
#     --output-dir output-boba-extensions --n-jobs 24 --resume

# --- Stage 3 (optional): the remaining two error models ------------------------
# dropout ("the rater did not respond; hold the last value") and spike ("an
# occasional gross misrating") complete the taxonomy. Run into a separate
# directory so stage 1's evaluation is not re-run:
#
#   python scripts\bo_synthetic_error_simulation.py --functions all --acq all `
#     --error-models dropout,spike --jitter-stds 0.05,0.25,1.0,5.0 `
#     --jitter-iterations 0,20 --seeds $SEEDS --output-dir output-boba-stage3 `
#     --n-jobs 24 --resume

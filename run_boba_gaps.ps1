# The five remaining gaps in the known-function study.
#
# Each was named as a limitation in the paper draft; each is closed by its own
# arm here, scoped to the smallest design that answers its own question.
#
#   F. FITTED-ORACLE COMPANION (~2 h)
#      The synthetic arm's whole premise is that a fitted human oracle confounds
#      the measurement. That is an argument, not a result, until the two arms
#      are compared on a matched design. Runs the data-driven simulator on the
#      three archival datasets at the SAME error grid, expressed per dataset in
#      its own sigma_f so a "1 sigma" error means the same thing in both arms.
#      Gaussian only: the synthetic arm showed the error process barely matters,
#      and the data-driven driver's bias offset does not scale with the sweep,
#      so gaussian is the only cleanly matched contrast.
#
#   M. MULTI-OBJECTIVE (~4 h)
#      The applied setting is multi-objective and the BOBA suite is scalar.
#      Seven BoTorch problems with a PUBLISHED maximum hypervolume, standardised
#      per objective; the standardised optimum follows in closed form because
#      hypervolume scales by prod(sigma) under the per-objective affine map.
#      Fewer seeds and reduced MC sampling: the hypervolume box decomposition is
#      the memory-heavy part, and carsideimpact has four objectives.
#
#   B. BUDGET SENSITIVITY (~5 h)
#      The onset effect is about WHERE in the budget the error falls, so a fixed
#      50-iteration budget is not a neutral choice. Re-runs at 25 and 100
#      iterations with the onset held at the same FRACTION of the budget, which
#      is the comparison that isolates budget from onset.
#
#   C. CONFIRMATORY SEEDS (~6 h)
#      Everything reported so far was selected and estimated on seeds 7-16.
#      Re-runs the headline design on ten fresh seeds that had no part in
#      generating any hypothesis. Read HYPOTHESIS.md before looking at it.
#
#   L. MATCHED DIMENSION LADDER (~1 h)
#      The Levy ladder confounds dimension with signal strength and cannot be
#      fixed -- opt_z is a property of the function at each dimension and is
#      invariant to rescaling the box. bump_d4/d7/d11 hold opt_z at 9.0 and the
#      spike's volume fraction at 1e-6, leaving dimension as the only difference.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_gaps.ps1
#
# Resumable. Each arm writes its own directory.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$log = "output-boba\gaps.log"
if (-not (Test-Path "output-boba")) { New-Item -ItemType Directory "output-boba" | Out-Null }
"=== BOBA gap-closing arms (re)started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

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

$SEEDS5  = "7,8,9,10,11"
$SEEDS10 = "7,8,9,10,11,12,13,14,15,16"
$FRESH   = "27,28,29,30,31,32,33,34,35,36"
$STDS    = "0.05,0.25,1.0,5.0"
$JOBS    = 24

function Fail($stage) {
    "FAILED: $stage (exit $LASTEXITCODE) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    [void][Power]::SetThreadExecutionState([uint32]"0x80000000")
    exit 1
}

function Synth($name, $outdir, $extra) {
    if (Test-Path "$outdir\SWEEP_COMPLETE") {
        "[$name] already complete." | Tee-Object -FilePath $log -Append
        return
    }
    "[$name] starting $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    # Build the argument array first. Inlining the concatenation into the
    # call makes PowerShell pass the "+" through as a literal argument.
    $argv = @("scripts\bo_synthetic_error_simulation.py", "--output-dir", $outdir,
              "--n-jobs", $JOBS, "--resume") + $extra
    & $PYTHON @argv *>> $log
    if ($LASTEXITCODE -ne 0) { Fail $name }
}

# --- L. matched dimension ladder (cheapest; run first so a failure is quick) ---
Synth "L ladder" "output-boba-ladder" @(
    "--functions", "bump_d4,bump_d7,bump_d11", "--acq", "all",
    "--iterations", "50", "--initial-samples", "5",
    "--error-models", "gaussian", "--jitter-stds", $STDS,
    "--jitter-iterations", "0,20", "--error-bias-mode", "scaled",
    "--seeds", $SEEDS10
)

# --- M. multi-objective --------------------------------------------------------
# 12 workers and reduced MC sampling: the hypervolume box decomposition is what
# uses the memory, and this machine has run out on that before.
Synth "M multi-objective" "output-boba-mo" @(
    "--multi-objective", "--functions", "all", "--acq", "all",
    "--iterations", "50", "--initial-samples", "5", "--acq-mc-samples", "64",
    "--error-models", "gaussian", "--jitter-stds", $STDS,
    "--jitter-iterations", "0,20", "--seeds", $SEEDS5, "--n-jobs", "12"
)

# --- B. budget sensitivity -----------------------------------------------------
# The onset is held at the same FRACTION of the budget (0 and 40%), so what
# varies is the budget and not where in it the error lands.
Synth "B budget 25" "output-boba-budget25" @(
    "--functions", "all", "--acq-list", "logei,ei,pi,ucb,qucb,qnei",
    "--iterations", "25", "--initial-samples", "5",
    "--error-models", "gaussian", "--jitter-stds", $STDS,
    "--jitter-iterations", "0,10", "--seeds", $SEEDS5
)
Synth "B budget 100" "output-boba-budget100" @(
    "--functions", "all", "--acq-list", "logei,ei,pi,ucb,qucb,qnei",
    "--iterations", "100", "--initial-samples", "5",
    "--error-models", "gaussian", "--jitter-stds", $STDS,
    "--jitter-iterations", "0,40", "--seeds", $SEEDS5
)

# --- C. confirmatory seeds -----------------------------------------------------
if (-not (Test-Path "output-boba-confirmatory\HYPOTHESIS.md")) {
    "[C] refusing to run: write output-boba-confirmatory\HYPOTHESIS.md first, so the" |
        Tee-Object -FilePath $log -Append
    "    claims being confirmed are fixed BEFORE the fresh seeds exist." |
        Tee-Object -FilePath $log -Append
} else {
    Synth "C confirmatory" "output-boba-confirmatory" @(
        "--functions", "all", "--acq", "all",
        "--iterations", "50", "--initial-samples", "5",
        "--error-models", "gaussian", "--jitter-stds", $STDS,
        "--jitter-iterations", "0,20", "--error-bias-mode", "scaled",
        "--seeds", $FRESH
    )
}

# --- F. fitted-oracle companion ------------------------------------------------
$manifest = "output\per_dataset\manifest.json"
if (-not (Test-Path $manifest)) {
    "[F] generating per-dataset configs..." | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\make_per_dataset_configs.py *>> $log
}
$entries = Get-Content $manifest | ConvertFrom-Json
foreach ($name in $entries.PSObject.Properties.Name) {
    $e = $entries.$name
    $outdir = "output-fitted\$name"
    if (Test-Path "$outdir\run_metadata.json") {
        "[F $name] already present." | Tee-Object -FilePath $log -Append
        continue
    }
    "[F $name] starting (sigma_f=$($e.sigma_f)) $(Get-Date -Format o)" |
        Tee-Object -FilePath $log -Append
    New-Item -ItemType Directory -Force $outdir | Out-Null
    & $PYTHON scripts\bo_sensor_error_simulation.py `
        --dataset-config $e.config --objective composite --acq all `
        --oracle-model auto --oracle-selection-path output\best_oracle_models.json `
        --iterations 50 --initial-samples 5 `
        --error-models gaussian --jitter-stds $e.jitter_stds --jitter-iterations 0,20 `
        --seeds $SEEDS10 --output-dir $outdir --parallel --n-jobs $JOBS --resume *>> $log
    if ($LASTEXITCODE -ne 0) { Fail "F $name" }
}

# --- evaluation ----------------------------------------------------------------
foreach ($dir in @("output-boba-ladder", "output-boba-mo", "output-boba-budget25",
                   "output-boba-budget100", "output-boba-confirmatory")) {
    if (-not (Test-Path $dir)) { continue }
    "[eval] $dir $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    $jobs = @()
    foreach ($d in Get-ChildItem -Path $dir -Directory | Where-Object { Test-Path "$($_.FullName)\*.csv" }) {
        $jobs += Start-Job -ScriptBlock {
            param($py, $root, $parent, $name)
            Set-Location $root
            & $py scripts\evaluate_research_question.py `
                --input-dir "$parent\$name" --output-dir "$parent\$name\evaluation" 2>&1
        } -ArgumentList $PYTHON, $PSScriptRoot, $dir, $d.Name
        while ((Get-Job -State Running).Count -ge 6) { Start-Sleep -Seconds 5 }
    }
    $jobs | Wait-Job | Out-Null
    $jobs | ForEach-Object { Receive-Job $_ *>> $log }
    $jobs | Remove-Job
}
foreach ($name in $entries.PSObject.Properties.Name) {
    $outdir = "output-fitted\$name"
    if (-not (Test-Path $outdir)) { continue }
    "[eval] $outdir" | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\evaluate_research_question.py `
        --input-dir $outdir --output-dir "$outdir\evaluation" *>> $log
}

"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")

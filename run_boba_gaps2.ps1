# The last three closable gaps in the known-function study.
#
#   R. RATING INSTRUMENT (~2 h)
#      The synthetic objective is unbounded and continuous; a person returns a
#      bounded, discrete rating, and at 5 sigma the simulated rater is producing
#      values no instrument could represent. The three archival studies put the
#      instrument's range at 13, 17 and 22 sigma with steps of 0.55, 0.54 and
#      1.12 sigma, so this arm clips to +/-8 sigma and rounds to 0.55 -- the real
#      instrument, expressed in the study's own currency. It could plausibly go
#      either way: discretisation destroys information, but clipping also bounds
#      how far a noise spike can move the incumbent.
#
#   S. ROBUST-BO BASELINES (~3 h)
#      All twelve arms so far are ordinary acquisitions run under noise, so the
#      study can say which standard choice survives best but not how much a
#      method built for the problem recovers. Two baselines:
#        qkg     the knowledge gradient, which values the improvement in the
#                posterior MAXIMUM rather than in any observed value.
#        replei  the practical answer: spend half the budget re-asking about the
#                incumbent. Compared at a fixed number of EVALUATIONS, so it buys
#                averaging with half the designs -- the honest trade.
#      Both are compared against the main sweep's arms at the same seeds and
#      conditions, so no reference runs are needed here.
#
#   N. FITTED ORACLE WITHOUT AUGMENTATION (~2 h)
#      The default jitter augmentation costs up to 0.37 held-out R^2 on the
#      archival data. The companion arm (run_boba_gaps_rest.ps1) uses the
#      deployed setting; this one repeats it with the augmentation off, so the
#      comparison between the arms is not silently a comparison between two
#      differently-handicapped oracles.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_gaps2.ps1
#
# Resumable. Intended to run AFTER run_boba_gaps_rest.ps1 frees its workers.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$log = "output-boba\gaps2.log"
if (-not (Test-Path "output-boba")) { New-Item -ItemType Directory "output-boba" | Out-Null }
"=== BOBA gap arms [instrument/robust/no-aug] (re)started $(Get-Date -Format o) ===" |
    Tee-Object -FilePath $log -Append

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
$STDS    = "0.05,0.25,1.0,5.0"
$JOBS    = 10

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
    $argv = @("scripts\bo_synthetic_error_simulation.py", "--output-dir", $outdir,
              "--n-jobs", $JOBS, "--resume") + $extra
    & $PYTHON @argv *>> $log
    if ($LASTEXITCODE -ne 0) { Fail $name }
}

# --- R. rating instrument ------------------------------------------------------
Synth "R instrument" "output-boba-instrument" @(
    "--functions", "all", "--acq-list", "logei,ei,pi,ucb,qucb,qnei",
    "--iterations", "50", "--initial-samples", "5",
    "--error-models", "gaussian", "--jitter-stds", $STDS,
    "--jitter-iterations", "0,20",
    "--response-clip=-8,8", "--response-round", "0.55",
    "--seeds", $SEEDS5
)

# --- S. robust-BO baselines ----------------------------------------------------
Synth "S robust baselines" "output-boba-robust" @(
    "--functions", "all", "--acq-list", "qkg,replei",
    "--iterations", "50", "--initial-samples", "5",
    "--error-models", "gaussian", "--jitter-stds", $STDS,
    "--jitter-iterations", "0,20", "--seeds", $SEEDS5
)

# --- N. fitted oracle, augmentation off ----------------------------------------
$manifest = "output\per_dataset\manifest.json"
if (-not (Test-Path $manifest)) {
    & $PYTHON scripts\make_per_dataset_configs.py *>> $log
}
$entries = Get-Content $manifest | ConvertFrom-Json
foreach ($name in $entries.PSObject.Properties.Name) {
    $e = $entries.$name
    $outdir = "output-fitted-noaug\$name"
    if (Test-Path "$outdir\run_metadata.json") {
        "[N $name] already present." | Tee-Object -FilePath $log -Append
        continue
    }
    "[N $name] starting $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    New-Item -ItemType Directory -Force $outdir | Out-Null
    & $PYTHON scripts\bo_sensor_error_simulation.py `
        --dataset-config $e.config --objective composite --acq all `
        --oracle-model auto --oracle-selection-path output\best_oracle_models.json `
        --oracle-augmentation none `
        --iterations 50 --initial-samples 5 `
        --error-models gaussian --jitter-stds $e.jitter_stds --jitter-iterations 0,20 `
        --seeds $SEEDS10 --output-dir $outdir --parallel --n-jobs $JOBS --resume *>> $log
    if ($LASTEXITCODE -ne 0) { Fail "N $name" }
}

# --- evaluation ----------------------------------------------------------------
foreach ($dir in @("output-boba-instrument", "output-boba-robust")) {
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
    $outdir = "output-fitted-noaug\$name"
    if (-not (Test-Path $outdir)) { continue }
    & $PYTHON scripts\evaluate_research_question.py `
        --input-dir $outdir --output-dir "$outdir\evaluation" *>> $log
}

"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")

# Follow-up arms for the known-function study (run after run_boba_pipeline.ps1).
#
# The main sweep answers "how much does feedback error cost BO, and what
# predicts it". These three arms answer the questions a reviewer asks next, and
# each is scoped to the smallest design that can answer its own question rather
# than repeating the full factorial.
#
#   A. KNOWN NOISE  (~2 h)
#      The main sweep's GP fits its observation noise as a free hyperparameter
#      under BoTorch's shrinking prior, so a measured cost of noise mixes two
#      things: the information the error destroys, and the surrogate never
#      realising the error is there. This arm passes the true injected variance
#      as train_Yvar. The difference between the two is the second thing.
#      Scoped to the gaussian error model -- the only one whose variance is
#      unambiguous -- and to six acquisitions spanning the improvement,
#      confidence-bound and noise-aware families.
#
#   B. INCUMBENT  (~2 h)
#      best_f is the max POSTERIOR MEAN at visited points, which shrinks under
#      noise and so makes improvement-based acquisitions automatically more
#      exploratory. The arms differ in exposure: logei/ei/pi/logpi/qei consume
#      best_f, ucb does not, qnei ignores it. So "acquisition robustness" is
#      partly "incumbent robustness". --incumbent observed_max separates them,
#      with ucb and qnei as the controls that should barely move.
#
#   C. CAUSAL MANIPULATIONS  (~6.5 h)
#      Across any real benchmark suite opt_z, sparsity and skew correlate at
#      ~0.9 -- close to a structural identity for a bounded function with an
#      isolated optimum -- so the main sweep's descriptor regression cannot say
#      which of them matters. These six purpose-built landscapes vary them
#      independently: amplitude moves opt_z 5.0 -> 21.7 at fixed width, width
#      moves sparsity at roughly fixed opt_z, and Levy at d=4/7/11 is one
#      function family across a dimension ladder. This turns the study's central
#      claim from a correlation into a manipulation.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_followups.ps1
#
# Resumable, like the main pipeline. Each arm writes its own directory so the
# main sweep's evaluation is never re-run.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$log = "output-boba\followups.log"
if (-not (Test-Path "output-boba")) { New-Item -ItemType Directory "output-boba" | Out-Null }
"=== BOBA follow-up arms (re)started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

if (-not (Test-Path "output-boba\SWEEP_COMPLETE")) {
    "The main sweep has not finished (no output-boba\SWEEP_COMPLETE). Run run_boba_pipeline.ps1 first." |
        Tee-Object -FilePath $log -Append
    exit 1
}

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

# Five seeds. The contrasts here are WITHIN-cell (known vs learned on the same
# benchmark, condition and seed), which is a far tighter comparison than the
# main sweep's between-condition one, so five buys as much as ten does there.
$SEEDS = "7,8,9,10,11"
$STDS  = "0.05,0.25,1.0,5.0"
$JITS  = "0,20"
$JOBS  = 24

function Fail($stage) {
    "FAILED: $stage (exit $LASTEXITCODE) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    [void][Power]::SetThreadExecutionState([uint32]"0x80000000")
    exit 1
}

function Sweep($name, $outdir, $extra) {
    if (Test-Path "$outdir\SWEEP_COMPLETE") {
        "[$name] already complete." | Tee-Object -FilePath $log -Append
        return
    }
    "[$name] starting $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    $args = @(
        "scripts\bo_synthetic_error_simulation.py",
        "--iterations", "50", "--initial-samples", "5",
        "--jitter-stds", $STDS, "--jitter-iterations", $JITS,
        "--error-bias-mode", "scaled",
        "--seeds", $SEEDS, "--output-dir", $outdir,
        "--n-jobs", $JOBS, "--resume"
    ) + $extra
    & $PYTHON @args *>> $log
    if ($LASTEXITCODE -ne 0) { Fail $name }
}

# --- A. known noise ------------------------------------------------------------
Sweep "A known-noise" "output-boba-knownnoise" @(
    "--functions", "all",
    "--acq-list", "logei,ei,pi,ucb,qucb,qnei",
    "--error-models", "gaussian",
    "--observation-noise", "known"
)

# --- B. incumbent --------------------------------------------------------------
# ucb and qnei are the controls: neither consumes best_f, so neither should move.
Sweep "B incumbent" "output-boba-incumbent" @(
    "--functions", "all",
    "--acq-list", "logei,ei,pi,logpi,qei,ucb,qnei",
    "--error-models", "gaussian",
    "--incumbent", "observed_max"
)

# --- C. causal manipulations ---------------------------------------------------
Sweep "C manipulations" "output-boba-extensions" @(
    "--functions", "extensions",
    "--acq", "all",
    "--error-models", "gaussian,bias,drift,ar1"
)

# --- evaluation and synthesis --------------------------------------------------
foreach ($dir in @("output-boba-knownnoise", "output-boba-incumbent", "output-boba-extensions")) {
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
    & $PYTHON scripts\analyse_boba_robustness.py --input-dir $dir --output-dir "$dir\analysis" *>> $log
}

# --- the arm-vs-arm contrasts, which are the actual point ----------------------
"[contrast] known-noise vs learned-noise" | Tee-Object -FilePath $log -Append
& $PYTHON scripts\compare_boba_arms.py `
    --reference output-boba --treatment output-boba-knownnoise `
    --label-reference "learned noise" --label-treatment "known noise" `
    --output-dir "output-boba-knownnoise\analysis" *>> $log

"[contrast] observed-max vs posterior-mean incumbent" | Tee-Object -FilePath $log -Append
& $PYTHON scripts\compare_boba_arms.py `
    --reference output-boba --treatment output-boba-incumbent `
    --label-reference "posterior-mean incumbent" --label-treatment "observed-max incumbent" `
    --output-dir "output-boba-incumbent\analysis" *>> $log

"=== DONE $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")

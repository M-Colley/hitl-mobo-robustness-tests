# I. INPUT ERROR -- the person acts on the wrong design.
#
# Every other arm in this study corrupts the RATING: the optimizer is told the
# wrong number about the design it proposed. A wrong value at the right place.
# This arm corrupts the DESIGN. The optimizer proposes x, the person actually
# experiences x' != x -- slider overshoot, wrong option pressed, wrong config
# applied -- rates x' HONESTLY, and that rating is filed against x. A right
# value at the wrong place.
#
# WHY IT NEEDS KNOWN FUNCTIONS, more than the response arm does. Scoring a slip
# requires the objective AT THE POINT ACTUALLY TOUCHED. With a fitted oracle you
# get f_hat(x'), so the measured damage is the slip's cost PLUS the oracle's
# error at a second location. With an exact function f(x') is exact.
#
# MAGNITUDE IS NOT IN LANDSCAPE SDs. --input-error-from-sweep repurposes the
# --jitter-stds grid: for `slip` the value is the positional SD as a FRACTION OF
# EACH COORDINATE'S RANGE, for `misclick` it is the probability of landing
# somewhere else entirely. Both are dimensionless in [0,1], which is why one
# grid serves both -- but neither is comparable with a response-error sigma_e,
# and the two families of arm must never be pooled.
#
# THREE VARIANTS, because "which design gets written down" is the interesting
# axis and not a detail:
#   slip          pointer imprecision, unnoticed. The surrogate trains on
#                 (x, f(x')), and the design eventually deployed is the logged x.
#   misclick      a wrong press, unnoticed. Rare and large.
#   slip-actual   the counterfactual where the slip IS detected and logged,
#                 (x', f(x')). Its slip draws are identical to `slip`'s (same
#                 seeds), so the contrast separates the cost of going to the
#                 wrong place from the cost of mislabelling it.
#
# MODEL-FREE FLOORS run alongside, and here they are not a zero control. Under a
# response error a floor's excess is identically zero, because it never reads
# an observation. Under an input error it evaluates the wrong points too, so its
# excess is the pure geometric cost of a slip with no learning to corrupt -- the
# baseline against which the model-based arms' extra damage is measured.
#
# LABELS AND SEEDS, both of which broke once on this arm (2026-09-10). Jittered
# runs are labelled by the corruption applied ("slip", "misclick"), never
# "none", which is reserved for baselines and is how the evaluator recognises
# them. The noise seed uses the carrier error model's index, which is frozen by
# tests/test_error_model_labels.py.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_inputerror.ps1              # all three in sequence, then evaluate
#     powershell -ExecutionPolicy Bypass -File run_boba_inputerror.ps1 -Arm slip    # one arm only
#     powershell -ExecutionPolicy Bypass -File run_boba_inputerror.ps1 -Arm eval    # evaluate whatever exists
#
# The three arms share no files, so they can run as three concurrent
# invocations (6 workers each) instead of one sequential one: the same results,
# because a run is fully determined by its seed and configuration, in well under
# half the wall-clock time. Resumable in every mode.

param(
    [ValidateSet("all", "slip", "misclick", "slip-actual", "eval")]
    [string]$Arm = "all",
    [int]$Jobs = 0
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
if (-not (Test-Path "output-boba")) { New-Item -ItemType Directory "output-boba" | Out-Null }

# One log per invocation, so concurrent arms never contend for a file.
$log = if ($Arm -eq "all") { "output-boba\inputerror.log" } else { "output-boba\inputerror-$Arm.log" }
"=== BOBA input-error arm [$Arm] (re)started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

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

$SEEDS = "7,8,9,10,11"
# Box fractions for `slip`, probabilities for `misclick`.
$GRID  = "0.01,0.05,0.15,0.4"
# Six model-based acquisitions, matching the instrument and budget arms, plus the
# two model-free floors described above.
$ACQ   = "logei,ei,pi,ucb,qucb,qnei,random,sobol"
if ($Jobs -le 0) {
    $Jobs = if ($env:INPUTERR_JOBS) { [int]$env:INPUTERR_JOBS } elseif ($Arm -eq "all") { 8 } else { 6 }
}
"Workers: $Jobs" | Tee-Object -FilePath $log -Append

$ARMS = [ordered]@{
    "slip"        = @{ Dir = "output-boba-slip";        Model = "slip";     Recorded = "proposed" }
    "misclick"    = @{ Dir = "output-boba-misclick";    Model = "misclick"; Recorded = "proposed" }
    "slip-actual" = @{ Dir = "output-boba-slip-actual"; Model = "slip";     Recorded = "actual" }
}

function Fail($stage) {
    "FAILED: $stage (exit $LASTEXITCODE) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    [void][Power]::SetThreadExecutionState([uint32]"0x80000000")
    exit 1
}

function Run-Arm($name) {
    $dir = $ARMS[$name].Dir
    $model = $ARMS[$name].Model
    $recorded = $ARMS[$name].Recorded
    if (Test-Path "$dir\SWEEP_COMPLETE") {
        "[$name] already complete." | Tee-Object -FilePath $log -Append
        return
    }
    "[$name] starting $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\bo_synthetic_error_simulation.py `
        --functions all --acq-list $ACQ `
        --iterations 50 --initial-samples 5 `
        --error-models gaussian --jitter-stds $GRID --jitter-iterations 0,20 `
        --input-error $model --input-error-from-sweep --input-error-recorded $recorded `
        --seeds $SEEDS --output-dir $dir --n-jobs $Jobs --resume *>> $log
    if ($LASTEXITCODE -ne 0) { Fail $name }
}

function Evaluate-Arms {
    foreach ($name in $ARMS.Keys) {
        $dir = $ARMS[$name].Dir
        if (-not (Test-Path $dir)) { continue }
        "[eval] $dir $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
        foreach ($d in Get-ChildItem -Path $dir -Directory) {
            if ($d.Name -in @("evaluation", "analysis")) { continue }
            if (-not (Get-ChildItem -Path $d.FullName -Filter *.csv -ErrorAction SilentlyContinue)) { continue }
            & $PYTHON scripts\evaluate_research_question.py `
                --input-dir $d.FullName --output-dir (Join-Path $d.FullName "evaluation") *>> $log
            # Loud, unlike the older runners: the evaluator now refuses a
            # corrupted run labelled as a baseline, and that must not scroll past.
            if ($LASTEXITCODE -ne 0) { Fail "eval $dir\$($d.Name)" }
        }
    }
}

switch ($Arm) {
    "all"   { foreach ($name in $ARMS.Keys) { Run-Arm $name }; Evaluate-Arms }
    "eval"  { Evaluate-Arms }
    default { Run-Arm $Arm }
}

"=== DONE [$Arm] $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")

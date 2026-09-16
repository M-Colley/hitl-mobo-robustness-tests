# The process-adaptation arms of docs/adaptations-proposal.md, one per call.
#
#   rep10        E4/E1  gaussian error; LogEI and qNEI; the first 10 proposals rated twice
#   rep10-obs    E1     as rep10 with LogEI on the observed-max incumbent (the bundle
#                       for an improvement-based acquisition)
#   rerate       E5     gaussian error; the last 6 trials re-rate the top 3 designs twice
#   rerate-slip  E5     the same under an unnoticed slip
#   nigp         E3     unnoticed slip with a noisy-input GP
#   studentt     E2     misclick with an outlier-robust surrogate (skipped until the
#                       driver has --likelihood)
#
# Every arm: 20 landscapes, seeds 7-11, onsets 0 and 20, T = 50 with 5 initial
# samples, so it pairs with the main sweep restricted to gaussian error and
# these seeds (rep10, rerate), or with the slip / misclick arms (rerate-slip,
# nigp, studentt) on the same acquisitions.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_adapt.ps1 -Arm rep10 -Jobs 8
#     powershell -ExecutionPolicy Bypass -File run_boba_adapt.ps1 -Arm eval

param(
    [ValidateSet("rep10", "rep10-obs", "rerate", "rerate-slip", "nigp", "studentt", "eval")]
    [string]$Arm = "rep10",
    [int]$Jobs = 8
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
if (-not (Test-Path "output-boba")) { New-Item -ItemType Directory "output-boba" | Out-Null }
$log = "output-boba\adapt-$Arm.log"
"=== adapt arm $Arm started $(Get-Date -Format o) (jobs $Jobs) ===" | Tee-Object -FilePath $log -Append

Add-Type @'
using System; using System.Runtime.InteropServices;
public static class PowerAdapt { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags); }
'@
[void][PowerAdapt]::SetThreadExecutionState([uint32]"0x80000001")

$PYTHON = if ($env:PYTHON) { $env:PYTHON } else {
    $c = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    & $c -c "import torch, botorch" 2>$null
    if ($LASTEXITCODE -eq 0) { $c } else { "python" }
}

$SEEDS = "7,8,9,10,11"
$RESPONSE_GRID = "0.05,0.25,1,5"
$INPUT_GRID = "0.01,0.05,0.15,0.4"

$ARMS = [ordered]@{
    "rep10"       = @{ Dir = "output-boba-adapt-rep10";       Acq = "logei,qnei"; Grid = $RESPONSE_GRID; Extra = @("--replicate-first", "10") }
    "rep10-obs"   = @{ Dir = "output-boba-adapt-rep10-obs";   Acq = "logei";      Grid = $RESPONSE_GRID; Extra = @("--replicate-first", "10", "--incumbent", "observed_max") }
    "rerate"      = @{ Dir = "output-boba-adapt-rerate";      Acq = "logei,qnei"; Grid = $RESPONSE_GRID; Extra = @("--final-rerate", "3,2") }
    "rerate-slip" = @{ Dir = "output-boba-adapt-rerate-slip"; Acq = "logei,qnei"; Grid = $INPUT_GRID;    Extra = @("--final-rerate", "3,2", "--input-error", "slip", "--input-error-from-sweep") }
    "nigp"        = @{ Dir = "output-boba-adapt-nigp";        Acq = "logei,qnei"; Grid = $INPUT_GRID;    Extra = @("--input-error", "slip", "--input-error-from-sweep", "--input-noise-model", "nigp") }
    "studentt"    = @{ Dir = "output-boba-adapt-studentt";    Acq = "logei,qnei"; Grid = $INPUT_GRID;    Extra = @("--input-error", "misclick", "--input-error-from-sweep", "--likelihood", "student_t") }
}

function Fail($stage) {
    "FAILED: $stage (exit $LASTEXITCODE) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    [void][PowerAdapt]::SetThreadExecutionState([uint32]"0x80000000")
    exit 1
}

function Run-Arm($name) {
    $a = $ARMS[$name]
    if (Test-Path "$($a.Dir)\SWEEP_COMPLETE") { "[$name] already complete." | Tee-Object -FilePath $log -Append; return }
    if ($name -eq "studentt") {
        $help = & $PYTHON scripts\bo_synthetic_error_simulation.py --help 2>&1 | Out-String
        if ($help -notmatch "--likelihood") { "[$name] skipped: the driver has no --likelihood yet." | Tee-Object -FilePath $log -Append; return }
    }
    "[$name] starting $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    & $PYTHON scripts\bo_synthetic_error_simulation.py `
        --functions all --acq-list $a.Acq `
        --iterations 50 --initial-samples 5 `
        --error-models gaussian --jitter-stds $a.Grid --jitter-iterations 0,20 `
        --seeds $SEEDS --output-dir $a.Dir --n-jobs $Jobs --resume @($a.Extra) *>> $log
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
            if ($LASTEXITCODE -ne 0) { Fail "eval $dir\$($d.Name)" }
        }
    }
}

switch ($Arm) {
    "eval"  { Evaluate-Arms }
    default { Run-Arm $Arm }
}
"=== adapt arm $Arm finished $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
[void][PowerAdapt]::SetThreadExecutionState([uint32]"0x80000000")

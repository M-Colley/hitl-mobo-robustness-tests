# Budget-neutral robustness arms: every change keeps the number of human trials
# equal to the standard process (or lowers it). One script owns the ordering.
#
#   q-*       acquisition changes: cautious (LCB) incumbent, augmented EI,
#             Thompson sampling, a noise-chasing guard, slip-aware acquisition
#   sched-*   the same total rating effort, spent unevenly across trials
#   session*  the same session time, spent on fewer precise or more quick trials
#   spike*    gross faults: standard, clipped scale, relevance-pursuit GP
#   relay*    several raters taking turns, with and without an offset model
#   missing*  ratings that never arrive, dropped or imputed low
#   ceiling   a rating scale that saturates, fixed or re-anchored
#   mo-halo*  shared rating error across objectives, with and without a model
#
# Every arm: 20 landscapes (scalar), seeds 7-11, T = 50 with 5 initial samples
# unless stated. Settings that change the CLEAN run get their own directory,
# because clean baselines are written without the variant suffix.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_budget_neutral.ps1 -Arm all -Jobs 20
#     powershell -ExecutionPolicy Bypass -File run_boba_budget_neutral.ps1 -Arm q-aei -Jobs 20
#
# Resumable: the driver's --resume skips completed runs, and a finished variant
# leaves a marker file in its directory.
#
# -Smoke runs every variant of every named arm end to end at toy size (one
# landscape, one seed, 8 trials, the first magnitude and onset of the grid) into
# -SmokeDir, writing no marker and touching no real output directory. A long run
# should never be started without it: a flag the simulator does not know, or an
# arm the guard silently skips, costs hours otherwise.
#
#     powershell -ExecutionPolicy Bypass -File run_boba_budget_neutral.ps1 -Smoke

param(
    [string]$Arm = "all",
    [int]$Jobs = 20,
    [int]$MoJobs = 8,
    [switch]$Smoke,
    [string]$SmokeDir = "$env:TEMP\boba-smoke"
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
if (-not (Test-Path "output-boba")) { New-Item -ItemType Directory "output-boba" | Out-Null }
if ($Smoke -and -not (Test-Path $SmokeDir)) { New-Item -ItemType Directory -Force $SmokeDir | Out-Null }
$log = if ($Smoke) { Join-Path $SmokeDir "smoke.log" } else { "output-boba\budget-neutral.log" }
"=== budget-neutral arms ($Arm) started $(Get-Date -Format o) (jobs $Jobs) ===" | Tee-Object -FilePath $log -Append
if (-not $Smoke) { "$PID" | Set-Content -Path "output-boba\budget-neutral.PID" -Encoding ascii }

Add-Type @'
using System; using System.Runtime.InteropServices;
public static class PowerBN { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags); }
'@
[void][PowerBN]::SetThreadExecutionState([uint32]"0x80000001")

$PYTHON = if ($env:PYTHON) { $env:PYTHON } else { "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" }

$S5 = "7,8,9,10,11"
$S10 = "7,8,9,10,11,12,13,14,15,16"
$GRID4 = "0.05,0.25,1,5"
$SPIKES = @(
    "--error-spike-std-mode fixed --error-spike-prob 0.05 --error-spike-std 5",
    "--error-spike-std-mode fixed --error-spike-prob 0.15 --error-spike-std 5",
    "--error-spike-std-mode fixed --error-spike-prob 0.05 --error-spike-std 20",
    "--error-spike-std-mode fixed --error-spike-prob 0.15 --error-spike-std 20"
)

# Name = @{ Dir; Acq; Err; Grid; Onsets; Iter; Seeds; Need (a flag the driver must know); Variants; Mo }
$ARMS = [ordered]@{
    "q-inclcb"      = @{ Dir = "output-boba-q-inclcb"; Acq = "logei,logpi"; Err = "gaussian"; Grid = $GRID4; Onsets = "0,20"; Need = "lcb";
                         Variants = @("--incumbent lcb") }
    "q-aei"         = @{ Dir = "output-boba-q-aei"; Acq = "aei"; Err = "gaussian"; Grid = $GRID4; Onsets = "0,20"; Need = "aei"; Variants = @("") }
    "q-ts"          = @{ Dir = "output-boba-q-ts"; Acq = "ts"; Err = "gaussian"; Grid = $GRID4; Onsets = "0,20"; Need = "ts"; Variants = @("") }
    "sched-front10" = @{ Dir = "output-boba-sched-front10"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "1,5"; Onsets = "0"; Need = "--noise-schedule";
                         Variants = @("--noise-schedule front10") }
    "sched-front20" = @{ Dir = "output-boba-sched-front20"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "1,5"; Onsets = "0"; Need = "--noise-schedule";
                         Variants = @("--noise-schedule front20") }
    "sched-U"       = @{ Dir = "output-boba-sched-U"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "1,5"; Onsets = "0"; Need = "--noise-schedule";
                         Variants = @("--noise-schedule U") }
    "sched-back10"  = @{ Dir = "output-boba-sched-back10"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "1,5"; Onsets = "0"; Need = "--noise-schedule";
                         Variants = @("--noise-schedule back10") }
    "session25"     = @{ Dir = "output-boba-session25"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "0.7071,3.5355"; Onsets = "0"; Iter = 25; Variants = @("") }
    "session100"    = @{ Dir = "output-boba-session100"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "1.4142,7.0711"; Onsets = "0"; Iter = 100; Variants = @("") }
    "spike"         = @{ Dir = "output-boba-spike"; Acq = "logei,qnei"; Err = "spike"; Grid = "0.25"; Onsets = "0"; Variants = $SPIKES }
    "spike-clip"    = @{ Dir = "output-boba-spike-clip"; Acq = "logei,qnei"; Err = "spike"; Grid = "0.25"; Onsets = "0";
                         Variants = @($SPIKES | ForEach-Object { "$_ --response-clip sample" }) }
    "relay"         = @{ Dir = "output-boba-relay"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "0.25,1"; Onsets = "0"; Need = "--rater-assign";
                         Variants = @("--rater-assign block:10 --rater-offset-ratio 2", "--rater-assign roundrobin:5 --rater-offset-ratio 2") }
    # The offset model changes the clean run, and the assignment decides who
    # rated what in it, so the two assignments cannot share a directory.
    "relay-backfit-block" = @{ Dir = "output-boba-relay-backfit-block"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "0.25,1"; Onsets = "0"; Need = "--rater-model";
                         Variants = @("--rater-assign block:10 --rater-offset-ratio 2 --rater-model backfit") }
    "relay-backfit-rr" = @{ Dir = "output-boba-relay-backfit-rr"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "0.25,1"; Onsets = "0"; Need = "--rater-model";
                         Variants = @("--rater-assign roundrobin:5 --rater-offset-ratio 2 --rater-model backfit") }
    "missing-drop"  = @{ Dir = "output-boba-missing-drop"; Acq = "logei,ucb"; Err = "gaussian"; Grid = "0.15,0.3"; Onsets = "0"; Need = "--missing-handling";
                         Variants = @("--input-error missing_mcar --input-error-from-sweep --missing-handling drop",
                                      "--input-error missing_low --input-error-from-sweep --missing-handling drop") }
    "missing-impute" = @{ Dir = "output-boba-missing-impute"; Acq = "logei,ucb"; Err = "gaussian"; Grid = "0.15,0.3"; Onsets = "0"; Need = "--missing-handling";
                         Variants = @("--input-error missing_mcar --input-error-from-sweep --missing-handling impute_low",
                                      "--input-error missing_low --input-error-from-sweep --missing-handling impute_low") }
    "ceiling"       = @{ Dir = "output-boba-ceiling"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "0.25,1"; Onsets = "0"; Need = "--response-ceiling";
                         Variants = @("--response-ceiling 0.9 --ceiling-mode fixed", "--response-ceiling 0.9 --ceiling-mode anchored") }
    "q-mind"        = @{ Dir = "output-boba-q-mind"; Acq = "logei,pi"; Err = "gaussian"; Grid = "0.25,1,5"; Onsets = "0,20"; Need = "--min-distance";
                         Variants = @("--min-distance 0.05") }
    "q-iu-0.05"     = @{ Dir = "output-boba-q-iu-0.05"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "0.05"; Onsets = "0,20"; Need = "--input-uncertain-acq";
                         Variants = @("--input-error slip --input-error-from-sweep --input-uncertain-acq 16 --input-uncertain-scale 0.05") }
    "q-iu-0.15"     = @{ Dir = "output-boba-q-iu-0.15"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "0.15"; Onsets = "0,20"; Need = "--input-uncertain-acq";
                         Variants = @("--input-error slip --input-error-from-sweep --input-uncertain-acq 16 --input-uncertain-scale 0.15") }
    "q-iu-0.4"      = @{ Dir = "output-boba-q-iu-0.4"; Acq = "logei,qnei"; Err = "gaussian"; Grid = "0.4"; Onsets = "0,20"; Need = "--input-uncertain-acq";
                         Variants = @("--input-error slip --input-error-from-sweep --input-uncertain-acq 16 --input-uncertain-scale 0.4") }
    "spike-rrp"     = @{ Dir = "output-boba-spike-rrp"; Acq = "logei,qnei"; Err = "spike"; Grid = "0.25"; Onsets = "0"; Need = "relevance_pursuit";
                         Variants = @($SPIKES | ForEach-Object { "$_ --likelihood relevance_pursuit" }) }
    "mo-halo"       = @{ Dir = "output-boba-mo-halo"; Acq = "qlognehvi"; Err = "gaussian"; Grid = "1"; Onsets = "0,20"; Seeds = $S10; Mo = $true;
                         Need = "--error-cross-corr"; Variants = @("", "--error-cross-corr 0.85") }
    "mo-halo-backfit" = @{ Dir = "output-boba-mo-halo-backfit"; Acq = "qlognehvi"; Err = "gaussian"; Grid = "1"; Onsets = "0,20"; Seeds = $S10; Mo = $true;
                         Need = "--mo-halo-model"; Variants = @("--mo-halo-model backfit", "--mo-halo-model backfit --error-cross-corr 0.85") }
}

$HELP = & $PYTHON scripts\bo_synthetic_error_simulation.py --help 2>&1 | Out-String

function Marker($dir, $variant) {
    $tag = if ($variant) { ($variant -replace '[^A-Za-z0-9\.]+', '_').Trim('_') } else { "standard" }
    return Join-Path $dir ".done-$tag"
}

$script:Failures = @()

function Run-Arm($name) {
    $a = $ARMS[$name]
    if ($a.Need -and ($HELP -notmatch [regex]::Escape($a.Need))) {
        "[$name] SKIPPED: the driver does not know '$($a.Need)'." | Tee-Object -FilePath $log -Append
        $script:Failures += "$name (guard: '$($a.Need)' not in --help)"
        return
    }
    $iter = if ($a.Iter) { $a.Iter } else { 50 }
    $seeds = if ($a.Seeds) { $a.Seeds } else { $S5 }
    $jobs = if ($a.Mo) { $MoJobs } else { $Jobs }
    $dir = $a.Dir
    $grid = $a.Grid
    $onsets = $a.Onsets
    $scope = if ($a.Mo) { @("--multi-objective", "--functions", "branincurrin,zdt1,dtlz2,vehiclesafety") } else { @("--functions", "all") }
    if ($Smoke) {
        # One landscape, one seed, 8 trials, the first magnitude and onset: the
        # point is that the command line runs, not what it finds. The effort
        # presets name absolute trials (1-20, ...), so those arms keep their T.
        $keepsT = @($a.Variants | Where-Object { $_ -match 'noise-schedule' }).Count -gt 0
        $dir = Join-Path $SmokeDir $name
        if (-not $keepsT) { $iter = 8 }
        $seeds = "7"
        $jobs = 2
        $grid = ($grid -split ',')[0]
        $onsets = ($onsets -split ',')[0]
        $scope = if ($a.Mo) { @("--multi-objective", "--functions", "branincurrin") } else { @("--functions", "branin") }
    }
    $before = $script:Failures.Count
    foreach ($v in $a.Variants) {
        $marker = Marker $dir $v
        if ((-not $Smoke) -and (Test-Path $marker)) { "[$name] variant '$v' already complete." | Tee-Object -FilePath $log -Append; continue }
        "[$name] variant '$v' starting $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
        $extra = @()
        if ($v) { $extra = $v -split ' ' | Where-Object { $_ } }
        & $PYTHON scripts\bo_synthetic_error_simulation.py @scope --acq-list $a.Acq `
            --iterations $iter --initial-samples 5 `
            --error-models $a.Err --jitter-stds $grid --jitter-iterations $onsets `
            --seeds $seeds --output-dir $dir --n-jobs $jobs --resume @extra *>> $log
        if ($LASTEXITCODE -ne 0) {
            "[$name] FAILED variant '$v' (exit $LASTEXITCODE) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
            $script:Failures += "$name variant '$v' (exit $LASTEXITCODE)"
            if ($Smoke) { continue }   # a smoke run reports every broken variant, not just the first
            return
        }
        if (-not $Smoke) { New-Item -ItemType File -Force -Path $marker | Out-Null }
    }
    if ($Smoke) {
        $verdict = if ($script:Failures.Count -eq $before) { "smoke ok" } else { "smoke FAILED" }
        "[$name] $verdict $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
        return
    }
    foreach ($d in Get-ChildItem -Path $a.Dir -Directory) {
        if ($d.Name -in @("evaluation", "analysis")) { continue }
        & $PYTHON scripts\evaluate_research_question.py --input-dir $d.FullName --output-dir (Join-Path $d.FullName "evaluation") *>> $log
        if ($LASTEXITCODE -ne 0) { "[$name] FAILED evaluation of $($d.Name)" | Tee-Object -FilePath $log -Append }
    }
    New-Item -ItemType File -Force -Path (Join-Path $a.Dir "ARM_COMPLETE") | Out-Null
    "[$name] complete $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
}

$names = if ($Arm -eq "all") { @($ARMS.Keys) } else { @($Arm -split ',') }
foreach ($n in $names) {
    if (-not $ARMS.Contains($n)) { "unknown arm $n" | Tee-Object -FilePath $log -Append; $script:Failures += "unknown arm $n"; continue }
    Run-Arm $n
}
if ($Smoke) {
    if ($script:Failures.Count -eq 0) {
        "=== smoke: all $($names.Count) arms ran $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
        exit 0
    }
    "=== smoke: $($script:Failures.Count) FAILURES ===" | Tee-Object -FilePath $log -Append
    $script:Failures | ForEach-Object { "  $_" | Tee-Object -FilePath $log -Append }
    exit 1
}
"done $(Get-Date -Format o)" | Set-Content -Path "output-boba\budget-neutral.DONE" -Encoding ascii
"=== budget-neutral arms finished $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
[void][PowerBN]::SetThreadExecutionState([uint32]"0x80000000")
if ($script:Failures.Count -gt 0) {
    "=== $($script:Failures.Count) arm(s) did not finish ===" | Tee-Object -FilePath $log -Append
    $script:Failures | ForEach-Object { "  $_" | Tee-Object -FilePath $log -Append }
    exit 1
}

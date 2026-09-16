# Finish the qNEHVI regeneration in output-chi\ -- FOR ehmi ONLY.
#
# Pre-existing technical debt, unrelated to the known-function study, but real:
# the multi-objective arm of the CHI sweep is incomplete and its completion
# marker lies about it.
#
# History. All pre-2026-07-21 qNEHVI results were a silent fall back to random
# sampling. The fix landed, the regeneration started, and it was interrupted:
#
#     dataset      qehvi  qnehvi  random  sobol   (target 1300 each)
#     ehmi          1300    1217    1300   1300
#     opticarvis    1300       0    1300   1300
#     provoice      1300       0    1300   1300
#
# The opticarvis and provoice qNEHVI CSVs were DELETED by the relaunch, not
# merely superseded.
#
# SCOPE: ehmi only, decided 2026-09-09. The cost is why. An earlier comment here
# guessed "104 core-hours, about 13 h at 8 workers" and was wrong by 10x. The
# measured rate is 19.1 runs/h: the 1,214 ehmi runs that DID complete took 63.7 h
# of production (07-21 19:22 -> 07-24 11:03), and a restart on 2026-09-09
# reproduced it at 10.7 runs/h under contention. qNEHVI is ~4.4x qEHVI here
# (84 runs/h) and ~25x random/sobol (471 runs/h): the noisy variant fantasises
# over the observed points, so cost grows with the design as the run proceeds.
#
#     all three datasets : 2,683 runs / 19.1 per h ~= 140 h ~= 6 days
#     ehmi only          :    83 runs / 19.1 per h ~=   4 h
#
# CONSEQUENCE, and it is not "fewer runs" -- it is a hole. opticarvis and
# provoice end with ZERO qNEHVI runs, so any CHI multi-objective claim about
# them can cover qEHVI, random and sobol but must not mention qNEHVI at all.
# Only ehmi gets the four-way acquisition comparison. Re-running the evaluation
# will silently produce per-dataset tables with qNEHVI missing for two of three
# datasets; say so in the text rather than letting a reader infer a null.
#
# Every flag below is copied from output-chi\run_metadata.json so the
# regenerated runs are comparable with the qehvi/random/sobol runs already
# there; do not "improve" any of them.
#
# The one flag that is NOT from run_metadata.json is --dataset-config, which
# that run did not pass (it defaulted to datasets.json, all three). datasets-
# ehmi.json holds the ehmi entry copied VERBATIM from datasets.json, not the
# resolved copy in run_metadata.json: the resolved one drops oracle_target, and
# the driver then defaults it to "individual", which is the wrong oracle for
# ehmi -- its per-design "mean" oracle is what took held-out R2 from 0.14 to
# 0.55. Verified to resolve to identical data_dirs, observation_glob,
# param_columns and objective_map.
#
# SEEDING. The noise seed includes ACQUISITION_CHOICES.index(acq). qnehvi was
# index 11 when the July runs were made and is 13 now (qkg and replei were
# inserted ahead of it), so the 86 runs this script adds are seeded differently
# from the 1,214 July ones. Each is still a valid draw paired with its own
# baseline and no estimate is biased, but only the new 86 reproduce bit for
# bit on current code.
#
# Two things this script deliberately does NOT do:
#   * It does not touch output-chi\SWEEP_COMPLETE. That marker is stale (written
#     2026-07-21, before the fix) and run_chi_pipeline.ps1 exits immediately when
#     it sees it, which is why the regeneration never resumed on its own. Remove
#     it by hand once you have re-run the evaluation.
#   * It does not re-run the evaluation. output-chi\evaluation\* predates the fix
#     and must be regenerated afterwards, but that is a decision about the CHI
#     paper's numbers rather than a repair.
#
#     powershell -ExecutionPolicy Bypass -File run_chi_qnehvi_repair.ps1

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$log = "output-chi\qnehvi-repair.log"
"=== qNEHVI repair [ehmi only] (re)started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append

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

if (-not (Test-Path "datasets-ehmi.json")) {
    "FAILED: datasets-ehmi.json is missing; regenerate it from the ehmi entry of datasets.json." |
        Tee-Object -FilePath $log -Append
    exit 1
}

$SEEDS = "7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26"

# Worker count is a scheduling choice, not a scientific one -- runs are seeded
# per (seed, condition) and are identical however they are distributed -- so it
# is safe to lower it when the machine is already busy. 8 matches the original.
$JOBS = if ($env:QNEHVI_JOBS) { $env:QNEHVI_JOBS } else { 8 }
"Workers: $JOBS" | Tee-Object -FilePath $log -Append

& $PYTHON scripts\bo_sensor_error_simulation.py `
    --dataset-config datasets-ehmi.json `
    --objective multi_objective --acq-list qnehvi `
    --oracle-model auto --oracle-selection-path output-chi\best_oracle_models.json `
    --error-models gaussian,bias,drift,ar1 --jitter-iterations 0,10,20,40 `
    --response-clip auto `
    --seeds $SEEDS --output-dir output-chi --parallel --n-jobs $JOBS `
    --acq-raw-samples 128 --acq-mc-samples 64 --acq-num-restarts 5 --resume *>> $log

if ($LASTEXITCODE -ne 0) {
    "FAILED (exit $LASTEXITCODE) $(Get-Date -Format o)" | Tee-Object -FilePath $log -Append
    [void][Power]::SetThreadExecutionState([uint32]"0x80000000")
    exit 1
}

"=== DONE [ehmi only] $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
"Next: re-run scripts\evaluate_research_question.py over output-chi, then delete" |
    Tee-Object -FilePath $log -Append
"the stale output-chi\SWEEP_COMPLETE marker. Remember opticarvis and provoice" |
    Tee-Object -FilePath $log -Append
"have NO qNEHVI runs -- do not report a qNEHVI result for either." |
    Tee-Object -FilePath $log -Append
[void][Power]::SetThreadExecutionState([uint32]"0x80000000")

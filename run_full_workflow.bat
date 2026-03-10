@echo off
setlocal

cd /d "%~dp0"

if not defined PYTHON set "PYTHON=python"
if not defined DATASET_CONFIG set "DATASET_CONFIG=datasets.json"
if not defined ORACLE_OUTPUT set "ORACLE_OUTPUT=output\best_oracle_models.json"
if not defined BROAD_OUTPUT set "BROAD_OUTPUT=output"
if not defined BROAD_EVAL_OUTPUT set "BROAD_EVAL_OUTPUT=output\evaluation"
if not defined BROAD_FIGURE_OUTPUT set "BROAD_FIGURE_OUTPUT=output\evaluation\figures"
if not defined CONFIRMATORY_OUTPUT set "CONFIRMATORY_OUTPUT=output\confirmatory"

if not defined ORACLE_MODELS set "ORACLE_MODELS=all"
if not defined ORACLE_CV_FOLDS set "ORACLE_CV_FOLDS=5"

if not defined BROAD_SEEDS set "BROAD_SEEDS=7,8,9,10,11"
if not defined SHORTLIST set "SHORTLIST=pi,logpi,qucb"

if not defined CONFIRMATORY_OBJECTIVE set "CONFIRMATORY_OBJECTIVE=composite"
if not defined CONFIRMATORY_ACQ set "CONFIRMATORY_ACQ=pi,logpi"
if not defined CONFIRMATORY_SEED_START set "CONFIRMATORY_SEED_START=12"
if not defined CONFIRMATORY_SEED_COUNT set "CONFIRMATORY_SEED_COUNT=20"

echo [1/4] Selecting oracle models...
%PYTHON% scripts\select_best_oracle_model.py ^
  --dataset-config %DATASET_CONFIG% ^
  --oracle-models %ORACLE_MODELS% ^
  --cv-folds %ORACLE_CV_FOLDS% ^
  --output-path %ORACLE_OUTPUT%
if errorlevel 1 goto :fail

echo [2/4] Running broad sweep...
%PYTHON% scripts\bo_sensor_error_simulation.py ^
  --dataset-config %DATASET_CONFIG% ^
  --acq all ^
  --oracle-model auto ^
  --oracle-selection-path %ORACLE_OUTPUT% ^
  --seeds %BROAD_SEEDS% ^
  --output-dir %BROAD_OUTPUT% ^
  --parallel
if errorlevel 1 goto :fail

echo [3/4] Evaluating broad sweep and creating figures...
%PYTHON% scripts\evaluate_research_question.py ^
  --input-dir %BROAD_OUTPUT% ^
  --output-dir %BROAD_EVAL_OUTPUT%
if errorlevel 1 goto :fail

%PYTHON% scripts\plot_combined_aspects.py ^
  --evaluation-dir %BROAD_EVAL_OUTPUT% ^
  --output-dir %BROAD_FIGURE_OUTPUT% ^
  --shortlist %SHORTLIST%
if errorlevel 1 goto :fail

echo [4/4] Running confirmatory follow-up...
%PYTHON% scripts\confirmatory_followup.py ^
  --base-input-dir %BROAD_OUTPUT% ^
  --base-evaluation-dir %BROAD_EVAL_OUTPUT% ^
  --output-root %CONFIRMATORY_OUTPUT% ^
  --objective %CONFIRMATORY_OBJECTIVE% ^
  --acquisitions %CONFIRMATORY_ACQ% ^
  --seed-start %CONFIRMATORY_SEED_START% ^
  --num-new-seeds %CONFIRMATORY_SEED_COUNT% ^
  --run-simulation ^
  --parallel
if errorlevel 1 goto :fail

echo Workflow complete.
echo Broad evaluation: %BROAD_EVAL_OUTPUT%
echo Confirmatory outputs: %CONFIRMATORY_OUTPUT%
exit /b 0

:fail
echo Workflow failed with exit code %errorlevel%.
exit /b %errorlevel%

# Runs the four hierarchical-oracle "additional test" configs + the summary figure.
# Idempotent (fixed --seed 7): safe to re-run from scratch after a pause.
#   Run A  : all 5 datasets, composite, extra_trees + tree_hier + hbm_rff + warm
#   Run C  : provoice normalized composite (fixes -Mental Demand 1-17 scale bug)
#   Run B1 : ehmi per-construct (5 UX dimensions)
#   Run B2 : provoice per-construct (3 constructs)
#   Figure : output/figures/hierarchical_oracle.png
# Note: outputs are only written at the END of each run, so an interrupted run
# leaves the previous JSON intact. Each run below is independent.
$ErrorActionPreference = "Continue"
$env:JAX_PLATFORMS = "cpu"
Set-Location $PSScriptRoot
$s = "scripts\hierarchical_oracle_test.py"

Write-Output "=== RUN A: all 5 datasets, composite, hybrid + warm (rff) ==="
python $s --dataset-config datasets-extended.json --objectives composite --mean rff --warm --max-rows-per-dataset 1500 --output-path output\hierarchical_oracle_test.json 2>&1 | Tee-Object output\hier_runA.log

Write-Output "=== RUN C: provoice NORMALIZED composite ==="
python $s --datasets provoice --objectives composite --normalize-objective --mean rff --warm --output-path output\hier_provoice_normalized.json 2>&1 | Tee-Object output\hier_runC.log

Write-Output "=== RUN B1: ehmi PER-CONSTRUCT ==="
python $s --datasets ehmi --objectives trust,understanding,perceived_safety,aesthetics,acceptance --mean rff --warm --output-path output\hier_perconstruct_ehmi.json 2>&1 | Tee-Object output\hier_runB1.log

Write-Output "=== RUN B2: provoice PER-CONSTRUCT ==="
python $s --datasets provoice --objectives predictability,usefulness,mental_demand --mean rff --warm --output-path output\hier_perconstruct_provoice.json 2>&1 | Tee-Object output\hier_runB2.log

Write-Output "=== FIGURE ==="
python scripts\plot_hierarchical_oracle.py --input output\hierarchical_oracle_test.json --output output\figures\hierarchical_oracle.png 2>&1 | Tee-Object output\hier_figure.log

Write-Output "ALL4_COMPLETE"

"""Exercise the actual PowerShell arm function without running experiments."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("exit_code,stale_marker", [(0, False), (23, False), (23, True)])
@pytest.mark.skipif(sys.platform != "win32", reason="Windows pipeline integration")
def test_arm_completion_requires_successful_evaluation(tmp_path, exit_code, stale_marker):
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell runtime required for the driver integration test")
    source = Path(__file__).resolve().parents[1] / "run_boba_budget_neutral.ps1"
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "evaluate_research_question.py").write_text(
        f"raise SystemExit({exit_code})\n", encoding="utf-8"
    )
    arm = tmp_path / "arm"
    (arm / "landscape").mkdir(parents=True)
    (arm / ".done-standard").touch()  # Reuse simulation; exercise only evaluation.
    if stale_marker:
        (arm / "ARM_COMPLETE").touch()
    harness = tmp_path / "harness.ps1"
    harness.write_text(r'''
param($Driver, $Interpreter)
$ErrorActionPreference = "Stop"
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Driver, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors) { throw $parseErrors[0] }
$function = $ast.Find({param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq "Run-Arm"
}, $true)
Invoke-Expression $function.Extent.Text
function Marker($dir, $variant) { Join-Path $dir ".done-standard" }
$PYTHON = $Interpreter
$Smoke = $false
$Jobs = 1
$MoJobs = 1
$log = Join-Path (Get-Location) "driver.log"
$script:Failures = @()
$ARMS = @{fake = @{Dir=(Join-Path (Get-Location) "arm"); Grid="1"; Onsets="0"; Variants=@("")}}
Run-Arm "fake" | Out-Null
@{failure_count=$script:Failures.Count; complete=(Test-Path "arm/ARM_COMPLETE")} | ConvertTo-Json -Compress
''', encoding="utf-8")
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(harness), str(source), sys.executable],
        cwd=tmp_path, capture_output=True, text=True, check=True, timeout=45,
    )
    state = json.loads(result.stdout.strip().splitlines()[-1])
    assert state["complete"] is (exit_code == 0)
    assert state["failure_count"] == (0 if exit_code == 0 else 1)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows pipeline integration")
def test_failed_driver_clears_stale_global_completion_marker(tmp_path):
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell runtime required for the driver integration test")
    source = Path(__file__).resolve().parents[1] / "run_boba_budget_neutral.ps1"
    driver = tmp_path / source.name
    shutil.copy2(source, driver)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/bo_synthetic_error_simulation.py").write_text(
        "print('fixture help')\n", encoding="utf-8"
    )
    output = tmp_path / "output-boba"
    output.mkdir()
    marker = output / "budget-neutral.DONE"
    marker.touch()
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(driver), "-Arm", "invalid-arm-fixture"],
        cwd=tmp_path, env={**os.environ, "PYTHON": sys.executable},
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert not marker.exists()

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$Python = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" } else { $null }
if (-not $Python -or -not (Test-Path $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

$Arguments = @("-B", "joint_sp\scripts\windows_preflight.py")
if ($env:JOINT_SP_MODEL) { $Arguments += @("--model", $env:JOINT_SP_MODEL) }
if ($env:JOINT_SP_DATA) { $Arguments += @("--data_dir", $env:JOINT_SP_DATA) }

& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

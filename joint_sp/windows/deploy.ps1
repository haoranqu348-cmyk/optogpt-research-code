param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Target = "broadband_high_T",
    [int]$NumCandidates = 64,
    [string]$Output = "joint_sp\deployment_result.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$Python = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" } else { $null }
if (-not $Python -or -not (Test-Path $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

& $Python -B joint_sp\scripts\windows_preflight.py --model $Model
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -B joint_sp\scripts\deploy.py `
    --model $Model `
    --target $Target `
    --num_candidates $NumCandidates `
    --output $Output
exit $LASTEXITCODE

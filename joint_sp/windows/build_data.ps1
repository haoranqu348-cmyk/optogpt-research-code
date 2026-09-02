param(
    [Parameter(Mandatory = $true)][string]$SDir,
    [Parameter(Mandatory = $true)][string]$PDir,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$Workers = 1,
    [int]$ChunkSize = 5000,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$Python = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" } else { $null }
if (-not $Python -or -not (Test-Path $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

$Arguments = @(
    "-B", "joint_sp\scripts\build_joint_data.py",
    "--s_dir", $SDir, "--p_dir", $PDir, "--out_dir", $OutDir,
    "--theta", "60", "--num_workers", $Workers,
    "--chunk_size", $ChunkSize
)
if ($Resume) { $Arguments += "--resume" }
& $Python @Arguments
exit $LASTEXITCODE

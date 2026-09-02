param(
    [Parameter(Mandatory=$true)][string]$ProjectRoot,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "=== GPU ==="
nvidia-smi

Write-Host "=== Python environment ==="
& $Python -c "import sys, torch, numpy, scipy, pandas, tmm; print(sys.version); print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('pandas', pandas.__version__)"

Write-Host "=== Checkpoint ==="
if (-not (Test-Path $Checkpoint)) { throw "Checkpoint not found: $Checkpoint" }
Get-FileHash -Algorithm SHA256 $Checkpoint

Write-Host "=== Double-sided tests ==="
& $Python -m unittest double_sided.tests.test_double_sided -v
if ($LASTEXITCODE -ne 0) { throw "Double-sided tests failed" }

Write-Host "Preflight passed."

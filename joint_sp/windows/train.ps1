param(
    [Parameter(Mandatory = $true)][string]$DataDir,
    [string]$Pretrained = "model\optogpt.pt",
    [string]$OutputName = "optogpt_60deg_sp_v1",
    [int]$Epochs = 10,
    [int]$BatchSize = 16,
    [double]$LearningRate = 0.00003,
    [int]$FusionWarmupEpochs = 2,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$Python = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" } else { $null }
if (-not $Python -or -not (Test-Path $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

& $Python -B joint_sp\scripts\windows_preflight.py --data_dir $DataDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Arguments = @(
    "-B", "joint_sp\scripts\finetune.py", "--data_dir", $DataDir,
    "--pretrained", $Pretrained, "--epochs", $Epochs,
    "--batch_size", $BatchSize, "--lr", $LearningRate,
    "--fusion_warmup_epochs", $FusionWarmupEpochs,
    "--early_stopping", "--patience", "5", "--output_name", $OutputName
)
if ($Resume) { $Arguments += "--resume" }
& $Python @Arguments
exit $LASTEXITCODE

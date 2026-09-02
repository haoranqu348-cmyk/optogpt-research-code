param(
    [Parameter(Mandatory=$true)][string]$ProjectRoot,
    [Parameter(Mandatory=$true)][string]$BaseRunRoot,
    [string]$RunRoot = "",
    [string]$Python = "python",
    [int]$GpuIndex = 1,
    [int]$Stage58Samples = 150000,
    [int]$Stage916Samples = 200000,
    [int]$ChunkSize = 2000,
    [int]$Workers = 8,
    [int]$BatchSize = 16,
    [int]$GradAccumSteps = 4,
    [int]$Max8Epochs = 10,
    [int]$Max16Epochs = 10,
    [int]$EvaluationCandidates = 256,
    [int]$RobustnessRandomTrials = 100,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$BaseRunRoot = (Resolve-Path $BaseRunRoot).Path
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    if ($Resume) { throw "-Resume requires the existing -RunRoot path" }
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunRoot = Join-Path $ProjectRoot "results\double_sided_inverse_design\${Stamp}_base_material_max8_max16"
} elseif ((Test-Path $RunRoot) -and -not $Resume) {
    throw "RunRoot already exists. Use -Resume to continue it: $RunRoot"
}
$env:CUDA_VISIBLE_DEVICES = "$GpuIndex"

$BaseData = Join-Path $BaseRunRoot "data_stage_1_4"
$BaseModel = Join-Path $BaseRunRoot "models\phase_C\best_physical.pt"
$HistoricalElite = Join-Path $ProjectRoot "results\double_sided_glass_study\20260728_60deg_formal_v2\all_rankings.csv"
$Data58Raw = Join-Path $RunRoot "data_raw_5_8"
$Data58 = Join-Path $RunRoot "data_max8"
$Data916Raw = Join-Path $RunRoot "data_raw_9_16"
$Data916 = Join-Path $RunRoot "data_max16"
$Model8 = Join-Path $RunRoot "models\max8"
$Model16 = Join-Path $RunRoot "models\max16"
$Eval8 = Join-Path $RunRoot "evaluation_max8"
$Eval16 = Join-Path $RunRoot "evaluation_max16"
$Robust8 = Join-Path $RunRoot "robustness_max8"
$Robust16 = Join-Path $RunRoot "robustness_max16"

function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

function Test-CompleteFile {
    param([string]$Path)
    return (Test-Path $Path) -and ((Get-Item $Path).Length -gt 0)
}

function Move-IncompleteDirectory {
    param([string]$Path)
    if (Test-Path $Path) {
        $Backup = $Path + ".incomplete_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Move-Item $Path $Backup
        Write-Host "Preserved incomplete output as: $Backup"
    }
}

function Invoke-DataGeneration {
    param(
        [string]$Output,
        [int]$Samples,
        [int]$MinimumLayers,
        [int]$MaximumLayers,
        [int]$Seed,
        [string]$EliteCsv
    )
    $Arguments = @(
        "-m", "double_sided.scripts.generate_formal_data", "--output", $Output,
        "--samples", "$Samples", "--chunk-size", "$ChunkSize", "--workers", "$Workers",
        "--stage-min-layers", "$MinimumLayers", "--stage-max-layers", "$MaximumLayers",
        "--seed", "$Seed", "--elite-csv", $EliteCsv
    )
    if ($Resume -and (Test-Path (Join-Path $Output "generation_contract.json"))) {
        $Arguments += "--resume"
    }
    Invoke-Python @Arguments
    Invoke-Python -m double_sided.scripts.finalize_formal_data --data-dir $Output
}

function Invoke-DatasetMerge {
    param([string]$Base, [string]$Extra, [string]$Output)
    $Contract = Join-Path $Output "dataset_contract.json"
    if ($Resume -and (Test-CompleteFile $Contract)) {
        Write-Host "Verified completed merge, skipping: $Output"
        return
    }
    Move-IncompleteDirectory $Output
    Invoke-Python -m double_sided.scripts.merge_formal_datasets `
        --base $Base --extra $Extra --output $Output
}

function Invoke-Training {
    param(
        [string]$InitializedCheckpoint,
        [string]$DataDir,
        [string]$OutputDir,
        [int]$Epochs,
        [int]$MaximumLayers,
        [double]$LearningRate
    )
    $Best = Join-Path $OutputDir "best_physical.pt"
    $Latest = Join-Path $OutputDir "latest.pt"
    $Log = Join-Path $OutputDir "training_log.json"
    if ($Resume -and (Test-CompleteFile $Best) -and (Test-CompleteFile $Log)) {
        $Rows = Get-Content $Log -Raw | ConvertFrom-Json
        if (@($Rows).Count -ge $Epochs) {
            Write-Host "Verified completed training, skipping: $OutputDir"
            return
        }
    }
    $Arguments = @(
        "-m", "double_sided.scripts.train",
        "--initialized-checkpoint", $InitializedCheckpoint,
        "--data-dir", $DataDir, "--output-dir", $OutputDir,
        "--epochs", "$Epochs", "--batch-size", "$BatchSize",
        "--grad-accum-steps", "$GradAccumSteps", "--lr", "$LearningRate",
        "--max-layers-per-side", "$MaximumLayers", "--phase", "C",
        "--smoothing", "0.1", "--physical-eval-every", "1",
        "--physical-eval-candidates", "32", "--amp", "--allow-base-only"
    )
    if ($Resume -and (Test-CompleteFile $Latest)) {
        $Arguments += @("--resume-from", $Latest)
    } elseif (Test-Path $OutputDir) {
        if (-not $Resume) { throw "Training output already exists: $OutputDir" }
        Move-IncompleteDirectory $OutputDir
    }
    Invoke-Python @Arguments
    if (-not (Test-CompleteFile $Best)) { throw "Training produced no best_physical.pt: $OutputDir" }
}

function Invoke-Evaluation {
    param([string]$Checkpoint, [string]$OutputDir, [int]$MaximumLayers, [int]$Seed)
    $Manifest = Join-Path $OutputDir "manifest.json"
    if ($Resume -and (Test-CompleteFile $Manifest)) {
        Write-Host "Verified completed evaluation, skipping: $OutputDir"
        return
    }
    Move-IncompleteDirectory $OutputDir
    Invoke-Python -m double_sided.scripts.evaluate_model `
        --checkpoint $Checkpoint --output-dir $OutputDir `
        --candidates $EvaluationCandidates --decode-batch-size 64 --temperature 0.9 `
        --top-k 32 --de-top 20 --de-maxiter 30 --de-popsize 6 `
        --max-layers-per-side $MaximumLayers --seed $Seed
}

function Invoke-Robustness {
    param([string]$EvaluationDir, [string]$OutputDir, [int]$MaximumLayers, [int]$Seed)
    $Manifest = Join-Path $OutputDir "manifest.json"
    if ($Resume -and (Test-CompleteFile $Manifest)) {
        Write-Host "Verified completed robustness, skipping: $OutputDir"
        return
    }
    Move-IncompleteDirectory $OutputDir
    Invoke-Python -m double_sided.scripts.evaluate_top20_robustness `
        --model-evaluation-dir $EvaluationDir --output-dir $OutputDir `
        --random-trials $RobustnessRandomTrials --max-layers-per-side $MaximumLayers --seed $Seed
}

foreach ($Required in @($BaseData, $BaseModel, $HistoricalElite)) {
    if (-not (Test-Path $Required)) { throw "Required input not found: $Required" }
}
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$RunContractPath = Join-Path $RunRoot "run_contract.json"
$RunContract = [ordered]@{
    schema_version = 1
    project_root = $ProjectRoot
    base_run_root = $BaseRunRoot
    stage_5_8_samples = $Stage58Samples
    stage_9_16_samples = $Stage916Samples
    chunk_size = $ChunkSize
    batch_size = $BatchSize
    grad_accum_steps = $GradAccumSteps
    max8_epochs = $Max8Epochs
    max16_epochs = $Max16Epochs
    evaluation_candidates = $EvaluationCandidates
    robustness_random_trials = $RobustnessRandomTrials
}
if (Test-Path $RunContractPath) {
    $ExistingContract = Get-Content $RunContractPath -Raw | ConvertFrom-Json
    foreach ($Key in $RunContract.Keys) {
        if ("$($ExistingContract.$Key)" -ne "$($RunContract[$Key])") {
            throw "Resume contract mismatch for ${Key}: $($ExistingContract.$Key) != $($RunContract[$Key])"
        }
    }
} else {
    $RunContract | ConvertTo-Json | Set-Content $RunContractPath -Encoding UTF8
}
$Transcript = Join-Path $RunRoot ("run_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
Start-Transcript -Path $Transcript
$Succeeded = $false
try {
    Write-Host "Run root: $RunRoot"
    Write-Host "Physical GPU index: $GpuIndex (visible as cuda:0)"
    nvidia-smi

    Invoke-DataGeneration $Data58Raw $Stage58Samples 5 8 20260730 $HistoricalElite
    Invoke-DatasetMerge $BaseData $Data58Raw $Data58
    Invoke-Training $BaseModel $Data58 $Model8 $Max8Epochs 8 0.00001
    $Best8 = Join-Path $Model8 "best_physical.pt"
    Invoke-Evaluation $Best8 $Eval8 8 20260730
    Invoke-Robustness $Eval8 $Robust8 8 20260730

    $Max8Elite = Join-Path $Eval8 "rankings.csv"
    Invoke-DataGeneration $Data916Raw $Stage916Samples 9 16 20260731 $Max8Elite
    Invoke-DatasetMerge $Data58 $Data916Raw $Data916
    Invoke-Training $Best8 $Data916 $Model16 $Max16Epochs 16 0.00001
    $Best16 = Join-Path $Model16 "best_physical.pt"
    Invoke-Evaluation $Best16 $Eval16 16 20260731
    Invoke-Robustness $Eval16 $Robust16 16 20260731

    Invoke-Python -m double_sided.scripts.compare_model_evaluations `
        --max8-evaluation $Eval8 --max16-evaluation $Eval16 `
        --max8-robustness $Robust8 --max16-robustness $Robust16 `
        --output (Join-Path $RunRoot "max8_vs_max16.json")

    $Completion = @{
        status = "complete"
        completed_at = (Get-Date).ToString("o")
        base_run_root = $BaseRunRoot
        max8_checkpoint = $Best8
        max16_checkpoint = $Best16
        comparison = (Join-Path $RunRoot "max8_vs_max16.json")
    }
    $Completion | ConvertTo-Json | Set-Content (Join-Path $RunRoot "RUN_COMPLETE.json") -Encoding UTF8
    $Succeeded = $true
}
finally {
    Stop-Transcript
}
if ($Succeeded) {
    Get-ChildItem -Recurse -File $RunRoot | Where-Object Name -ne "SHA256SUMS.csv" |
        Get-FileHash -Algorithm SHA256 |
        Export-Csv -NoTypeInformation (Join-Path $RunRoot "SHA256SUMS.csv")
    Write-Host "Complete max8/max16 run: $RunRoot"
}

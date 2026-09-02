param(
    [Parameter(Mandatory=$true)][string]$ProjectRoot,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$Python = "python",
    [int]$Samples = 200000,
    [int]$ChunkSize = 2000,
    [int]$Workers = 8,
    [int]$BatchSize = 16,
    [int]$GradAccumSteps = 2,
    [int]$PhaseAEpochs = 3,
    [int]$PhaseBEpochs = 5,
    [int]$PhaseCEpochs = 10,
    [switch]$ResumeData
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $ProjectRoot "results\double_sided_inverse_design\${Stamp}_base_material_stage_1_4"
$InitDir = Join-Path $RunRoot "checkpoint_init"
$DataDir = Join-Path $RunRoot "data_stage_1_4"
$ModelsDir = Join-Path $RunRoot "models"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

Start-Transcript -Path (Join-Path $RunRoot "formal_run.log")
try {
    Write-Host "Run root: $RunRoot"
    nvidia-smi

    & $Python -m double_sided.scripts.stage_checkpoint `
        --checkpoint $Checkpoint --output-dir $InitDir --device cpu
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint staging failed" }

    $DataArgs = @(
        "-m", "double_sided.scripts.generate_formal_data",
        "--output", $DataDir, "--samples", $Samples,
        "--chunk-size", $ChunkSize, "--workers", $Workers,
        "--stage-min-layers", 1, "--stage-max-layers", 4,
        "--seed", 20260728,
        "--elite-csv", (Join-Path $ProjectRoot "results\double_sided_glass_study\20260728_60deg_formal_v2\all_rankings.csv")
    )
    if ($ResumeData) { $DataArgs += "--resume" }
    & $Python @DataArgs
    if ($LASTEXITCODE -ne 0) { throw "Data generation failed" }

    & $Python -m double_sided.scripts.finalize_formal_data --data-dir $DataDir
    if ($LASTEXITCODE -ne 0) { throw "Data finalization failed" }

    $Initialized = Join-Path $InitDir "double_sided_initialized.pt"
    $PhaseA = Join-Path $ModelsDir "phase_A"
    & $Python -m double_sided.scripts.train `
        --initialized-checkpoint $Initialized --data-dir $DataDir --output-dir $PhaseA `
        --epochs $PhaseAEpochs --batch-size $BatchSize --grad-accum-steps $GradAccumSteps `
        --lr 3e-5 --max-layers-per-side 4 --phase A --smoothing 0.1 `
        --physical-eval-every 1 --physical-eval-candidates 32 --amp --allow-base-only
    if ($LASTEXITCODE -ne 0) { throw "Phase A training failed" }

    $PhaseABest = Join-Path $PhaseA "best_physical.pt"
    $PhaseB = Join-Path $ModelsDir "phase_B"
    & $Python -m double_sided.scripts.train `
        --initialized-checkpoint $PhaseABest `
        --data-dir $DataDir --output-dir $PhaseB `
        --epochs $PhaseBEpochs --batch-size $BatchSize --grad-accum-steps $GradAccumSteps `
        --lr 2e-5 --max-layers-per-side 4 --phase B --smoothing 0.1 `
        --physical-eval-every 1 --physical-eval-candidates 32 --amp --allow-base-only
    if ($LASTEXITCODE -ne 0) { throw "Phase B training failed" }

    $PhaseBBest = Join-Path $PhaseB "best_physical.pt"
    $PhaseC = Join-Path $ModelsDir "phase_C"
    & $Python -m double_sided.scripts.train `
        --initialized-checkpoint $PhaseBBest `
        --data-dir $DataDir --output-dir $PhaseC `
        --epochs $PhaseCEpochs --batch-size $BatchSize --grad-accum-steps $GradAccumSteps `
        --lr 1e-5 --max-layers-per-side 4 --phase C --smoothing 0.1 `
        --physical-eval-every 1 --physical-eval-candidates 32 --amp --allow-base-only
    if ($LASTEXITCODE -ne 0) { throw "Phase C training failed" }

    $PhaseCBest = Join-Path $PhaseC "best_physical.pt"
    $EvaluationDir = Join-Path $RunRoot "evaluation_model_topk_de"
    & $Python -m double_sided.scripts.evaluate_model `
        --checkpoint $PhaseCBest `
        --output-dir $EvaluationDir --candidates 256 --decode-batch-size 64 `
        --temperature 0.9 --top-k 32 --de-top 20 --de-maxiter 30 --de-popsize 6 `
        --max-layers-per-side 4 --seed 20260728
    if ($LASTEXITCODE -ne 0) { throw "Model TMM/DE evaluation failed" }

    $ComparisonDir = Join-Path $RunRoot "comparison_equal_budget"
    & $Python -m double_sided.scripts.compare_equal_budget `
        --model-evaluation-dir $EvaluationDir --joint-checkpoint $Checkpoint `
        --output-dir $ComparisonDir --de-maxiter 5 --de-popsize 3 --seed 20260728
    if ($LASTEXITCODE -ne 0) { throw "Equal-budget comparison failed" }

    $RobustnessDir = Join-Path $RunRoot "robustness_top20"
    & $Python -m double_sided.scripts.evaluate_top20_robustness `
        --model-evaluation-dir $EvaluationDir --output-dir $RobustnessDir `
        --random-trials 100 --max-layers-per-side 4 --seed 20260728
    if ($LASTEXITCODE -ne 0) { throw "Top-20 robustness failed" }

    Get-ChildItem -Recurse -File $RunRoot | Get-FileHash -Algorithm SHA256 |
        Export-Csv -NoTypeInformation (Join-Path $RunRoot "SHA256SUMS.csv")
    Write-Host "Stage 1-4 complete: $RunRoot"
}
finally {
    Stop-Transcript
}

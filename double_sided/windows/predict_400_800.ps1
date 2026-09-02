param(
    [Parameter(Mandatory=$true)][string]$ProjectRoot,
    [string]$Checkpoint = "",
    [string]$Python = "python",
    [int]$GpuIndex = 1,
    [int]$Candidates = 2048,
    [int]$DeTop = 64,
    [int]$DeMaxIter = 80,
    [int]$DePopSize = 10,
    [int]$Seed = 20260801
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$env:CUDA_VISIBLE_DEVICES = "$GpuIndex"

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $Checkpoint = Get-ChildItem `
        (Join-Path $ProjectRoot "results\double_sided_inverse_design") `
        -Filter "best_physical.pt" -File -Recurse |
        Where-Object FullName -Like "*base_material_max8_max16*models\max16\best_physical.pt" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if ([string]::IsNullOrWhiteSpace($Checkpoint) -or -not (Test-Path $Checkpoint)) {
    throw "No max16 best_physical.pt checkpoint was found"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $ProjectRoot `
    "results\double_sided_inverse_design\${Stamp}_max16_prediction_400_800nm"

Write-Host "Checkpoint: $Checkpoint"
Write-Host "Output: $OutputDir"
Write-Host "Physical GPU index: $GpuIndex (visible as cuda:0)"

& $Python -m double_sided.scripts.predict_band `
    --checkpoint $Checkpoint `
    --output-dir $OutputDir `
    --wavelength-min 400 `
    --wavelength-max 800 `
    --out-of-band-reflectances "0,0.05,0.15,0.30" `
    --candidates $Candidates `
    --decode-batch-size 64 `
    --temperature 0.9 `
    --top-k 32 `
    --de-top $DeTop `
    --de-maxiter $DeMaxIter `
    --de-popsize $DePopSize `
    --de-search-stride 4 `
    --max-layers-per-side 16 `
    --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "400-800 nm prediction failed" }

Set-Content `
    (Join-Path $ProjectRoot "results\double_sided_inverse_design\latest_400_800_prediction.txt") `
    $OutputDir -Encoding UTF8
Get-ChildItem -Recurse -File $OutputDir | Get-FileHash -Algorithm SHA256 |
    Export-Csv -NoTypeInformation (Join-Path $OutputDir "SHA256SUMS.csv")

Write-Host "Prediction complete: $OutputDir"
Write-Host "Best structure: $(Join-Path $OutputDir 'best_structure.json')"
Write-Host "Full spectrum: $(Join-Path $OutputDir 'best_spectrum_400_1100.csv')"


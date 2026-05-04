# Hyperparameter search script - ETF

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$START_TIME = Get-Date

# Activate virtual environment
if (Test-Path ".venv") {
    & ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv") {
    & "venv\Scripts\Activate.ps1"
}

# Models to search
$SEARCH_MODEL_TYPES = @("tcn", "itransformer")

# Common config
$CONFIG_NAME = "config_etf"
$SEQUENCE_LENGTH = 60
$FEATURE_NUM = "39"
$TOPK = 3
$DATA_FILE = "etf_74.csv"

foreach ($MODEL_TYPE in $SEARCH_MODEL_TYPES) {
    Write-Host "========================================"
    Write-Host "Running search for model type: $MODEL_TYPE"
    Write-Host "========================================"

    # Compute search dir from config values
    $SEARCH_DIR = "./etf_model/search_${MODEL_TYPE}_${N}_${TOPK}"
    New-Item -ItemType Directory -Force -Path $SEARCH_DIR | Out-Null

    Write-Host "Config: $CONFIG_NAME"
    Write-Host "Output directory: $SEARCH_DIR"
    Write-Host "DATA_FILE: $DATA_FILE"
    Write-Host "TOPK: $TOPK"
    Write-Host "FEATURE_NUM: $FEATURE_NUM"
    Write-Host ""

    # Check for preprocessed data
    if (Test-Path "$SEARCH_DIR/preprocessed_data.pkl") {
        Write-Host "Found preprocessed data, using --resume"
        $RESUME = "--resume"
    } else {
        $RESUME = ""
    }

    # Set CUDA memory allocation strategy
    $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

    # Run search with CLI args (no config file modification needed)
    python code/src/train_search_v2.py `
        --config $CONFIG_NAME `
        $RESUME `
        --model-type $MODEL_TYPE `
        --feature-num $FEATURE_NUM `
        --data-file $DATA_FILE `
        --topk $TOPK `
        --sequence-length $SEQUENCE_LENGTH `
        --N $N

    $RESULTS_FILE = "$SEARCH_DIR/search_results.json"

    if (Test-Path $RESULTS_FILE) {
        Write-Host ""
        Write-Host "=================="
        Write-Host "Search completed for $MODEL_TYPE!"
        Write-Host ""
        Write-Host "Top 5 results:"
        python -c "
import json
with open('$RESULTS_FILE') as f:
    results = json.load(f)
sorted_results = sorted([r for r in results if r.get('success')], key=lambda x: x.get('score', 0), reverse=True)
for i, r in enumerate(sorted_results[:5]):
    params = r['params']
    print(f\"{i+1}. LR={params['learning_rate']}, DM={params['d_model']}, NL={params['num_layers']}, DP={params['dropout']} -> {r['score']:.4f}\")
"
    }

    Write-Host ""
    Write-Host "Finished $MODEL_TYPE, moving to next..."
    Write-Host ""
}

# Calculate total elapsed time
$END_TIME = Get-Date
$ELAPSED = ($END_TIME - $START_TIME).TotalSeconds
$HOURS = [math]::Floor($ELAPSED / 3600)
$MINUTES = [math]::Floor(($ELAPSED % 3600) / 60)
$SECONDS = [math]::Floor($ELAPSED % 60)

Write-Host "========================================"
Write-Host "All searches completed!"
Write-Host "========================================"
Write-Host "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s ($([math]::Floor($ELAPSED)) seconds)"

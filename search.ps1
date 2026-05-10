# 超参搜索脚本 - ETF

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$START_TIME = Get-Date

# 激活虚拟环境
if (Test-Path ".venv") {
    & ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv") {
    & "venv\Scripts\Activate.ps1"
}

# 要搜索的模型类型 (空格分隔)
$SEARCH_MODEL_TYPES = @("itransformer","dlinear","lstm")

# 通用配置
$CONFIG_NAME = "config"
$SEQUENCE_LENGTH = 60
$FEATURE_NUM = "39"
$TOPK = 3
$DATA_FILE = "etf_74_train.csv"
$SEARCH_METHOD = "bayesian"   # "bayesian" 或 "grid"
$N_TRIALS = 50                # 贝叶斯搜索的试验次数 (仅 bayesian 模式生效)
$SEARCH_METRIC = "ndcg"       # 优化指标 (ndcg/mrr/excess_return/final_score)

foreach ($MODEL_TYPE in $SEARCH_MODEL_TYPES) {
    Write-Host "========================================"
    Write-Host "Running search for model type: $MODEL_TYPE"
    Write-Host "========================================"

    # 计算搜索目录 (与 train_search_v2.py 逻辑一致)
    $N = 74
    $METHOD_PREFIX = if ($SEARCH_METHOD -eq "grid") { "grid" } else { "bayes" }
    $SEARCH_DIR = "./model/${METHOD_PREFIX}_${MODEL_TYPE}_${N}_${TOPK}"

    Write-Host "Config: $CONFIG_NAME"
    Write-Host "Output directory: $SEARCH_DIR"
    Write-Host "DATA_FILE: $DATA_FILE"
    Write-Host "TOPK: $TOPK"
    Write-Host "FEATURE_NUM: $FEATURE_NUM"
    Write-Host ""

    # 检查是否已有预处理数据（兼容旧 search_ 前缀）
    $OLD_SEARCH_DIR = "./model/search_${MODEL_TYPE}_${N}_${TOPK}"
    if (Test-Path "$SEARCH_DIR/preprocessed_data.pkl") {
        Write-Host "Found preprocessed data, using --resume"
        $RESUME = "--resume"
    } elseif (Test-Path "$OLD_SEARCH_DIR/preprocessed_data.pkl") {
        Write-Host "Found preprocessed data (legacy search_ dir), using --resume"
        $RESUME = "--resume"
    } else {
        $RESUME = ""
    }

    # 设置CUDA内存分配策略
    $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

    # 运行搜索 (通过CLI参数传递配置, 不修改config文件)
    python code/src/train_search_v2.py `
        --config $CONFIG_NAME `
        $RESUME `
        --model-type $MODEL_TYPE `
        --feature-num $FEATURE_NUM `
        --data-file $DATA_FILE `
        --topk $TOPK `
        --sequence-length $SEQUENCE_LENGTH `
        --N $N `
        --search-method $SEARCH_METHOD `
        --n-trials $N_TRIALS `
        --search-metric $SEARCH_METRIC

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
    print(f'{i+1}. LR={params[\"learning_rate\"]}, DM={params[\"d_model\"]}, NL={params[\"num_layers\"]}, DP={params[\"dropout\"]} -> {r[\"score\"]:.4f}')
"
    }

    Write-Host ""
    Write-Host "Finished $MODEL_TYPE, moving to next..."
    Write-Host ""
}

# 计算总用时
$END_TIME = Get-Date
$ELAPSED = ($END_TIME - $START_TIME).TotalSeconds
$HOURS = [math]::Floor($ELAPSED / 3600)
$MINUTES = [math]::Floor(($ELAPSED % 3600) / 60)
$SECONDS = [math]::Floor($ELAPSED % 60)

Write-Host "========================================"
Write-Host "All searches completed!"
Write-Host "========================================"
Write-Host "总用时: ${HOURS}h ${MINUTES}m ${SECONDS}s ($([math]::Floor($ELAPSED))秒)"

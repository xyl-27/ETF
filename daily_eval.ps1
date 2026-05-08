param(
    [Parameter(Position=0)]
    [ValidateSet("Init", "Update", "Run")]
    [string]$Mode = "Update",

    [string]$StartDate,
    [switch]$NoUpdate,
    [int]$TopK = 3,
    [int]$RebalanceDays = 5,
    [double]$PositionPct = 0.95,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1. 激活虚拟环境
if (Test-Path ".venv") {
    & ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv") {
    & "venv\Scripts\Activate.ps1"
}

# 2. 模式逻辑处理
$PYTHON_ARGS = @("code/src/daily_eval.py", "--config=config")

if ($Mode -eq "Init") {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  ETF 策略初始化 (Init)" -ForegroundColor Cyan
    Write-Host "========================================"

    # 清理旧状态文件
    $filesToRemove = @(
        "output\backtest_state.json",
        "output\latest_report.json",
        "output\latest_report.html",
        "output\equity_curves.png",
        "output\portfolio.json",
        "output\history_report\*.html"
    )
    foreach ($f in $filesToRemove) {
        Remove-Item $f -ErrorAction SilentlyContinue
        $displayPath = $f -replace [regex]::Escape("\*.html"), " (历史报告)"
        Write-Host "  [清理] 删除 $displayPath" -ForegroundColor DarkYellow
    }
    Write-Host ""

    # 初始化默认起始日 (通常设为数据可用的起始点)
    if (-not $PSBoundParameters.ContainsKey('StartDate')) {
        $StartDate = "2026-04-01"
    }

    # 初始化通常强制获取最新数据 (除非显式指定 -NoUpdate)
    if ($NoUpdate) {
        Write-Host "  [跳过] 数据更新 (用户指定)" -ForegroundColor Yellow
    }

} else {
    # Update / Run Mode
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  ETF 每日更新 (Update)" -ForegroundColor Cyan
    Write-Host "========================================"

    # 检查是否已初始化
    $STATE_PATH = "output\backtest_state.json"
    if (-not (Test-Path $STATE_PATH)) {
        Write-Host "  [提示] 未检测到状态文件，建议先运行 Init 初始化" -ForegroundColor Yellow
        Write-Host "         .\daily_eval.ps1 Init" -ForegroundColor Yellow
        Write-Host ""
    }

    # 更新模式默认起始日
    if (-not $PSBoundParameters.ContainsKey('StartDate')) {
        $StartDate = "2026-04-01"
    }
}

Write-Host "  [起始日] $StartDate"
Write-Host "  [Top-K]  $TopK"
Write-Host ""

# 3. 构建 Python 参数
if ($StartDate) {
    $PYTHON_ARGS += "--start-date"
    $PYTHON_ARGS += $StartDate
}

if ($NoUpdate) {
    $PYTHON_ARGS += "--no-update"
}

if ($Quiet) {
    $PYTHON_ARGS += "--quiet"
}

$PYTHON_ARGS += "--topk"
$PYTHON_ARGS += $TopK.ToString()

$PYTHON_ARGS += "--rebalance-days"
$PYTHON_ARGS += $RebalanceDays.ToString()

$PYTHON_ARGS += "--position-pct"
$PYTHON_ARGS += $PositionPct.ToString()

# 4. 显示当前持仓状态
$PORTFOLIO_PATH = "output/portfolio.json"
if (Test-Path $PORTFOLIO_PATH) {
    Write-Host "----------------------------------------" -ForegroundColor Gray
    try {
        $portfolio = Get-Content $PORTFOLIO_PATH -Raw | ConvertFrom-Json
        Write-Host "  [当前持仓] $($portfolio.predict_date) | 总值: $($portfolio.total_value)" -ForegroundColor Gray
        foreach ($h in $portfolio.holdings) {
            Write-Host "    $($h.stock_id) ($($h.shares)股)" -ForegroundColor Gray
        }
    } catch {
        Write-Host "  [读取持仓失败]" -ForegroundColor Gray
    }
    Write-Host "----------------------------------------" -ForegroundColor Gray
    Write-Host ""
}

# 5. 执行
Write-Host "执行中..." -ForegroundColor Green
python @PYTHON_ARGS

$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "完成" -ForegroundColor Green
} else {
    Write-Host "失败 (exit code: $exitCode)" -ForegroundColor Red
}
exit $exitCode

param(
    [switch]$NoUpdate,
    [int]$TopK = 5,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 激活虚拟环境
if (Test-Path ".venv") {
    & ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv") {
    & "venv\Scripts\Activate.ps1"
}

# 构建参数
$PYTHON_ARGS = @("code/src/daily_eval.py", "--config=config")

if ($NoUpdate) { $PYTHON_ARGS += "--no-update" }
if ($Quiet) { $PYTHON_ARGS += "--quiet" }
$PYTHON_ARGS += "--topk"
$PYTHON_ARGS += $TopK.ToString()

# 显示当前持仓
$PORTFOLIO_PATH = "output/portfolio.json"
if (Test-Path $PORTFOLIO_PATH) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  当前持仓" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    $portfolio = Get-Content $PORTFOLIO_PATH -Raw | ConvertFrom-Json
    Write-Host "  上次更新: $($portfolio.last_updated)"
    Write-Host "  预测日期: $($portfolio.predict_date)"
    Write-Host "  持有 $($portfolio.holdings.Count) 只:"
    foreach ($h in $portfolio.holdings) {
        Write-Host "    $($h.stock_id) (买入: $($h.buy_date))"
    }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ETF每日测评 $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python @PYTHON_ARGS

$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "测评完成" -ForegroundColor Green
} else {
    Write-Host "测评失败 (exit code: $exitCode)" -ForegroundColor Red
}
exit $exitCode

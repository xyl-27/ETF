# 每日测评脚本
# 用法:
#   .\daily_eval.ps1                          # 默认测评
#   .\daily_eval.ps1 --no-update              # 跳过数据更新
#   .\daily_eval.ps1 --backtest-months 3      # 回测3个月
#   .\daily_eval.ps1 --topk 3                 # Top-3推荐

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

if ($args.Count -gt 0) {
    foreach ($arg in $args) {
        $PYTHON_ARGS += $arg
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  每日测评 $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Cyan
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

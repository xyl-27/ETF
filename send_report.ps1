param(
    [string]$To = $null
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
$PYTHON_ARGS = @("code/src/send_report.py")

if ($To) {
    $PYTHON_ARGS += "--to"
    $PYTHON_ARGS += $To
}

# 检查报告是否存在
if (-not (Test-Path "output\latest_report.json")) {
    Write-Host "错误: 未找到 output\latest_report.json" -ForegroundColor Red
    Write-Host "请先运行 .\daily_eval.ps1 Init 或 Update 生成报告" -ForegroundColor Yellow
    exit 1
}

# 检查环境变量
if (-not $env:SMTP_USER -or -not $env:SMTP_PASSWORD) {
    Write-Host "错误: 未设置 SMTP 环境变量" -ForegroundColor Red
    Write-Host "请设置: SMTP_USER, SMTP_PASSWORD, SMTP_SERVER (可选), SMTP_PORT (可选)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "示例 (QQ邮箱):" -ForegroundColor Cyan
    Write-Host '  $env:SMTP_USER="your_email@qq.com"' -ForegroundColor Cyan
    Write-Host '  $env:SMTP_PASSWORD="your_auth_code"' -ForegroundColor Cyan
    exit 1
}

Write-Host "发送 ETF 测评报告..." -ForegroundColor Cyan
python @PYTHON_ARGS

if ($LASTEXITCODE -eq 0) {
    Write-Host "完成" -ForegroundColor Green
} else {
    Write-Host "失败 (exit code: $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

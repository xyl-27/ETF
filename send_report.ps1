param(
    [string]$To = $null,
    [string]$ModelKey = $null
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 激活虚拟环境
if (Test-Path ".venv") {
    & ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv") {
    & "venv\Scripts\Activate.ps1"
}

# 自动获取最佳模型 (从 model_selection.txt 读取第一行)
if (-not $ModelKey -and (Test-Path "output\model_selection.txt")) {
    $lines = Get-Content "output\model_selection.txt" | Where-Object { $_ -notmatch "^[#\s]*$" }
    if ($lines.Count -ge 2) {
        # 第二行开始是模型路径，格式: model_type/exp_X
        $firstModelLine = $lines[1].Trim()
        if ($firstModelLine) {
            # 提取路径中的 exp_X 部分和上一层目录名
            $pathParts = $firstModelLine -split '[\\/]+'
            $expName = $pathParts[-1]
            $modelType = $pathParts[-2]
            if ($modelType -match "^search_") { $modelType = $modelType.Substring(7) }
            $modelType = $modelType -replace '_\d+_\d+', ''
            $ModelKey = "${modelType}_${expName}"
            Write-Host "[自动] 检测到最佳模型: $ModelKey" -ForegroundColor Yellow
        }
    }
}

# 构建参数
$PYTHON_ARGS = @("code/src/send_report.py")

if ($ModelKey) {
    $PYTHON_ARGS += "--model-key"
    $PYTHON_ARGS += $ModelKey
}

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
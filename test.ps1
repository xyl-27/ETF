# 测试脚本 - 预测ETF排序
# 用法:
#   .\test.ps1                              # 使用 output/model_selection.txt 自动选择模型
#   .\test.ps1 -Select                      # 先运行模型选择, 再预测
#   .\test.ps1 -Select -TopN 5              # 选择最好的5个模型融合
#   .\test.ps1 -Select -Mode single         # 只选最好的1个模型
#   .\test.ps1 -Exp exp_12                  # 指定单个实验(覆盖自动选择)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 激活虚拟环境
if (Test-Path ".venv") {
    & ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv") {
    & "venv\Scripts\Activate.ps1"
}

$CONFIG_NAME = "config"
$PYTHON_ARGS = @("code/src/predict.py", "--config", $CONFIG_NAME)

# 解析参数
param(
    [switch]$Select,
    [int]$TopN = 10,
    [string]$Mode = "fusion",
    [string]$Exp = "",
    [string[]]$Manual = @()
)

# 如果指定了 -Select, 先运行模型选择
if ($Select) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  模型选择" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    $SELECT_ARGS = @("code/src/select_models.py", "--top-n", $TopN.ToString(), "--mode", $Mode)
    if ($Manual.Count -gt 0) {
        $SELECT_ARGS += "--manual"
        $SELECT_ARGS += $Manual
    }

    python @SELECT_ARGS
    if ($LASTEXITCODE -ne 0) {
        Write-Host "模型选择失败" -ForegroundColor Red
        exit 1
    }
    Write-Host ""
}

# 如果指定了 -Exp, 添加 --exp 参数
if ($Exp -ne "") {
    $PYTHON_ARGS += "--exp"
    $PYTHON_ARGS += $Exp
}

# 显示使用的模型信息
$SELECTION_PATH = "output/model_selection.txt"
if (Test-Path $SELECTION_PATH) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  使用的模型" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Get-Content $SELECTION_PATH | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  开始预测" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python @PYTHON_ARGS

$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "预测完成, 结果已保存到 output/result.csv" -ForegroundColor Green
} else {
    Write-Host "预测失败 (exit code: $exitCode)" -ForegroundColor Red
}
exit $exitCode

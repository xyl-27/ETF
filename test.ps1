# 测试脚本 - 预测HS300股票排序

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 激活虚拟环境
if (Test-Path ".venv") {
    & ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv") {
    & "venv\Scripts\Activate.ps1"
}

$CONFIG_NAME = "config"

# 读取输出目录
$OUTPUT_DIR = python -c "
import sys
sys.path.insert(0, 'code/src')
import importlib
m = importlib.import_module('$CONFIG_NAME')
print(m.config.get('output_dir', './model/default'))
"

Write-Host "Using config: $CONFIG_NAME"
Write-Host "Output directory: $OUTPUT_DIR"
Write-Host ""

# 支持 --exp 参数
$EXP_ARG = ""
if ($args.Count -gt 0) {
    $EXP_ARG = "--exp=$($args[0])"
}

python code/src/predict.py --config=$CONFIG_NAME $EXP_ARG

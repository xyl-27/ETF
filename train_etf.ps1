# 训练脚本 - ETF

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 激活虚拟环境
if (Test-Path ".venv") {
    & ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv") {
    & "venv\Scripts\Activate.ps1"
}

$CONFIG_NAME = "config_etf"

# 读取输出目录
$OUTPUT_DIR = python -c "
import sys
sys.path.insert(0, 'code/src')
import importlib
m = importlib.import_module('$CONFIG_NAME')
print(m.config.get('output_dir', './etf_model/default'))
"

Write-Host "Using config: $CONFIG_NAME"
Write-Host "Output directory: $OUTPUT_DIR"
Write-Host ""

python code/src/train.py --config $CONFIG_NAME

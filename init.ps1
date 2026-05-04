# 初始化目录结构 - ETF
Write-Host "初始化环境..."

# 创建必要的目录
New-Item -ItemType Directory -Force -Path "code\src" | Out-Null
New-Item -ItemType Directory -Force -Path "etf_model" | Out-Null
New-Item -ItemType Directory -Force -Path "output" | Out-Null
New-Item -ItemType Directory -Force -Path "temp" | Out-Null
New-Item -ItemType Directory -Force -Path "etf_data" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host "初始化完成"

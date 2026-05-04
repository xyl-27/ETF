# 设置Windows定时任务 - 每日测评
# 管理员权限运行: powershell -ExecutionPolicy Bypass -File .\setup_daily_task.ps1
#
# 用法:
#   .\setup_daily_task.ps1                          # 默认: 每个交易日 18:00
#   .\setup_daily_task.ps1 -Time "17:30"            # 自定义时间
#   .\setup_daily_task.ps1 -TaskName "ETF_MorningEval" -Time "08:00"  # 自定义名称和时间
#   .\setup_daily_task.ps1 -Remove                  # 删除定时任务

param(
    [string]$Time = "18:00",
    [string]$TaskName = "ETF_DailyEval",
    [switch]$Remove,
    [switch]$ShowHistory
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 任务脚本路径
$SCRIPT_PATH = Join-Path $PSScriptRoot "daily_eval.ps1"
$LOG_PATH = Join-Path $PSScriptRoot "logs\daily_eval.log"

if ($ShowHistory) {
    $historyPath = Join-Path $PSScriptRoot "output\daily_eval_history.json"
    if (Test-Path $historyPath) {
        $history = Get-Content $historyPath -Raw | ConvertFrom-Json
        Write-Host "`n测评历史 (最近10次):" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        $history | Select-Object -Last 10 | ForEach-Object {
            $statusColor = if ($_.status -eq "success") { "Green" } else { "Red" }
            Write-Host "$($_.timestamp)  [$($_.status)]" -ForegroundColor $statusColor
            if ($_.prediction) {
                Write-Host "  Top-5: $($_.prediction.top_stocks -join ', ')" -ForegroundColor Yellow
            }
            if ($_.backtest) {
                Write-Host "  回测收益: $($_.backtest.strategy_return)% | 超额: $($_.backtest.excess_return)%" -ForegroundColor Yellow
            }
            Write-Host ""
        }
    } else {
        Write-Host "没有找到历史记录" -ForegroundColor Yellow
    }
    exit
}

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "定时任务 '$TaskName' 已删除" -ForegroundColor Yellow
    exit
}

# 确保日志目录存在
$logDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# 检查脚本是否存在
if (-not (Test-Path $SCRIPT_PATH)) {
    Write-Host "错误: 找不到脚本 $SCRIPT_PATH" -ForegroundColor Red
    exit 1
}

# 删除已存在的同名任务
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# 创建触发器 - 每天指定时间
$trigger = New-ScheduledTaskTrigger -Daily -At $Time

# 创建动作 - 运行PowerShell脚本并记录日志
$psExe = (Get-Command powershell).Source
$action = New-ScheduledTaskAction `
    -Execute $psExe `
    -Argument "-ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -File `"$SCRIPT_PATH`" >> `"$LOG_PATH`" 2>&1"

# 设置 - 如果错过运行则补跑, 允许唤醒计算机
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

# 注册任务
Register-ScheduledTask `
    -TaskName $TaskName `
    -Trigger $trigger `
    -Action $action `
    -Settings $settings `
    -Description "ETF每日测评: 运行预测和回测" `
    | Out-Null

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "定时任务创建成功!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "任务名称: $TaskName"
Write-Host "执行时间: 每天 $Time"
Write-Host "脚本路径: $SCRIPT_PATH"
Write-Host "日志路径: $LOG_PATH"
Write-Host ""
Write-Host "管理命令:" -ForegroundColor Cyan
Write-Host "  查看历史: .\setup_daily_task.ps1 -ShowHistory"
Write-Host "  删除任务: .\setup_daily_task.ps1 -Remove"
Write-Host "  手动运行: .\daily_eval.ps1"
Write-Host ""
Write-Host "注意: 如果需要修改时间, 先 -Remove 再重新创建" -ForegroundColor Yellow

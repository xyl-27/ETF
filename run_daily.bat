@echo off
cd /d C:\Users\xyl\Desktop\ETF

echo [%TIME%] ==================== ETF Daily Eval Start ====================

REM 检查 goldminer3.exe 是否已运行
tasklist /FI "IMAGENAME eq goldminer3.exe" 2>NUL | find /I /N "goldminer3.exe" >NUL
if "%ERRORLEVEL%" NEQ "0" (
    echo [%TIME%] goldminer3.exe not found, starting ...
    start "" "D:\opt\Goldminer\Hongshu Goldminer3\goldminer3.exe"
    echo [%TIME%] Waiting 30 seconds for initialization ...
    ping 127.0.0.1 -n 31 > NUL
) else (
    echo [%TIME%] goldminer3.exe already running, skipping start.
)

REM 按日期轮转日志
set LOG_DIR=output\logs
if not exist %LOG_DIR% mkdir %LOG_DIR%
for /f "usebackq" %%i in (`powershell -Command "Get-Date -Format 'yyyy-MM-dd'"`) do set TODAY=%%i
set LOG_FILE=%LOG_DIR%\%TODAY%.log

echo [%TIME%] Starting daily_eval.py ... >> %LOG_FILE%
venv\Scripts\python.exe code\src\daily_eval.py >> %LOG_FILE% 2>&1
echo [%TIME%] daily_eval.py finished. >> %LOG_FILE%

echo [%TIME%] ==================== ETF Daily Eval End ====================

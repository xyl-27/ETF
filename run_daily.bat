@echo off
cd /d C:\Users\xyl\Desktop\ETF

start "" "D:\opt\Goldminer\Hongshu Goldminer3\goldminer3.exe"

timeout /t 5 /nobreak > nul

.venv\Scripts\python.exe code\src\daily_eval.py

pause
@echo off
cd /d C:\Users\xyl\Desktop\ETF

start explorer.exe "D:\opt\Goldminer\Hongshu Goldminer3\goldminer3.exe"

.venv\Scripts\python.exe code\src\daily_eval.py

pause
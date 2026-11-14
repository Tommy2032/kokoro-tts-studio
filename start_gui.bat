@echo off
chcp 65001 >nul
"C:\Users\MadKalmar\AppData\Local\Python\bin\python.exe" app.py
if %ERRORLEVEL% neq 0 (
    python app.py
)
pause

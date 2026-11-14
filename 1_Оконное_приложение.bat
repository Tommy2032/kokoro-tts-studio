@echo off
cd /d "%~dp0"
echo Starting Desktop GUI (PyQt6)...
"C:\Users\MadKalmar\AppData\Local\Python\bin\python.exe" app.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error running app.py
    pause
)

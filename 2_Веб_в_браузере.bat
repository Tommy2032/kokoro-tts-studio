@echo off
cd /d "%~dp0"
echo Starting Web Studio in your browser...
"C:\Users\MadKalmar\AppData\Local\Python\bin\python.exe" web_app.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error running web_app.py
    pause
)

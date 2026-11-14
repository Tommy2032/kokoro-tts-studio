@echo off
cd /d "%~dp0"
echo ========================================================
echo   Starting Kokoro TTS Web Studio (Browser Version)...
echo ========================================================
echo.

set "PY=C:\Users\MadKalmar\AppData\Local\Python\bin\python.exe"

if exist "%PY%" (
    echo [OK] Python: %PY%
    "%PY%" web_app.py
) else (
    echo [OK] Python: system default
    python web_app.py
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo ========================================================
    echo Error during startup. Exit code: %ERRORLEVEL%
    echo ========================================================
    pause
)

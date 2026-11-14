@echo off
chcp 65001 >nul
cd /d "%~dp0"
cls

echo ========================================================
echo        Kokoro TTS Studio - Launcher Menu
echo ========================================================
echo.
echo   [1] Windows Desktop GUI (PyQt6 - Rekomenduetsya)
echo   [2] Web Studio in Browser (http://127.0.0.1:7860)
echo   [3] Console Quick Test (input.txt to output.wav)
echo.
echo ========================================================
set "choice=1"
set /p choice="Enter choice [1, 2, 3] (default 1): "

set "PY=C:\Users\MadKalmar\AppData\Local\Python\bin\python.exe"

if "%choice%"=="2" goto run_web
if "%choice%"=="3" goto run_cli

:run_gui
echo.
echo [OK] Launching Desktop GUI...
if exist "%PY%" (
    "%PY%" app.py
) else (
    python app.py
)
if %ERRORLEVEL% neq 0 (
    echo.
    echo [Notice] Desktop GUI exited with code %ERRORLEVEL%.
    echo Falling back to Web Studio in browser...
    goto run_web
)
goto end

:run_web
echo.
echo [OK] Launching Web Studio in Browser...
if exist "%PY%" (
    "%PY%" web_app.py
) else (
    python web_app.py
)
goto end

:run_cli
echo.
echo [OK] Synthesizing input.txt...
if exist "%PY%" (
    "%PY%" tts_simple.py
) else (
    python tts_simple.py
)
echo.
pause
goto end

:end

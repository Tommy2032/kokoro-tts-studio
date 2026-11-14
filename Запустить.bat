@echo off
chcp 65001 >nul
cd /d "%~dp0"
cls

echo ========================================================
echo        Kokoro TTS Studio - Меню запуска
echo ========================================================
echo.
echo   [1] Оконная программа Windows (PyQt6 Desktop)
echo   [2] Веб-студия в браузере (Chrome / Edge / Яндекс)
echo   [3] Консольный тест (озвучить input.txt)
echo.
echo ========================================================
set "choice=1"
set /p choice="Выберите вариант [1, 2, 3] (по умолчанию 1): "

set "PY=C:\Users\MadKalmar\AppData\Local\Python\bin\python.exe"

if "%choice%"=="2" goto run_web
if "%choice%"=="3" goto run_cli

:run_gui
echo.
echo Запуск оконного приложения...
if exist "%PY%" (
    "%PY%" app.py
) else (
    python app.py
)
if %ERRORLEVEL% neq 0 (
    echo.
    echo Оконное приложение завершилось с ошибкой.
    echo Запуск веб-версии в браузере...
    goto run_web
)
goto end

:run_web
echo.
echo Запуск веб-студии в браузере (http://127.0.0.1:7860)...
if exist "%PY%" (
    "%PY%" web_app.py
) else (
    python web_app.py
)
goto end

:run_cli
echo.
echo Синтез текста из input.txt...
if exist "%PY%" (
    "%PY%" tts_simple.py
) else (
    python tts_simple.py
)
echo.
pause
goto end

:end

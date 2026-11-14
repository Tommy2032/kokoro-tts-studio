@echo off
cd /d "%~dp0"
echo Synthesizing input.txt to output.wav...
"C:\Users\MadKalmar\AppData\Local\Python\bin\python.exe" tts_simple.py
echo.
pause

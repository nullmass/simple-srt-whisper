@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1" cpu %*
exit /b %ERRORLEVEL%

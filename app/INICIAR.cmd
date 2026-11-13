@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\iniciar.ps1" %*
set RESULTADO=%errorlevel%
if not "%RESULTADO%"=="0" pause
exit /b %RESULTADO%

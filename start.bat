@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Bitte zuerst setup.bat erfolgreich ausfuehren.
  exit /b 1
)
".venv\Scripts\python.exe" -m stockfish_vision %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%

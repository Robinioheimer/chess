@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Bitte zuerst setup.bat ausfuehren.
  exit /b 1
)
".venv\Scripts\python.exe" -m stockfish_vision %*
endlocal

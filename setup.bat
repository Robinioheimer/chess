@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher "py" nicht gefunden. Python 3.12+ installieren.
  exit /b 1
)
py -3.12 -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -e ".[dev,packaging]"
python scripts\train_piece_model.py --samples 48
echo.
echo Setup fertig. Stockfish separat herunterladen und den Pfad in der GUI setzen.
echo Danach start.bat ausfuehren.
endlocal

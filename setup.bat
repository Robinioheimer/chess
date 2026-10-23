@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher "py" nicht gefunden. Python 3.11+ installieren und "Add to PATH" aktivieren.
  exit /b 1
)
py -3 -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11 oder neuer wird benoetigt.'"
if errorlevel 1 (
  echo Python 3.11 oder neuer wird benoetigt. Python 3.12 wird empfohlen.
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 goto :error
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -e ".[dev,packaging]"
if errorlevel 1 goto :error
if not exist "src\stockfish_vision\assets\models\piece_hog_svm.xml" (
  echo Modell fehlt und wird aus den mitgelieferten synthetischen Glyphen trainiert ...
  python scripts\train_piece_model.py --samples 48 --seed 7
  if errorlevel 1 goto :error
) else (
  echo Vorhandenes HOG-SVM-Modell wird verwendet und nicht ueberschrieben.
)
echo.
echo Setup fertig. Stockfish separat von https://stockfishchess.org/download/ installieren.
echo Pfad in der GUI unter "Engine ^& Overlay" setzen, danach start.bat ausfuehren.
endlocal
exit /b 0
:error
echo.
echo Setup FEHLGESCHLAGEN. Meldungen oben pruefen; es wurde kein erfolgreicher Start vorgetaeuscht.
endlocal
exit /b 1

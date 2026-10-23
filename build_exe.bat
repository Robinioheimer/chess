@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Bitte zuerst setup.bat erfolgreich ausfuehren.
  exit /b 1
)
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean ^
  --windowed ^
  --name StockfishVisionChessAI ^
  --paths src ^
  --add-data "src\stockfish_vision\assets;stockfish_vision\assets" ^
  --hidden-import PySide6.QtCore ^
  --hidden-import PySide6.QtGui ^
  --hidden-import PySide6.QtWidgets ^
  src\stockfish_vision\__main__.py
if errorlevel 1 goto :error
echo.
echo Fertig: dist\StockfishVisionChessAI\StockfishVisionChessAI.exe
echo Stockfish.exe wird NICHT mitgeliefert und muss separat konfiguriert werden.
endlocal
exit /b 0
:error
echo PyInstaller-Build FEHLGESCHLAGEN. Meldungen oben pruefen.
endlocal
exit /b 1

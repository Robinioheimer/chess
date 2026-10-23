# Stockfish Vision Chess AI

Python-Programm für Windows 10/11: erkennt ein Schachbrett auf dem Bildschirm (eingefrorener Snapshot), rekonstruiert die Stellung, analysiert sie mit einer **lokalen Stockfish-UCI-Engine** und zeigt Varianten in einem transparenten Overlay.

**Live-Engine-Hilfe in laufenden Online-Partien ist deaktiviert.** Analyse gibt es nur für:

- eigene Trainingsstellungen (internes Brett / FEN)
- abgeschlossene Partien (PGN-Import)
- bewusst eingefrorene Snapshots nach Bestätigung, dass es *keine* laufende Online-Partie ist

Automatisches Nachziehen der Engine, sobald sich ein Live-Brett ändert, ist nicht implementiert. Schnelle Stellungswechsel werden als laufende Partie gewertet und sperren die Snapshot-Analyse.

## Voraussetzungen

- Windows 10 oder 11
- Python 3.12 oder neuer (`py -3.12`)
- Stockfish-Binary (**nicht** Bestandteil dieses Repos)

### Stockfish – Lizenz und Bezugsquelle

Stockfish ist eine eigenständige Schachengine unter der **GNU General Public License v3.0**.

- Download (offiziell): https://stockfishchess.org/download/
- Quellcode: https://github.com/official-stockfish/Stockfish
- Lizenztext: https://github.com/official-stockfish/Stockfish/blob/master/Copying.txt

Dieses Projekt enthält **keine** Stockfish-Binary und verkauft Stockfish nicht. Sie laden Stockfish selbst herunter und tragen den Pfad zur `stockfish.exe` in den Engine-Einstellungen ein. Alternativ Umgebungsvariable `STOCKFISH_PATH` setzen.

Der Anwendungscode steht unter der MIT-Lizenz (`LICENSE`). Die GPL von Stockfish gilt für die Engine, nicht für diesen Wrapper.

## Installation

Im Projektordner:

```bat
setup.bat
```

Das erzeugt `.venv`, installiert Abhängigkeiten und trainiert das lokale HOG-SVM-Figurenmodell.

Manuell:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,packaging]"
python scripts\train_piece_model.py --samples 48
```

Abhängigkeiten stehen in `requirements.txt` bzw. `pyproject.toml`: OpenCV, python-chess, NumPy, PySide6, mss, Pillow.

## Start

```bat
start.bat
```

oder:

```bat
.venv\Scripts\python.exe -m stockfish_vision
```

1. Unter **Engine** den Pfad zu `stockfish.exe` setzen und **Engine starten**.
2. **Training**: FEN einfügen oder Figuren setzen, dann analysieren.
3. **PGN**: abgeschlossene Partie laden, Zug auswählen, analysieren.
4. **Aufnahme** (nur Trainingsdiagramme / stehende Stellungen): Monitor oder Bereich wählen, Snapshot, vier Brettecken klicken, Orientierung setzen, Bestätigung aktivieren, erkennen, bei Bedarf Felder korrigieren, dann Snapshot analysieren.

Hotkey Overlay ein/aus (Standard): `Ctrl+Shift+F8`.

Konfiguration und Logs: `%APPDATA%\StockfishVisionChessAI\`

## Tests

```bat
.venv\Scripts\python.exe -m pytest
```

Abgedeckt: FEN, UCI (Fake-Engine, nicht Stockfish), Koordinaten, Brettorientierung, PGN, Erkennung, Policy gegen Live-Hilfe.

## EXE mit PyInstaller (optional)

```bat
build_exe.bat
```

Ergebnis: `dist\StockfishVisionChessAI\StockfishVisionChessAI.exe`. Stockfish muss weiterhin separat liegen.

## Architektur

- `capture` – Monitor/Bereich via `mss`
- `board` – Perspektivtransform, 64 Felder, Orientierung
- `recognition` – Belegung, HOG-SVM, Session-Templates, manuelle Korrektur
- `chesslogic` – FEN, Validierung mit python-chess, PGN, taktische Hinweise
- `engine` – UCI über `chess.engine.SimpleEngine` im Worker-Thread
- `overlay` – transparentes, klickdurchlässiges Fenster
- `gui` – PySide6
- `policy` – Sperre für Live-Online-Assistenz

## Figuren-Erkennung

Mitgeliefert wird ein trainierbares OpenCV-SVM auf HOG-Merkmalen synthetischer Figuren. Unterschiedliche Sets, Farben und Skalierungen werden über die Vier-Punkte-Kalibrierung, Session-Templates nach manueller Korrektur und die Orientierungseinstellung abgedeckt. Unsichere Felder müssen Sie prüfen – das Programm erfindet keine Stellungen.

## Hinweis zur Fairness

Nicht in laufenden Online-Partien als Zugvorschlag nutzen. Viele Plattformen verbieten Engine-Unterstützung. Dieses Programm ist für Training und Nachbereitung gedacht und setzt das technisch um.

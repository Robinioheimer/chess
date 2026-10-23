# Stockfish Vision Chess AI

Windows-10/11-Anwendung für **eigene Schachstellungen, Training und Nachbereitung abgeschlossener Partien**. Sie erfasst einen manuell ausgewählten Bildschirmbereich, kalibriert ein Schachbrett über vier Punkte, erkennt Figuren mit OpenCV/HOG-SVM, rekonstruiert eine FEN und analysiert gültige Stellungen über eine separat installierte Stockfish-UCI-Engine.

Die PySide6-Oberfläche zeigt Erkennungs-Scores, Stellungsfehler und mehrere Engine-Varianten. Ein transparentes Overlay stellt bei kalibrierten Snapshots farbige Pfeile und Feldmarkierungen dar. Das unabhängige Analyse-Panel ist verschieb- und skalierbar.

> **Fairness / Nutzungsrahmen:** Es gibt keine automatische Live-Engine-Analyse und keine Anti-Cheat-Umgehung. Engine-Ausgaben werden nur manuell für Trainingsstellungen, ausdrücklich bestätigte eingefrorene Snapshots oder abgeschlossene PGN-Partien angefordert. Nutzen Sie das Programm nicht während laufender gewerteter Online-Partien.

## Funktionsumfang

- Monitor oder frei definierter Aufnahmebereich, auch mit negativen Koordinaten bei Multi-Monitor-Setups
- eingefrorener Snapshot; die Vorschau überschreibt ihn nicht
- Vier-Punkte-Perspektivkalibrierung mit Prüfung auf ungültige Vierecke
- vier Brettorientierungen und konfigurierbare Warp-Größe
- Belegungserkennung, HOG-SVM und Session-Templates aus manuellen Korrekturen
- Konfidenzscore und Quelle für alle 64 Felder, Alternativklassen und sichtbare Warnungen
- manuelle Korrektur jedes Feldes; ungelöste belegte Felder werden nicht erfunden
- FEN-Rekonstruktion und Validierung mit `python-chess`
- separate Eingabe von Zugrecht, Rochaderechten und En-passant-Feld
- Stockfish über UCI in einem Hintergrund-Thread; Threads, Hash, Tiefe, Zeit und bis zu fünf MultiPV-Linien
- Capture und 64-Feld-Erkennung in weiteren Hintergrund-Threads
- FEN-/PGN-Import und -Export sowie Textexport eines Analyseergebnisses
- bis zu 50 PGN-Partien pro Datei, Zugnavigation und Kommentare der Hauptvariante
- transparente, klickdurchlässige Pfeilebene und interaktives Overlay-Panel
- atomare JSON-Konfiguration, Sicherung defekter Konfigurationen und Protokolldateien

## Voraussetzungen

- Windows 10 oder 11, 64 Bit empfohlen
- Python **3.11 oder neuer**; Python 3.12 wird empfohlen
- eine separate Stockfish-Binärdatei für die eigene Windows-Architektur
- ausreichend Rechte für Bildschirmaufnahme und den globalen Hotkey

Die Python-Pakete stehen in `requirements.txt` und `pyproject.toml`. OpenCV wird als **headless**-Variante installiert, weil alle Fenster durch Qt/PySide6 und nicht durch `cv2.imshow` erzeugt werden.

## Stockfish legal beziehen und einrichten

Stockfish ist **nicht** Bestandteil dieses Repositories oder des EXE-Builds.

- offizieller Download: <https://stockfishchess.org/download/>
- offizieller Quellcode: <https://github.com/official-stockfish/Stockfish>
- GPLv3-Lizenz: <https://github.com/official-stockfish/Stockfish/blob/master/Copying.txt>

Laden Sie Stockfish nur aus einer vertrauenswürdigen offiziellen Quelle herunter und entpacken Sie es beispielsweise nach `C:\Tools\Stockfish\stockfish.exe`. Prüfen Sie bei Downloads die vom Anbieter angegebenen Signaturen bzw. Hashwerte, sofern vorhanden. Tragen Sie den Pfad anschließend unter **Engine & Overlay** ein.

Alternativen:

- Umgebungsvariable `STOCKFISH_PATH` auf die ausführbare Datei setzen;
- `stockfish` bzw. `stockfish.exe` über `PATH` auffindbar machen;
- eine Datei `engines\stockfish.exe` im aktuellen Projektverzeichnis verwenden.

Stockfish steht unter GPLv3; der Wrapper-Code dieses Projekts steht unter der MIT-Lizenz in `LICENSE`. Es wird keine Engine-Binärdatei ungeprüft in Git aufgenommen.

## Installation und Start unter Windows

Im Projektordner:

```bat
setup.bat
start.bat
```

`setup.bat` erzeugt `.venv`, installiert Laufzeit-, Test- und Build-Abhängigkeiten und verwendet das mitgelieferte Modell. Nur falls die Modelldatei fehlt, wird sie aus den im Projekt enthaltenen synthetischen Glyphen neu trainiert. Fehler brechen das Skript mit einem Exit-Code ungleich null ab.

Manuelle Installation:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,packaging]"
python -m stockfish_vision
```

Alternativ nach `start.bat`:

```bat
.venv\Scripts\python.exe -m stockfish_vision
```

Beim ersten Start:

1. **Engine & Overlay** öffnen, `stockfish.exe` auswählen und **Engine starten** wählen.
2. Für eine sofortige Funktionsprüfung unter **Training & Korrektur** die Grundstellung verwenden und **Trainingsstellung analysieren** wählen.
3. Für Bilderkennung den folgenden Kalibrierungsworkflow ausführen.

## Bildschirmaufnahme und Kalibrierung

1. Öffnen Sie eine **ruhende eigene Trainingsstellung** oder ein Diagramm einer abgeschlossenen Partie.
2. Wählen Sie unter **Aufnahme** einen Monitor oder aktivieren Sie den manuellen Bereich. `L/T` sind die globale linke obere Ecke, `B/H` Breite und Höhe in Bildschirmpixeln.
3. Starten Sie optional die Vorschau. Sie dient nur zur Kalibrierung und startet weder Erkennung noch Engine automatisch.
4. Wählen Sie **Snapshot aufnehmen & Vorschau einfrieren**.
5. Klicken Sie die äußeren Brettecken in dieser Reihenfolge:
   - oben links,
   - oben rechts,
   - unten rechts,
   - unten links.

   Ein fünfter Klick beginnt die Auswahl neu; **Ecken zurücksetzen** löscht sie vollständig. Das Viereck muss konvex sein und darf nicht überkreuzt sein.
6. Wählen Sie die Orientierung aus Sicht des Bildes:
   - `white_bottom`: Weiß unten, normale Ansicht;
   - `white_top`: Weiß oben / Schwarzperspektive;
   - `white_left` / `white_right`: Weiß an der entsprechenden Seite.
7. Bestätigen Sie den Trainings-/Nachbereitungsrahmen und wählen Sie **Snapshot im Hintergrund erkennen**.
8. Prüfen Sie unter **Training & Korrektur** Stellung und Scores. Erst danach **Geprüften Snapshot analysieren** wählen.

Die Kalibrierung bezieht sich auf den ausgewählten Snapshot. Bei geändertem Ausschnitt, verschobenem Brett oder anderer Skalierung muss sie erneut geprüft werden.

## Erkennung, Scores und manuelle Korrektur

Das mitgelieferte Modell verwendet:

- 32×32-Pixel-Felder;
- OpenCV-HOG mit 8×8-Zellen, 9 Bins und 324 Merkmalen;
- eine C-SVC/RBF-SVM mit zwölf Klassen (`P N B R Q K p n b r q k`);
- ausschließlich synthetische, im Repository erzeugte Figuren-Glyphen.

OpenCV liefert für diese Mehrklassen-SVM keine kalibrierten Wahrscheinlichkeiten. Der angezeigte Score ist deshalb eine **heuristische Bewertung**: Die 66 paarweisen SVM-Entscheidungsmargen werden sigmoid-transformiert und als mittlerer paarweiser Gewinn-Score aggregiert. Die Klassenauswahl folgt dem nativen OpenCV-Voting. `0,80` bedeutet daher nicht „80 % gesicherte Wahrscheinlichkeit“.

- **Rot / unter Warnschwelle:** Feld unbedingt prüfen.
- **`?` / `unresolved`:** belegt, aber keine Figur zuverlässig bestimmt; es wird kein Bauer erfunden.
- **`svm`:** Ergebnis aus HOG-SVM und Farbheuristik.
- **`template`:** Treffer aus einer manuellen Korrektur dieser Sitzung.
- **`manual`:** manuell gesetzt.
- **`empty`:** als leer eingestuft.

Zum Korrigieren ein Feld auf dem Brett oder in der Tabelle auswählen, die Figur rechts wählen und **Auswahl auf Feld anwenden** drücken. Korrekturen erzeugen Session-Templates für ähnliche Felder. Diese Templates sind bewusst nur sitzungslokal und werden nicht als angeblich trainiertes Modell gespeichert.

**Wichtige Grenze:** Ein synthetisches Modell kann nicht jedes Figurenset, jede Website, 3D-Ansicht oder beliebige perspektivische Verzerrung zuverlässig erkennen. Für unbekannte Sets sind manuelle Korrekturen erforderlich; bei systematisch falschen Ergebnissen ist die Stellung per FEN/Bretteditor einzugeben. Es wird keine universelle Genauigkeit behauptet.

## FEN und nicht sichtbare Stellungsdaten

Aus Pixeln lassen sich Zugrecht, Rochaderechte, En-passant und Zugzähler nicht zuverlässig ableiten. Die Bilderkennung setzt deshalb standardmäßig:

- das unter **Zugrecht** gewählte Zugrecht;
- keine Rochaderechte (`-`);
- kein En-passant-Feld (`-`);
- Halbzugzähler `0`, Zugnummer `1`.

Korrigieren Sie diese Felder unter **Training & Korrektur**, bevor Sie analysieren. **FEN übernehmen** erlaubt auch das Bearbeiten einer unvollständigen Stellung; die Engine-Analyse wird jedoch bei fehlenden Königen, ungültigen Rochaderechten, illegalen Bauernstellungen oder anderen Validierungsfehlern blockiert.

- **FEN importieren:** genau eine Stellung aus `.fen`/`.txt`.
- **FEN exportieren:** aktuelle vollständige FEN.
- Änderungen an einer importierten FEN werden als eigene Trainingsstellung behandelt.

## Stockfish-Analyse

Unter **Engine & Overlay**:

- **Suchtiefe `0`:** keine Tiefenbegrenzung, nur Zeit verwenden;
- **Zeit `0 ms`:** keine Zeitbegrenzung, nur Tiefe verwenden;
- sind beide Werte aktiv, gilt die zuerst erreichte Engine-Grenze;
- sind beide `0`, verwendet der Client eine sichere Standardtiefe von 12;
- **Threads / Hash:** werden an die von der Engine gemeldeten Optionsgrenzen angepasst;
- **MultiPV 1–5:** mehrere beste Zugfolgen, ebenfalls auf das Engine-Maximum begrenzt.

Bewertungen erscheinen **aus Weiß-Sicht**: positive Bauernwerte favorisieren Weiß, negative Schwarz; `M+3` bedeutet Matt für Weiß in drei Zügen, entsprechend negativ für Schwarz. Die Analyseansicht zeigt Tiefe, besten Zug und SAN-Variante. Die Historie enthält zusätzlich FEN und UCI-Kontext.

Wenn sich die Stellung während einer Suche ändert, bleibt das Ergebnis in der Historie sichtbar, wird aber nicht irreführend auf das neue Brett oder Overlay projiziert.

## Overlay

- Standard-Hotkey unter Windows: **Strg+Umschalt+F8**.
- Hotkey im Einstellungsfeld ändern und ausdrücklich **Hotkey anwenden** wählen.
- Deckkraft und Basisskalierung sind einstellbar.
- Panel am Titelbereich ziehen; an der unteren rechten Ecke **`◢`** skalieren.
- Das Panel ist interaktiv. Die separate Pfeilebene bleibt klickdurchlässig.
- PV1–PV5 verwenden unterschiedliche Farben; Start- und Zielfelder werden als perspektivisch transformierte Polygone markiert.

Bildschirmpfeile erscheinen nur für einen analysierten, kalibrierten Snapshot. Bei internen FEN-/PGN-Stellungen zeigt das Overlay nur das Analyse-Panel, damit keine Pfeile auf ein nicht zugehöriges Bildschirmbrett projiziert werden.

## PGN-Nachbereitung und Export

**PGN laden** liest bis zu 50 Partien. Die Partie kann oben gewählt werden; die Liste zeigt die Hauptvariante einschließlich Ausgangsstellung. Kommentare des gewählten Knotens erscheinen in der Statuszeile.

Nur Partien mit Ergebnis `1-0`, `0-1` oder `1/2-1/2` sind für den PGN-Analysemodus freigegeben. Ergebnis `*` wird als nicht abgeschlossen behandelt und im PGN-Modus gesperrt.

**PGN exportieren** schreibt:

- die aktuell ausgewählte geladene Partie einschließlich ihrer PGN-Struktur; oder
- falls keine Partie geladen ist, die aktuelle Trainingsstellung mit `SetUp`- und `FEN`-Headern.

Die Analyse kann zusätzlich als UTF-8-Textdatei exportiert werden.

## Konfiguration, Modelle und Logs

Benutzerverzeichnis:

```text
%APPDATA%\StockfishVisionChessAI\
```

- `config.json`: Einstellungen und letzte Stellung
- `logs\vision-YYYYMMDD.log`: Protokolle

`config.example.json` zeigt alle Felder mit nachvollziehbaren Standardwerten. Es kann als Referenz kopiert und angepasst werden; Pfade müssen zur eigenen Installation passen. Schreiben erfolgt atomar. Eine unlesbare JSON-Datei wird mit Zeitstempel als `config.corrupt-…json` gesichert und durch Standardwerte ersetzt; Details stehen im Log.

Das Standardmodell liegt unter:

```text
src\stockfish_vision\assets\models\piece_hog_svm.xml
```

Ein kompatibles alternatives OpenCV-XML kann unter **Engine & Overlay → Figurenmodell** gewählt werden. Es muss dieselben zwölf Klassen und 324 HOG-Merkmale verwenden. Nach einem Modellwechsel wird der Klassifikator neu aufgebaut.

Modell reproduzierbar neu erzeugen:

```bat
.venv\Scripts\python.exe scripts\train_piece_model.py --samples 64 --seed 7 --out %APPDATA%\StockfishVisionChessAI\piece_hog_svm.xml
```

Dabei entsteht neben der XML eine JSON-Metadatendatei mit Klassen, HOG-Parametern, Seed und Trainingsdatenbeschreibung. Ohne `--out` wird im Source-Checkout das gebündelte Modell ersetzt; diese Änderung vor dem Commit prüfen. Es werden keine externen Bilddaten oder nicht dokumentierten Modelle heruntergeladen.

## Tests und statische Prüfung

```bat
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check src tests scripts
```

Die Tests decken unter anderem Koordinaten, Perspektivkalibrierung, FEN, Dateiimport/-export, PGN, Konfiguration, Hotkey-Parsing, Fairness-Sperren, SVM-Voting und variable Konfidenzscores, Belegung, fehlende Modelle, Templates und UCI mit einer echten lokalen Fake-UCI-Engine ab. Die Fake-Engine ist ausdrücklich **kein Stockfish** und dient nur der Protokollprüfung.

`tests/test_gui_smoke.py` erzeugt auf einem System mit vollständigen nativen Qt-Bibliotheken ein Offscreen-Fenster und prüft Brettkoordinaten sowie den Hauptfenster-Lebenszyklus. Auf minimalen Headless-Systemen ohne `libGL` wird dieses Modul übersprungen; das ist kein bestandener GUI-Test.

Für den optionalen echten Stockfish-Integrationstest setzen Sie `STOCKFISH_PATH` auf die offizielle Binary und führen denselben Testbefehl aus. Dann wird zusätzlich `tests/test_stockfish_integration.py` ausgeführt; ohne diese Umgebungsvariable wird nur dieser optionale Test übersprungen. Der automatische Standardtest benötigt keinen Stockfish-Download.

## EXE mit PyInstaller

Der Build muss unter Windows mit derselben Python-Architektur erfolgen, die auch die Zielsysteme verwenden:

```bat
build_exe.bat
```

Ergebnis:

```text
dist\StockfishVisionChessAI\StockfishVisionChessAI.exe
```

Den **gesamten Ordner** `dist\StockfishVisionChessAI\` weitergeben, nicht nur die EXE. Modelldaten werden gebündelt; Stockfish wird aus lizenz- und wartungsgründen nicht gebündelt und muss auf dem Zielsystem separat ausgewählt werden. Konfiguration und Logs liegen weiterhin unter `%APPDATA%`.

PyInstaller ist kein Cross-Compiler: Ein Windows-Build kann nicht unter Linux erzeugt und gleichzeitig als getesteter Windows-Build ausgegeben werden. SmartScreen-/Signaturhinweise hängen von der eigenen Zertifikats- und Verteilungskonfiguration ab; das Projekt enthält kein Code-Signing-Zertifikat.

## Fehlersuche

- **Engine nicht gefunden:** Pfad und Architektur prüfen; Stockfish nicht nur entpacken, sondern die tatsächliche `stockfish.exe` wählen.
- **Engine startet nicht:** Log öffnen, Pfad nicht auf ein ZIP/Verzeichnis setzen, Zeit/Tiefe zunächst auf 800 ms / 18 lassen.
- **Viele `unresolved`-Felder:** Brettecken und Warp-Größe prüfen, höher aufgelösten Bereich verwenden, Felder korrigieren; nicht einfach eine ungültige Stellung analysieren.
- **Ungültige FEN:** Könige, Zugrecht, Rochaderechte, En-passant und Bauern auf Grundreihen prüfen.
- **Hotkey belegt:** andere Kombination wählen und erneut anwenden; die GUI-Schaltfläche funktioniert unabhängig davon.
- **Kein Bildschirmzugriff:** Monitorindex/Bereich prüfen und sicherstellen, dass keine Restriktion des Systems die Aufnahme verhindert.
- **Overlay falsch platziert:** nach Verschieben des Bildschirmbretts Snapshot und Kalibrierung erneuern. Multi-Monitor-/DPI-Einstellungen nach einer Änderung ebenfalls neu prüfen.

## Projektstruktur

```text
src/stockfish_vision/
  capture/       Monitor-/Bereichsaufnahme via mss
  board/         Perspektive, 64 Felder, Orientierung
  recognition/   Belegung, HOG-SVM, Scores, Session-Templates
  chesslogic/    FEN, PGN, Validierung, taktische Hinweise
  engine/        UCI-Client und Ergebnisdaten
  gui/           PySide6-Fenster und Hintergrund-Worker
  overlay/       Pfeilebene, interaktives Panel, Windows-Hotkey
  assets/models/ mitgeliefertes OpenCV-Modell
  config.py      validierte JSON-Einstellungen
  policy.py      Trainings-/Snapshot-/PGN-Freigaben
scripts/         reproduzierbares Modelltraining
tests/           Kern-, Protokoll- und GUI-Smoke-Tests
```

## Lizenz und Fairness

Anwendungscode: MIT, siehe `LICENSE`. Stockfish: separat und GPLv3. Die Anwendung ist für Training, eigene Stellungen und abgeschlossene Partien bestimmt. Die technische Trennung von Vorschau und manueller Analyse ist bewusst erhalten geblieben.

# Stockfish Vision Chess AI – **v8**

Windows-10/11-Anwendung für **eigene Schachstellungen, Training und Nachbereitung abgeschlossener Partien**. Sie erfasst einen manuell ausgewählten Bildschirmbereich, kalibriert ein Schachbrett über vier Punkte, erkennt Figuren mit OpenCV/HOG-SVM, rekonstruiert eine FEN und analysiert gültige Stellungen über eine separat installierte Stockfish-UCI-Engine.

Neu in **v8**: **Eigener Figurensatz** – statt nur mit synthetischen Glyphen zu arbeiten, lernt das Programm die tatsächlich sichtbaren Figuren einmalig aus der **Grundstellung** an (Start-Tab oder *4 · Einstellungen → Eigener Figurensatz*) und speichert sie als Vorlagen. Gemessen am Standard-Figurensatz (cburnett) steigt die Trefferquote damit von **17 % auf über 90 %**. Außerdem sitzt der **rote Rahmen im Overlay** jetzt auch bei gemischter Windows-Skalierung und mehreren Monitoren korrekt: Die Umrechnung nutzt die physischen Monitorrechtecke statt der falschen Annahme „logischer Ursprung × DPI-Faktor“.

Neu in **v7**: **Live-Tracking** – das Programm verfolgt den eingerahmten Brettbereich selbstständig: es nimmt in einem einstellbaren Intervall ein Bild auf, erkennt die Stellung neu, sobald sich sichtbar etwas geändert hat, und zieht Brett, Overlay und Historie mit. Die Engine bleibt dabei aus; analysiert wird weiterhin nur ein bewusst eingefrorener Snapshot. Außerdem eine **Plausibilitätskorrektur**: unmögliche Erkennungen wie zwei Könige einer Farbe, neun Bauern oder ein Bauer auf der Grundreihe werden automatisch auf die zweitbeste Klasse abgebildet – die Meldung „zu viele Könige“ blockiert die Analyse nicht mehr, und jede Korrektur wird im Prüf-Tab angezeigt.

Neu in **v6**: Die Oberfläche ist jetzt **hell** – weißer Hintergrund, schwarzer Text, graue Akzente (Grün nur noch als Bedeutung „erledigt“, Rot/Orange nur für Warnungen). Die Ausschnitt-Auswahl legt für **jeden Monitor eine eigene Ebene** über den kompletten Bildschirm, sodass wirklich der ganze Desktop bedeckt ist. Die Zustimmung zum Trainingsmodus wird **automatisch** erteilt und nicht mehr per Checkbox abgefragt. Details unter [Versionen](#versionen-v1-v2-v4-v5-v6-v7-v8-…).

Neu in **v5**: Die Ausschnitt-Auswahl deckt jetzt immer den **kompletten Desktop** ab (auch mehrere Monitore, negative Koordinaten und hohe Windows-Skalierung) – vorher konnte das Auswahlfenster kleiner als der Bildschirm sein. Das Helfer-Popup ist breiter, lässt sich **am rechten Rand ziehen** und merkt sich seine Breite.

Neu in **v4**: Die Einrichtung läuft über ein eigenes **Windows-Popup** statt über ein Fenster in der App. Der Bildschirmausschnitt wird mit der **Maus** aufgezogen, das Helfer-Fenster führt durch die drei Schritte und bleibt zusammen mit dem Overlay dauerhaft im Vordergrund – die trainierte Seite im Browser wird nicht mehr verdrängt. (v3 wurde übersprungen; v2 brachte Start-Tab, Größen-/Schriftsteuerung, automatischen roten Rahmen.)

Die PySide6-Oberfläche zeigt Erkennungs-Scores, Stellungsfehler und mehrere Engine-Varianten. Ein transparentes Overlay stellt bei kalibrierten Snapshots farbige Pfeile und Feldmarkierungen dar. Das unabhängige Analyse-Panel ist verschieb- und skalierbar.

> **Fairness / Nutzungsrahmen:** Es gibt keine automatische Live-Engine-Analyse und keine Anti-Cheat-Umgehung. Engine-Ausgaben werden nur manuell für Trainingsstellungen, ausdrücklich bestätigte eingefrorene Snapshots oder abgeschlossene PGN-Partien angefordert. Das Live-Tracking beobachtet zwar das Brett, ruft die Engine aber nie selbst auf – für eine Bewertung muss der Snapshot bewusst eingefroren und analysiert werden. Nutzen Sie das Programm nicht während laufender gewerteter Online-Partien.

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
- transparenter roter Brettrahmen, automatisch erkannt oder manuell gesetzt
- klickdurchlässige Pfeilebene, interaktives Overlay-Panel, DPI-Korrektur für Windows-Skalierung
- einstellbare Fenstergröße (Voreinstellungen, freie Pixelwerte, maximierter Start) und Schriftgröße
- geführter Start-Tab mit Statusübersicht und Overlay-Test
- **Einrichtungs-Popup** (immer im Vordergrund) mit drei Schritten: Ausschnitt, Snapshot + Erkennen, Analysieren
- **Live-Tracking**: erkennt Stellungsänderungen im Brettrahmen automatisch neu (Intervall einstellbar), Engine bleibt aus
- **Plausibilitätskorrektur** der Erkennung: höchstens ein König je Seite, höchstens acht Bauern, maximal 16 Figuren je Seite, keine Bauern auf der Grundreihe
- **Eigener Figurensatz**: einmaliges Anlernen der echten Figuren aus der Grundstellung, danach vorlagenbasierte Erkennung
- **Bildschirmausschnitt per Maus** aufziehen: Abdunkelung, Auswahlrahmen mit Größenanzeige, `Esc` bricht ab
- Vordergrund-Wächter, der Popup und Overlay regelmäßig wieder nach vorn holt
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

Laden Sie Stockfish nur aus einer vertrauenswürdigen offiziellen Quelle herunter und entpacken Sie es beispielsweise nach `C:\Tools\Stockfish\stockfish.exe`. Prüfen Sie bei Downloads die vom Anbieter angegebenen Signaturen bzw. Hashwerte, sofern vorhanden. Tragen Sie den Pfad anschließend unter **4 · Einstellungen** ein.

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

1. **4 · Einstellungen** öffnen, `stockfish.exe` auswählen und **Engine starten** wählen.
2. Für eine sofortige Funktionsprüfung unter **2 · Prüfen** die Grundstellung verwenden und **Trainingsstellung analysieren** wählen.
3. Für Bilderkennung den folgenden Kalibrierungsworkflow ausführen.

## Einrichtung über das Popup (v4)

Beim Start erscheint oben rechts ein kleines Fenster **Brett einrichten** – unabhängig vom Hauptfenster und immer im Vordergrund. Es führt durch drei Schritte:

1. **Ausschnitt wählen** – „Ausschnitt mit der Maus wählen“ drückt den **gesamten Desktop** ab (alle Monitore, mit 2 px Rand, damit auch der äußerste Bildschirmrand erreichbar bleibt). Rahmen mit gedrückter linker Maustaste um das Spielfeld ziehen (Größe wird live angezeigt), `Esc` oder rechte Maustaste bricht ab. Danach wird der Bereich gespeichert, ein Snapshot aufgenommen, das Brett automatisch rot eingerahmt und die Figuren erkannt.
2. **Snapshot und erkennen** – „Snapshot + erkennen“ wiederholt die Aufnahme jederzeit; der Status nennt geprüfte und unsichere Felder.
3. **Analysieren** – startet bei Bedarf zuerst die Engine und analysiert die geprüfte Stellung; Ergebnis erscheint im Panel und als Pfeile im Overlay.

Bedienung des Popups:

- Am Fenster ziehen verschiebt es; am rechten Rand ziehen ändert die Breite (340–1400 px). Alternativ *4 · Einstellungen → Popup-Breite*.
- Position, Breite und Sichtbarkeit werden gespeichert.
- **✕** oder **Ausblenden** blendet es aus; zurück über *Start → Popup einblenden*, `Strg+Umschalt+H` oder *Ansicht → Einrichtungs-Popup anzeigen*.
- **Einstellungen** öffnet das Hauptfenster im Einstellungs-Tab, **Overlay testen** zeigt Rahmen und Demo-Pfeil ohne Engine.
- Das Hauptfenster dient weiter zum Korrigieren von Feldern, für FEN/PGN und alle Detail-Einstellungen; es muss nicht im Vordergrund stehen.

Damit die trainierte Seite (z. B. Google im Browser) sichtbar bleibt, bringen Popup und Overlay sich alle zwei Sekunden selbst wieder nach vorn (`WS_EX_TOPMOST`, ohne Fokus zu stehlen). Falls eine Anwendung im echten Vollbild (`F11`) arbeitet, hilft der Fenstermodus.

## Bildschirmaufnahme und Kalibrierung

1. Öffnen Sie eine **ruhende eigene Trainingsstellung** oder ein Diagramm einer abgeschlossenen Partie.
2. Wählen Sie unter **1 · Aufnahme** einen Monitor oder aktivieren Sie den manuellen Bereich. `L/T` sind die globale linke obere Ecke, `B/H` Breite und Höhe in Bildschirmpixeln.
3. Starten Sie optional die Vorschau. Sie dient nur zur Kalibrierung und startet weder Erkennung noch Engine automatisch.
4. Wählen Sie **Snapshot aufnehmen & Vorschau einfrieren**.
5. Der Rahmen entsteht auf drei Wegen:
   - **automatisch:** Ist *Beim Snapshot automatisch rot einrahmen* aktiv, sucht das Programm das Brett direkt nach dem Snapshot und rahmt es rot ein. **Brett jetzt automatisch einrahmen** wiederholt die Suche.
   - **vier Klicks:** oben links, oben rechts, unten rechts, unten links. Ein fünfter Klick beginnt die Auswahl neu.
   - **Rechteck aufziehen:** mit gedrückter linker Maustaste ein Rechteck um das Brett ziehen.

   **Rahmen / Ecken zurücksetzen** löscht die Auswahl vollständig. Das Viereck muss konvex sein und darf nicht überkreuzt sein.
   Sitzt der automatische Rahmen nicht exakt, Ecken einzeln nachklicken – der rote Rahmen bleibt auch ohne Engine-Analyse sichtbar.
6. Wählen Sie die Orientierung aus Sicht des Bildes:
   - `white_bottom`: Weiß unten, normale Ansicht;
   - `white_top`: Weiß oben / Schwarzperspektive;
   - `white_left` / `white_right`: Weiß an der entsprechenden Seite.
7. Der Trainingsrahmen ist automatisch bestätigt (siehe Hinweis *Trainingsmodus aktiv*); wählen Sie **Snapshot im Hintergrund erkennen**.
   Alternativ **Live-Tracking starten** (im Popup oder unter *1 · Aufnahme*): Das Programm erkennt die Stellung dann selbstständig neu, sobald sich das Brett sichtbar ändert, und schreibt jeden Schritt in die Historie. Für eine Engine-Bewertung den Snapshot einfrieren – Tracking stoppt dabei automatisch.
8. Prüfen Sie unter **2 · Prüfen** Stellung und Scores. Erst danach **Geprüften Snapshot analysieren** wählen.

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

Korrigieren Sie diese Felder unter **2 · Prüfen**, bevor Sie analysieren. **FEN übernehmen** erlaubt auch das Bearbeiten einer unvollständigen Stellung; die Engine-Analyse wird jedoch bei fehlenden Königen, ungültigen Rochaderechten, illegalen Bauernstellungen oder anderen Validierungsfehlern blockiert.

- **FEN importieren:** genau eine Stellung aus `.fen`/`.txt`.
- **FEN exportieren:** aktuelle vollständige FEN.
- Änderungen an einer importierten FEN werden als eigene Trainingsstellung behandelt.

## Stockfish-Analyse

Unter **4 · Einstellungen**:

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

### Roter Brettrahmen

- Der Rahmen wird aus den vier kalibrierten Ecken perspektivisch gezeichnet und ist auch ohne Engine-Ergebnis sichtbar.
- **Rahmenbreite** und **Rahmenfarbe** sind unter *4 · Einstellungen → Overlay und Brettrahmen* einstellbar (Standard: `#ff2d2d`, 5 px).
- Der Rahmen erscheint erst nach einem Snapshot mit Kalibrierung; bei internen FEN-/PGN-Stellungen ohne Aufnahme bleibt er aus.

### Wenn das Overlay „nicht funktioniert“

1. **Overlay testen** (Start-Tab oder Einstellungen) zeigt Panel, Rahmen und einen Demo-Pfeil ohne Engine.
2. Statuszeile im Overlay-Panel nennt den Grund: keine Analyse, kein Rahmen oder Overlay ausgeblendet.
3. Windows-Skalierung (125/150/200 %): Die Umrechnung von Aufnahmepixeln in Fensterkoordinaten erfolgt automatisch über die DPI-Faktoren der Monitore. Falls Pfeile dennoch versetzt sind – vor allem bei **gemischter Skalierung mehrerer Monitore** –, unter *4 · Einstellungen → DPI-Korrektur* denselben Prozentwert wie die Windows-Skalierung einstellen (`0 %` = automatisch).
4. Der Hotkey schaltet das Overlay nur ein/aus; die GUI-Schaltfläche funktioniert unabhängig davon.

Bildschirmpfeile erscheinen nur für einen analysierten, kalibrierten Snapshot. Bei internen FEN-/PGN-Stellungen zeigt das Overlay nur das Analyse-Panel und – falls kalibriert – den Rahmen.

## Programmgröße und Schriftgröße

*4 · Einstellungen → Programmgröße und Darstellung*:

- **Fenstergröße:** Breite/Höhe in Pixeln eintragen und **Fenstergröße anwenden** wählen (wird gespeichert).
- **Voreinstellungen:** *Klein* (1100 × 760), *Mittel* (1380 × 900), *Groß* (1680 × 1050).
- **Beim Start maximiert:** startet bildschirmfüllend.
- **Schriftgröße:** 70–200 % wirken auf alle Bedienelemente, inklusive Tabellen; `Strg ++`, `Strg +-` und `Strg +0` ändern sie direkt.
- Auch das Menü **Ansicht** enthält diese Funktionen, zusätzlich *Aktuelle Fenstergröße übernehmen*.

Alle Werte landen in der Konfiguration (`ui`-Abschnitt) und gelten beim nächsten Start.

## Eigenen Figurensatz anlernen (empfohlen für chess.com & Co.)

Das mitgelieferte Modell kennt nur die Zeichnungen des Programms. Für echte Screenshots – etwa chess.com oder lichess – ist der mitgelieferte Figurensatz zu verschieden. Deshalb einmalig anlernen:

1. Auf der Schachseite die **Grundstellung** anzeigen (alle Figuren auf ihren Startfeldern).
2. Ausschnitt um das Brett wählen und **Snapshot aufnehmen**; der Rahmen muss stimmen.
3. **Figurensatz anlernen** klicken – im Start-Tab unter *Schnellzugriff* oder unter *4 · Einstellungen → Eigener Figurensatz*.

Das Programm schneidet die 32 Figuren frei, normiert sie auf 32 × 32 Pixel und legt sie unter `%APPDATA%\StockfishVisionChessAI\templates\piece_templates.npz` ab. Ab dann prüft die Erkennung zuerst diese Vorlagen (Quelle „vorlage“ in der Tabelle) und greift nur bei Unsicherheit auf das Modell zurück. **Gelernte Figurenvorlagen löschen** entfernt die Datei wieder.

> Der Vergleich ist größen- und positionsunabhängig, weil jede Figur auf ihre Bounding-Box beschnitten wird – ein anderes Brett-Layout oder eine andere Fenstergröße ändert die Trefferquote praktisch nicht. Wird der Figurensatz der Seite gewechselt (anderes Design/Theme), einmal neu anlernen.

## Versionen (v1, v2, v4, v5, v6, v7, v8, …)

Jede neue Fassung erhält die nächste laufende Nummer. Die Version steht zentral in `src/stockfish_vision/version.py` und erscheint

- im Fenstertitel (`Stockfish Vision Chess AI v8 – Trainingsmodus`),
- im Titel des Einrichtungs-Popups,
- groß im Start-Tab,
- unter **Hilfe → Version / Info** samt Änderungsliste.

Für eine neue Fassung genügen dort `APP_VERSION` (z. B. `"v8"`) und ein neuer Eintrag in `VERSION_HISTORY`; `pyproject.toml` erhält die passende `version`.

- `v1` – Grundfassung
- `v2` – Start-Tab, Größen/Schrift, roter Rahmen, Overlay-DPI
- `v4` – Popup-Einrichtung, Maus-Ausschnitt, Vordergrund-Wächter (**v3 übersprungen**)
- `v5` – Größenkorrektur: Auswahl über den ganzen Desktop, Popup breiter und frei skalierbar
- `v6` – helles Design (Weiß/Grau/Schwarz), Auswahl-Ebene je Monitor über den ganzen Bildschirm, automatische Trainings-Zustimmung
- `v7` – Live-Tracking der Bildschirmstellung ohne Engine-Automatik, Plausibilitätskorrektur gegen „zu viele Könige“ und ähnliche Fehlererkennungen
- `v8` – eigener Figurensatz lernbar (echte Screenshots statt Glyphen), Overlay-Rahmen mit physischen Monitorrechtecken (gemischte DPI-Skalierung)

> Hinweis zu den Formulierungen: Das Popup ersetzt nicht die Kalibrierung, sondern führt zu ihr hin – der Ausschnitt wird per Maus gesetzt, danach rastet die automatische Brettecken-Erkennung ein.

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

Ein kompatibles alternatives OpenCV-XML kann unter **4 · Einstellungen → Figurenmodell** gewählt werden. Es muss dieselben zwölf Klassen und 324 HOG-Merkmale verwenden. Nach einem Modellwechsel wird der Klassifikator neu aufgebaut.

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

Die Tests decken unter anderem Koordinaten, Perspektivkalibrierung, automatische Brettecken-Erkennung, DPI-Umrechnung des Overlays, Maus-Ausschnitt inklusive DPI-Umrechnung, Desktop-Geometrie mehrerer Monitore, Popup-Zustände, Popup-Breite und Vordergrund-Wächter, FEN, Dateiimport/-export, PGN, Konfiguration, Hotkey-Parsing, Fairness-Sperren, SVM-Voting und variable Konfidenzscores, Belegung, fehlende Modelle, Templates und UCI mit einer echten lokalen Fake-UCI-Engine ab. Die Fake-Engine ist ausdrücklich **kein Stockfish** und dient nur der Protokollprüfung.

`tests/test_gui_smoke.py` erzeugt auf einem System mit vollständigen nativen Qt-Bibliotheken ein Offscreen-Fenster und prüft Brettkoordinaten, den Hauptfenster-Lebenszyklus, die automatische Einrahmung, den Overlay-Test, Fenster- und Schriftgröße, die Popup-Breite sowie den kompletten Popup-Ablauf vom Maus-Ausschnitt bis zur Analyse. Auf minimalen Headless-Systemen ohne `libGL` wird dieses Modul übersprungen; das ist kein bestandener GUI-Test.

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
- **Overlay falsch platziert:** nach Verschieben des Bildschirmbretts Snapshot und Kalibrierung erneuern. Multi-Monitor-/DPI-Einstellungen nach einer Änderung ebenfalls neu prüfen; notfalls **DPI-Korrektur** manuell setzen.
- **Overlay zeigt nichts:** **Overlay testen** wählen. Erscheint nur das Panel, fehlt die Kalibrierung (Rahmen) bzw. eine Analyse (Pfeile).
- **Automatischer Rahmen trifft nicht:** Brett vollständig im Aufnahmebereich zeigen, Rahmenkontrast erhöhen oder Ecken per Klick/Rechteck setzen.
- **Schrift zu klein/groß:** *4 · Einstellungen → Schriftgröße* oder `Strg ++` / `Strg +-`.
- **Popup verschwunden:** `Strg+Umschalt+H` oder *Start → Popup einblenden*.
- **Ausschnitt sitzt falsch:** erneut „Bereich mit der Maus wählen“; der Rahmen wird bei jeder Auswahl neu gesetzt.
- **Auswahl-Ebene erreicht den Bildschirmrand nicht:** in v5 behoben. Sie bedeckt jetzt alle Monitore samt Rand. Das Log (`%APPDATA%\StockfishVisionChessAI\logs\`) protokolliert die tatsächliche Auswahlgröße.
- **Popup zu schmal/breit:** rechten Fensterrand ziehen oder *4 · Einstellungen → Popup-Breite* setzen.
- **Auswahl bedeckt nur einen Teil / einen Monitor:** in v6 behoben. Das Programm legt je Monitor eine eigene Auswahl-Ebene an; das Log nennt Anzahl und Rechtecke der Ebenen. Bleibt eine Stelle unbedeckt, Skalierung prüfen und erneut starten.
- **Figuren werden falsch erkannt (z. B. bei chess.com):** in v8 über das Anlernen gelöst. Einmal die Grundstellung zeigen, Snapshot aufnehmen und **Figurensatz anlernen** klicken. Ohne diesen Schritt arbeitet nur das generische Modell, das auf fremde Figurendesigns schlecht passt.
- **Roter Rahmen im Overlay sitzt versetzt:** in v8 behoben. Die Umrechnung nutzt jetzt logische *und* physische Monitorrechtecke; bei gemischter Skalierung stimmt der Rahmen auch auf dem zweiten Monitor. Bleibt eine Restverschiebung, *4 · Einstellungen → DPI-Korrektur* auf „automatisch“ (0) stellen und das Log prüfen – es nennt die erkannten Monitorrechtecke.
- **„Zu viele Könige auf dem Brett“ / ungültige Figurenzahl:** in v7 behoben. Die Erkennung korrigiert höchstens einen König je Seite (Stammfeld e1/e8 bzw. Grundreihe gewinnt), kappt Bauern auf acht und ersetzt Bauern auf der Grundreihe. Jede Änderung steht als „Plausibilität korrigiert: feld: alt → neu“ im Prüf-Tab; bleibt die Stellung danach ungültig, das betroffene Feld per Klick korrigieren.
- **Live-Tracking reagiert nicht:** Es braucht einen eingerahmten Brettbereich (vier Ecken) und eine sichtbare Änderung am Brett. Ist der Rahmen gesetzt und passiert nichts, das Intervall unter *4 · Einstellungen → Live-Tracking-Intervall* kleiner stellen; bei Dauerregen von Fehlalarmen größer. Bei starker Bewegung im Bild (Videos, Animationen) die Belegungsschwelle erhöhen.
- **Live-Tracking erkennt, aber die Engine bleibt still:** Das ist beabsichtigt. Das Tracking ruft die Engine nie selbst auf; für eine Bewertung **Snapshot aufnehmen und einfrieren** und diesen analysieren.
- **Zustimmung zum Trainingsmodus:** wird automatisch erteilt und mit Zeitstempel in `config.json` (`training_acknowledged`, `training_acknowledged_at`) festgehalten – es gibt keine Checkbox mehr.
- **Browser verdrängt Popup:** Fenstermodus statt `F11`-Vollbild verwenden; der Wächter holt das Popup alle zwei Sekunden zurück.

## Projektstruktur

```text
src/stockfish_vision/
  capture/       Monitor-/Bereichsaufnahme via mss
  board/         Perspektive, 64 Felder, Orientierung
  recognition/   Belegung, HOG-SVM, Scores, Session-Templates
  chesslogic/    FEN, PGN, Validierung, taktische Hinweise
  engine/        UCI-Client und Ergebnisdaten
  board/detect.py automatische Brettecken-Erkennung
  gui/           PySide6-Fenster, Theme/Schriftgröße, Hintergrund-Worker
  version.py     Versionsschema v1, v2, …
  overlay/       Pfeilebene, roter Rahmen, Popup, Maus-Auswahl, Vordergrund-Wächter, Hotkey
  assets/models/ mitgeliefertes OpenCV-Modell
  config.py      validierte JSON-Einstellungen
  policy.py      Trainings-/Snapshot-/PGN-Freigaben
scripts/         reproduzierbares Modelltraining
tests/           Kern-, Protokoll- und GUI-Smoke-Tests
```

## Lizenz und Fairness

Anwendungscode: MIT, siehe `LICENSE`. Stockfish: separat und GPLv3. Die Anwendung ist für Training, eigene Stellungen und abgeschlossene Partien bestimmt. Die technische Trennung von Vorschau und manueller Analyse ist bewusst erhalten geblieben.

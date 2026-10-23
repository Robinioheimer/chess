from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import chess
import cv2
import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QCloseEvent,
    QColor,
    QGuiApplication,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from stockfish_vision.board.detect import detect_board_corners, rectangle_to_corners
from stockfish_vision.board.geometry import (
    board_mean_abs_diff,
    extract_square,
    square_to_image_rc,
    validate_corners,
    warp_board,
)
from stockfish_vision.capture.screen import list_monitors
from stockfish_vision.chesslogic.fenutil import (
    apply_manual_piece,
    describe_validity,
    load_fen,
    parse_fen,
    save_fen,
    update_board_state,
)
from stockfish_vision.chesslogic.pgnutil import (
    game_from_board,
    game_is_completed,
    game_label,
    game_positions,
    load_pgn_games,
    save_pgn_game,
)
from stockfish_vision.config import AppConfig, OverlaySettings, save_config
from stockfish_vision.gui.board_widget import BoardWidget
from stockfish_vision.gui.theme import build_stylesheet, clamp_font_scale
from stockfish_vision.gui.worker import CaptureWorker, EngineWorker, RecognitionResult, RecognitionWorker
from stockfish_vision.overlay.helper import HelperPopup
from stockfish_vision.overlay.hotkey import WindowsHotkeyFilter
from stockfish_vision.overlay.mapping import logical_rect_to_region
from stockfish_vision.overlay.selector import RegionSelector
from stockfish_vision.overlay.topmost import TopmostKeeper, ensure_topmost
from stockfish_vision.overlay.window import OverlayWindow
from stockfish_vision.policy import LiveGameGuard, can_analyze
from stockfish_vision.recognition.pieces import Detection
from stockfish_vision.version import (
    VERSION_HISTORY,
    full_version_label,
    version_label,
    window_title,
)

logger = logging.getLogger(__name__)

PIECE_CHOICES = [
    ("(leer)", None),
    ("Weißer Bauer (P)", "P"),
    ("Weißer Springer (N)", "N"),
    ("Weißer Läufer (B)", "B"),
    ("Weißer Turm (R)", "R"),
    ("Weiße Dame (Q)", "Q"),
    ("Weißer König (K)", "K"),
    ("Schwarzer Bauer (p)", "p"),
    ("Schwarzer Springer (n)", "n"),
    ("Schwarzer Läufer (b)", "b"),
    ("Schwarzer Turm (r)", "r"),
    ("Schwarze Dame (q)", "q"),
    ("Schwarzer König (k)", "k"),
]


class CalibrationView(QLabel):
    """Vorschaubild der Aufnahme.

    Ein einzelner Klick setzt eine Brettecke; mit gedrückter Maustaste lässt
    sich alternativ direkt ein Rechteck um das Brett aufziehen.
    """

    corner_clicked = Signal(float, float)
    rect_selected = Signal(float, float, float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(300, 240)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#111; color:#ccc;")
        self.setText("Noch kein Screenshot. Snapshot aufnehmen – das Brett wird rot eingerahmt.")
        self._image: np.ndarray | None = None
        self._qimage: QImage | None = None
        self._press_position: tuple[float, float] | None = None
        self._rubber_start = None
        self._rubber_current = None

    def set_frame(self, bgr: np.ndarray) -> None:
        self._image = bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        height, width, channels = rgb.shape
        self._qimage = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._qimage is not None:
            self.setPixmap(
                QPixmap.fromImage(self._qimage).scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_pixmap()

    def _to_image(self, position) -> tuple[float, float] | None:
        if self._image is None or self.pixmap() is None:
            return None
        pixmap = self.pixmap()
        pw, ph = pixmap.width(), pixmap.height()
        x0 = (self.width() - pw) // 2
        y0 = (self.height() - ph) // 2
        mx, my = float(position.x()), float(position.y())
        if not (x0 <= mx < x0 + pw and y0 <= my < y0 + ph):
            return None
        height, width = self._image.shape[:2]
        return (mx - x0) / pw * width, (my - y0) / ph * height

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._to_image(event.position())
        if point is None:
            return
        self._press_position = point
        self._rubber_start = event.position()
        self._rubber_current = event.position()
        self.update()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._rubber_start is None:
            return
        self._rubber_current = event.position()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._press_position is None:
            return
        start_image = self._press_position
        end_image = self._to_image(event.position())
        start_widget = self._rubber_start
        end_widget = self._rubber_current
        self._press_position = None
        self._rubber_start = None
        self._rubber_current = None
        self.update()
        if end_image is None or start_widget is None or end_widget is None:
            return
        dragged = (
            abs(end_widget.x() - start_widget.x()) > 12
            or abs(end_widget.y() - start_widget.y()) > 12
        )
        if dragged:
            self.rect_selected.emit(start_image[0], start_image[1], end_image[0], end_image[1])
        else:
            self.corner_clicked.emit(start_image[0], start_image[1])

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        if self._rubber_start is None or self._rubber_current is None:
            return
        painter = QPainter(self)
        pen = QPen(QColor(255, 45, 45), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QColor(255, 45, 45, 40))
        painter.drawRect(QRectF(self._rubber_start, self._rubber_current).normalized())


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.setWindowTitle(window_title())
        self._ui_geometry_applied = False
        self._initial_geometry()

        self.guard = LiveGameGuard()
        self.snapshot_confirmed = False
        self.snapshot_recognized = False
        self.mode = "training"
        self.live_preview = False
        self.last_warped: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self.display_frame: np.ndarray | None = None
        self.preview_warped: np.ndarray | None = None
        self.pending_warped: np.ndarray | None = None
        self.capture_origin = (0.0, 0.0)
        self.corners: list[list[float]] = deepcopy(config.capture.corners or [])
        self.detections: list[Detection] = []
        self.last_result = None
        self.capture_in_flight = False
        self.recognition_in_flight = False
        self.analysis_in_flight = False
        self.engine_start_in_flight = False
        self.engine_ready = False
        self.hotkey_filter: WindowsHotkeyFilter | None = None
        self.pgn_games: list[chess.pgn.Game] = []
        self.pgn_positions: list[tuple[str, str, chess.Board]] = []
        self.selector: RegionSelector | None = None
        self._chain_recognize = False
        self._pending_analysis = False
        self._helper_offset = (config.ui.helper_x, config.ui.helper_y)

        try:
            self.board = parse_fen(config.last_fen)
        except ValueError as exc:
            logger.warning("Gespeicherte FEN ist ungültig: %s", exc)
            self.board = chess.Board()

        self.overlay = OverlayWindow()
        self.overlay.apply_settings(config.overlay)
        self.overlay.panel_settings_changed.connect(self._overlay_panel_changed)
        self.overlay.cover_virtual_desktop()
        self.overlay.user_hidden = not config.overlay.visible
        self.overlay.set_click_through(True)
        self.overlay.set_status("Noch keine Analyse. Engine starten und Stellung analysieren.")
        if config.overlay.visible:
            self.overlay.present()

        self.topmost = TopmostKeeper(parent=self)
        self.helper = HelperPopup(version_label(), None)
        self.helper.select_region_requested.connect(self._select_region_with_mouse)
        self.helper.snapshot_requested.connect(self._helper_snapshot)
        self.helper.analyze_requested.connect(self._helper_analyze)
        self.helper.settings_requested.connect(self._helper_open_settings)
        self.helper.overlay_test_requested.connect(self._test_overlay)
        self.helper.position_changed.connect(self._helper_moved)
        self.helper.size_changed.connect(self._helper_resized)
        self.helper.hidden_by_user.connect(self._helper_hidden)
        self.helper.apply_width(config.ui.helper_width)
        self._place_helper()
        self.helper.setVisible(config.ui.helper_visible)
        if config.ui.helper_visible:
            self.helper.show_helper()
        self.topmost.watch(self.helper)
        self.topmost.watch(self.overlay)
        self.topmost.watch(self.overlay.panel)
        self.topmost.start()

        self.engine_thread = QThread(self)
        self.worker = EngineWorker()
        self.worker.moveToThread(self.engine_thread)
        self.worker.request_start.connect(self.worker.start_engine, Qt.ConnectionType.QueuedConnection)
        self.worker.request_analyse.connect(self.worker.analyse, Qt.ConnectionType.QueuedConnection)
        self.worker.request_stop.connect(self.worker.stop_engine, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self._on_analysis)
        self.worker.failed.connect(self._on_engine_error)
        self.worker.started.connect(self._on_engine_started)
        self.worker.stopped.connect(self._on_engine_stopped)
        self.engine_thread.finished.connect(self.worker.client.stop)
        self.engine_thread.start()

        self.capture_thread = QThread(self)
        self.capture_worker = CaptureWorker()
        self.capture_worker.moveToThread(self.capture_thread)
        self.capture_worker.request_grab.connect(self.capture_worker.grab, Qt.ConnectionType.QueuedConnection)
        self.capture_worker.finished.connect(self._on_capture)
        self.capture_worker.failed.connect(self._on_capture_error)
        self.capture_thread.start()

        self.recognition_thread = QThread(self)
        self.recognition_worker = RecognitionWorker(
            config.occupancy_std_threshold,
            config.min_piece_confidence,
            config.model_path,
        )
        self.recognition_worker.moveToThread(self.recognition_thread)
        self.recognition_worker.request_recognize.connect(
            self.recognition_worker.recognize, Qt.ConnectionType.QueuedConnection
        )
        self.recognition_worker.request_correction.connect(
            self.recognition_worker.remember_correction, Qt.ConnectionType.QueuedConnection
        )
        self.recognition_worker.request_configure.connect(
            self.recognition_worker.configure, Qt.ConnectionType.QueuedConnection
        )
        self.recognition_worker.finished.connect(self._on_recognition)
        self.recognition_worker.failed.connect(self._on_recognition_error)
        self.recognition_thread.start()

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(700)
        self.preview_timer.timeout.connect(self._tick_preview)

        self._build_ui()
        self._refresh_monitors()
        self._load_settings_into_ui()
        self._apply_font_scale(config.ui.font_scale_percent, announce=False)
        self._refresh_region_label()
        self._set_board(self.board)
        self._update_overlay_frame()
        hotkey_available = self._install_hotkey()
        self._update_mode_banner()
        self._update_action_states()
        self.statusBar().showMessage(
            "Bereit. Trainings- und Nachbereitungsmodus; keine Live-Online-Hilfe."
            if hotkey_available
            else "Bereit. Globaler Hotkey nicht verfügbar; Overlay-Schaltfläche nutzen. Details im Log."
        )

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(8, 8, 8, 8)
        self.mode_banner = QLabel()
        self.mode_banner.setWordWrap(True)
        self.mode_banner.setStyleSheet(
            "background:#234d35; color:#e6ffed; padding:9px; border-radius:5px; font-weight:600;"
        )
        central_layout.addWidget(self.mode_banner)
        self.tabs = QTabWidget()
        self.overview_tab = self._overview_tab()
        self.capture_tab = self._capture_tab()
        self.training_tab = self._training_tab()
        self.analysis_tab = self._analysis_tab()
        self.engine_tab = self._settings_tab()
        self.pgn_tab = self._pgn_tab()
        self.tabs.addTab(self.overview_tab, "Start")
        self.tabs.addTab(self.capture_tab, "1 · Aufnahme")
        self.tabs.addTab(self.training_tab, "2 · Prüfen")
        self.tabs.addTab(self.analysis_tab, "3 · Analyse")
        self.tabs.addTab(self.engine_tab, "4 · Einstellungen")
        self.tabs.addTab(self.pgn_tab, "5 · PGN")
        self.tabs.addTab(self._history_tab(), "6 · Historie")
        central_layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        file_menu = self.menuBar().addMenu("Datei")
        for text, callback in (
            ("FEN importieren…", self._import_fen),
            ("FEN exportieren…", self._export_fen),
            ("PGN importieren…", self._load_pgn),
            ("PGN exportieren…", self._export_pgn),
        ):
            action = QAction(text, self)
            action.triggered.connect(callback)
            file_menu.addAction(action)
        file_menu.addSeparator()
        save_act = QAction("Konfiguration speichern", self)
        save_act.triggered.connect(self._save_config)
        file_menu.addAction(save_act)
        quit_act = QAction("Beenden", self)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = self.menuBar().addMenu("Ansicht")
        for label, width, height in (
            ("Klein (1100 × 760)", 1100, 760),
            ("Mittel (1380 × 900)", 1380, 900),
            ("Groß (1680 × 1050)", 1680, 1050),
        ):
            action = QAction(label, self)
            action.triggered.connect(
                lambda _checked=False, w=width, h=height: self._apply_size_preset(w, h)
            )
            view_menu.addAction(action)
        max_act = QAction("Maximiert starten", self)
        max_act.setCheckable(True)
        max_act.setChecked(self.config.ui.start_maximized)
        max_act.toggled.connect(self._set_start_maximized)
        self.maximized_action = max_act
        view_menu.addAction(max_act)
        view_menu.addSeparator()
        bigger = QAction("Schrift größer", self)
        bigger.setShortcut("Ctrl+=")
        bigger.triggered.connect(lambda: self._adjust_font_scale(10))
        smaller = QAction("Schrift kleiner", self)
        smaller.setShortcut("Ctrl+-")
        smaller.triggered.connect(lambda: self._adjust_font_scale(-10))
        reset = QAction("Schrift Standard", self)
        reset.setShortcut("Ctrl+0")
        reset.triggered.connect(lambda: self._apply_font_scale(100))
        view_menu.addAction(bigger)
        view_menu.addAction(smaller)
        view_menu.addAction(reset)
        view_menu.addSeparator()
        helper_act = QAction("Einrichtungs-Popup anzeigen", self)
        helper_act.setCheckable(True)
        helper_act.setChecked(self.config.ui.helper_visible)
        helper_act.setShortcut("Ctrl+Shift+H")
        helper_act.toggled.connect(self._toggle_helper_from_menu)
        self.helper_action = helper_act
        view_menu.addAction(helper_act)
        pick_act = QAction("Ausschnitt mit der Maus wählen", self)
        pick_act.setShortcut("Ctrl+Shift+R")
        pick_act.triggered.connect(self._select_region_with_mouse)
        view_menu.addAction(pick_act)
        view_menu.addSeparator()
        save_size = QAction("Aktuelle Fenstergröße übernehmen", self)
        save_size.triggered.connect(self._capture_window_size)
        view_menu.addAction(save_size)

        help_menu = self.menuBar().addMenu("Hilfe")
        fairness_act = QAction("Trainingsmodus und Fairness", self)
        fairness_act.triggered.connect(self._show_fairness)
        help_menu.addAction(fairness_act)
        version_act = QAction(f"Version / Info ({version_label()})", self)
        version_act.triggered.connect(self._show_version_info)
        help_menu.addAction(version_act)

    def _overview_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        title = QLabel(full_version_label())
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #79d99a;")
        subtitle = QLabel(
            "Geführter Ablauf in vier Schritten. Alles bleibt manuell ausgelöst – "
            "keine automatische Live-Analyse laufender Online-Partien."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.overview_status = QLabel()
        self.overview_status.setWordWrap(True)
        self.overview_status.setStyleSheet(
            "background:#1b2026; border:1px solid #2a313a; border-radius:6px; padding:10px;"
        )
        layout.addWidget(self.overview_status)

        steps = QGroupBox("So funktioniert es")
        steps_layout = QGridLayout(steps)
        entries = (
            ("1", "Bildschirmbereich wählen", "Monitor oder Bereich einstellen und einen Snapshot aufnehmen.",
             "Aufnahme öffnen", lambda: self.tabs.setCurrentWidget(self.capture_tab)),
            ("2", "Brett einrahmen", "Bereich mit der Maus um das Spielfeld ziehen – das Brett wird automatisch rot eingerahmt und erkannt.",
             "Ausschnitt wählen", self._select_region_with_mouse),
            ("3", "Erkennung prüfen", "Unsicheres Feld anklicken und die Figur manuell korrigieren.",
             "Prüfen öffnen", lambda: self.tabs.setCurrentWidget(self.training_tab)),
            ("4", "Analysieren", "Engine starten und die geprüfte Stellung analysieren; das Overlay zeigt Rahmen und Pfeile.",
             "Engine und Analyse", lambda: self.tabs.setCurrentWidget(self.analysis_tab)),
        )
        for row, (number, headline, description, button_text, callback) in enumerate(entries):
            number_label = QLabel(number)
            number_label.setFixedWidth(26)
            number_label.setStyleSheet(
                "font-size: 18px; font-weight: 700; color:#79d99a;"
            )
            text = QLabel(f"<b>{headline}</b><br>{description}")
            text.setWordWrap(True)
            button = QPushButton(button_text)
            button.clicked.connect(callback)
            steps_layout.addWidget(number_label, row, 0, Qt.AlignmentFlag.AlignTop)
            steps_layout.addWidget(text, row, 1)
            steps_layout.addWidget(button, row, 2)
        steps_layout.setColumnStretch(1, 1)
        layout.addWidget(steps)

        quick = QGroupBox("Schnellzugriff")
        quick_layout = QHBoxLayout(quick)
        popup_button = QPushButton("Popup einblenden")
        popup_button.setToolTip("Kleines Helfer-Fenster, das durch die Einrichtung führt.")
        popup_button.clicked.connect(self._show_helper)
        pick_button = QPushButton("Ausschnitt mit der Maus wählen")
        pick_button.clicked.connect(self._select_region_with_mouse)
        overlay_test = QPushButton("Overlay testen")
        overlay_test.setToolTip("Zeigt Rahmen, Demo-Pfeil und Panel ohne Engine-Analyse.")
        overlay_test.clicked.connect(self._test_overlay)
        overlay_toggle = QPushButton("Overlay ein/aus")
        overlay_toggle.clicked.connect(self._toggle_overlay)
        settings_button = QPushButton("Größe und Einstellungen")
        settings_button.clicked.connect(lambda: self.tabs.setCurrentWidget(self.engine_tab))
        quick_layout.addWidget(popup_button)
        quick_layout.addWidget(pick_button)
        quick_layout.addWidget(overlay_test)
        quick_layout.addWidget(overlay_toggle)
        quick_layout.addWidget(settings_button)
        quick_layout.addStretch()
        layout.addWidget(quick)

        note = QLabel(
            f"Version {version_label()} · Neue Fassungen heißen v1, v2, v3 … "
            "Die Nummer steht im Fenstertitel und unter Hilfe → Version / Info."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#9fb0c0;")
        layout.addWidget(note)
        layout.addStretch()
        return widget

    def _refresh_overview(self) -> None:
        if not hasattr(self, "overview_status"):
            return
        overlay_state = "aus" if self.overlay.user_hidden else "an"
        engine_state = "bereit" if self.engine_ready else "nicht gestartet"
        corners = f"{len(self.corners)}/4 Ecken"
        snapshot = "vorhanden" if self.last_frame is not None else "keins"
        frame = "an" if self.config.overlay.show_frame else "aus"
        auto_frame = "an" if self.config.capture.auto_frame else "aus"
        self.overview_status.setText(
            "   ·   ".join(
                (
                    f"Fenster {self.width()}×{self.height()}",
                    f"Schrift {self.config.ui.font_scale_percent} %",
                    f"Monitor {self.config.capture.monitor_index}",
                    f"Kalibrierung {corners}",
                    f"Snapshot {snapshot}",
                    f"Engine {engine_state}",
                    f"Overlay {overlay_state}",
                    f"Roter Rahmen {frame} (auto {auto_frame})",
                    f"Popup {'sichtbar' if self.helper.isVisible() else 'ausgeblendet'}",
                )
            )
        )

    def _capture_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        splitter = QSplitter()

        left = QWidget()
        form = QVBoxLayout(left)

        source = QGroupBox("1 · Bildschirmausschnitt")
        source_layout = QVBoxLayout(source)
        self.btn_pick_region = QPushButton("Bereich mit der Maus wählen")
        self.btn_pick_region.setToolTip(
            "Blendet den Bildschirm ab: Rahmen mit gedrückter Maustaste um das Spielfeld "
            "ziehen. Esc bricht ab."
        )
        self.btn_pick_region.clicked.connect(self._select_region_with_mouse)
        self.region_label = QLabel()
        self.region_label.setWordWrap(True)
        region_hint = QLabel(
            "Der Bildschirm wird kurz abgedunkelt – Rahmen um das Spielfeld ziehen. "
            "Danach wird das Brett automatisch rot eingerahmt und erkannt."
        )
        region_hint.setWordWrap(True)
        region_hint.setStyleSheet("color:#8a97a3;")
        source_layout.addWidget(self.btn_pick_region)
        source_layout.addWidget(self.region_label)
        source_layout.addWidget(region_hint)

        self.monitor_combo = QComboBox()
        refresh = QPushButton("Monitore aktualisieren")
        refresh.clicked.connect(self._refresh_monitors)
        self.use_region = QCheckBox("Gespeicherten Ausschnitt verwenden")
        self.region_left = QSpinBox()
        self.region_top = QSpinBox()
        self.region_width = QSpinBox()
        self.region_height = QSpinBox()
        for spin in (self.region_left, self.region_top):
            spin.setRange(-30_000, 30_000)
        for spin in (self.region_width, self.region_height):
            spin.setRange(16, 16_384)
        self.region_width.setValue(800)
        self.region_height.setValue(800)
        region_row = QGridLayout()
        for index, (label, spin) in enumerate(
            (
                ("Links (L)", self.region_left),
                ("Oben (T)", self.region_top),
                ("Breite (B)", self.region_width),
                ("Höhe (H)", self.region_height),
            )
        ):
            region_row.addWidget(QLabel(label), index // 2, (index % 2) * 2)
            region_row.addWidget(spin, index // 2, (index % 2) * 2 + 1)

        self.advanced_group = QGroupBox("Erweitert: Monitor und Pixelwerte")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        advanced_body = QWidget()
        advanced_layout = QVBoxLayout(advanced_body)
        advanced_layout.setContentsMargins(0, 4, 0, 0)
        advanced_layout.addWidget(QLabel("Monitor (ohne eigenen Ausschnitt)"))
        advanced_layout.addWidget(self.monitor_combo)
        advanced_layout.addWidget(refresh)
        advanced_layout.addWidget(self.use_region)
        advanced_layout.addLayout(region_row)
        group_layout = QVBoxLayout(self.advanced_group)
        group_layout.setContentsMargins(8, 4, 8, 8)
        group_layout.addWidget(advanced_body)
        self.advanced_body = advanced_body
        advanced_body.setVisible(False)
        self.advanced_group.toggled.connect(advanced_body.setVisible)
        source_layout.addWidget(self.advanced_group)
        form.addWidget(source)

        calibration = QGroupBox("2 · Brett einrahmen")
        calibration_layout = QVBoxLayout(calibration)
        calibration_hint = QLabel(
            "Automatisch: Das Brett wird nach dem Snapshot rot eingerahmt.\n"
            "Manuell: vier Ecken klicken (oben links zuerst) oder mit gedrückter "
            "Maustaste ein Rechteck um das Brett aufziehen."
        )
        calibration_hint.setWordWrap(True)
        calibration_layout.addWidget(calibration_hint)
        self.auto_frame_check = QCheckBox("Beim Snapshot automatisch rot einrahmen")
        self.auto_frame_check.setChecked(True)
        self.auto_frame_check.toggled.connect(self._auto_frame_toggled)
        self.btn_auto_frame = QPushButton("Brett jetzt automatisch einrahmen")
        self.btn_auto_frame.clicked.connect(self._auto_frame_and_show)
        self.btn_clear_corners = QPushButton("Rahmen / Ecken zurücksetzen")
        self.btn_clear_corners.clicked.connect(self._reset_corners)
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["white_bottom", "white_top", "white_left", "white_right"])
        self.warp_size_spin = QSpinBox()
        self.warp_size_spin.setRange(128, 1600)
        self.warp_size_spin.setSingleStep(64)
        calibration_layout.addWidget(self.auto_frame_check)
        calibration_layout.addWidget(self.btn_auto_frame)
        calibration_layout.addWidget(self.btn_clear_corners)
        calibration_layout.addWidget(QLabel("Brettorientierung (aus Sicht des Bildes)"))
        calibration_layout.addWidget(self.orientation_combo)
        calibration_layout.addWidget(QLabel("Erkennungsgröße (quadratisch)"))
        calibration_layout.addWidget(self.warp_size_spin)
        form.addWidget(calibration)

        actions = QGroupBox("3 · Aufnahme und Analyse")
        actions_layout = QVBoxLayout(actions)
        self.btn_preview = QPushButton("Vorschau starten")
        self.btn_preview.clicked.connect(self._toggle_preview)
        self.btn_snapshot = QPushButton("Snapshot aufnehmen und einfrieren")
        self.btn_snapshot.clicked.connect(self._take_snapshot)
        self.confirm_box = QCheckBox(
            "Keine laufende Online-Partie (nur Training/Nachbereitung)"
        )
        self.confirm_box.setToolTip(
            "Bestätigt eine eigene Trainingsstellung, einen eingefrorenen Snapshot "
            "oder eine abgeschlossene Partie – keine Hilfe während laufender Online-Partien."
        )
        self.confirm_box.stateChanged.connect(self._on_confirm_changed)
        self.btn_recognize = QPushButton("Snapshot im Hintergrund erkennen")
        self.btn_recognize.clicked.connect(self._recognize_snapshot)
        self.btn_review = QPushButton("Erkennung prüfen und korrigieren")
        self.btn_review.clicked.connect(lambda: self.tabs.setCurrentWidget(self.training_tab))
        self.btn_analyze_snap = QPushButton("Geprüften Snapshot analysieren")
        self.btn_analyze_snap.clicked.connect(self._analyze_snapshot)
        self.capture_status = QLabel(
            "Keine Aufnahme. Vorschau und Erkennung starten keine Engine automatisch."
        )
        self.capture_status.setWordWrap(True)
        actions_layout.addWidget(self.btn_preview)
        actions_layout.addWidget(self.btn_snapshot)
        actions_layout.addWidget(self.confirm_box)
        actions_layout.addWidget(self.btn_recognize)
        actions_layout.addWidget(self.btn_review)
        actions_layout.addWidget(self.btn_analyze_snap)
        actions_layout.addWidget(self.capture_status)
        form.addWidget(actions)
        form.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        calibration_caption = QLabel(
            "Kalibrieransicht: rot = gewählter Brettrahmen. Klick = Ecke, Ziehen = Rechteck."
        )
        calibration_caption.setWordWrap(True)
        right_layout.addWidget(calibration_caption)
        self.calib_view = CalibrationView()
        self.calib_view.corner_clicked.connect(self._add_corner)
        self.calib_view.rect_selected.connect(self._set_rectangle_corners)
        right_layout.addWidget(self.calib_view)

        left_scroll.setMinimumWidth(440)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([470, 930])
        layout.addWidget(splitter)
        return widget

    def _settings_tab(self) -> QWidget:
        content = QWidget()
        outer = QVBoxLayout(content)

        engine_group = QGroupBox("Stockfish-Engine")
        engine_form = QFormLayout(engine_group)
        self.engine_path = QLineEdit()
        self.engine_path.setPlaceholderText(
            "Pfad zur separat installierten stockfish.exe oder STOCKFISH_PATH"
        )
        browse = QPushButton("Durchsuchen…")
        browse.clicked.connect(self._browse_engine)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(0, 40)
        self.depth_spin.setSpecialValueText("aus (nur Zeit)")
        self.time_spin = QSpinBox()
        self.time_spin.setRange(0, 60_000)
        self.time_spin.setSuffix(" ms")
        self.time_spin.setSpecialValueText("0 ms (aus)")
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        self.hash_spin = QSpinBox()
        self.hash_spin.setRange(16, 16_384)
        self.hash_spin.setSuffix(" MB")
        self.multipv_spin = QSpinBox()
        self.multipv_spin.setRange(1, 5)
        self.btn_engine_start = QPushButton("Engine starten")
        self.btn_engine_start.clicked.connect(self._start_engine)
        self.btn_engine_stop = QPushButton("Engine stoppen")
        self.btn_engine_stop.clicked.connect(self._stop_engine)
        self.engine_status = QLabel("Engine nicht gestartet.")
        self.engine_status.setWordWrap(True)
        for spin in (
            self.depth_spin,
            self.time_spin,
            self.threads_spin,
            self.hash_spin,
            self.multipv_spin,
        ):
            spin.setMaximumWidth(180)
        engine_form.addRow("Stockfish.exe", self._with_button(self.engine_path, browse))
        engine_form.addRow("Suchtiefe", self.depth_spin)
        engine_form.addRow("Zeitbudget", self.time_spin)
        engine_form.addRow("Threads", self.threads_spin)
        engine_form.addRow("Hash", self.hash_spin)
        engine_form.addRow("Beste Zugfolgen (MultiPV)", self.multipv_spin)
        engine_buttons = QWidget()
        engine_buttons_layout = QHBoxLayout(engine_buttons)
        engine_buttons_layout.setContentsMargins(0, 0, 0, 0)
        engine_buttons_layout.addWidget(self.btn_engine_start)
        engine_buttons_layout.addWidget(self.btn_engine_stop)
        engine_buttons_layout.addStretch()
        engine_form.addRow(engine_buttons)
        engine_form.addRow("Status", self.engine_status)

        display_group = QGroupBox("Programmgröße und Darstellung")
        display_form = QFormLayout(display_group)
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(720, 6000)
        self.window_width_spin.setSuffix(" px")
        self.window_height_spin = QSpinBox()
        self.window_height_spin.setRange(520, 4000)
        self.window_height_spin.setSuffix(" px")
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Breite"))
        size_row.addWidget(self.window_width_spin)
        size_row.addSpacing(12)
        size_row.addWidget(QLabel("Höhe"))
        size_row.addWidget(self.window_height_spin)
        size_row.addStretch()
        self.apply_size_button = QPushButton("Fenstergröße anwenden")
        self.apply_size_button.clicked.connect(self._apply_size_from_fields)
        self.apply_size_button.setToolTip("Die aktuelle Fenstergröße wird dauerhaft gespeichert.")
        preset_row = QHBoxLayout()
        for label, width, height in (
            ("Klein", 1100, 760),
            ("Mittel", 1380, 900),
            ("Groß", 1680, 1050),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, w=width, h=height: self._apply_size_preset(w, h)
            )
            preset_row.addWidget(button)
        preset_row.addStretch()
        self.maximized_check = QCheckBox("Beim Start maximiert")
        self.maximized_check.toggled.connect(self._set_start_maximized)
        self.popup_width_spin = QSpinBox()
        self.popup_width_spin.setRange(340, 1400)
        self.popup_width_spin.setSingleStep(20)
        self.popup_width_spin.setSuffix(" px")
        self.popup_width_spin.setMaximumWidth(180)
        self.popup_width_spin.setToolTip(
            "Breite des Helfer-Popups. Alternativ am rechten Fensterrand ziehen."
        )
        self.popup_width_spin.valueChanged.connect(self._popup_width_changed)
        self.font_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_scale_slider.setRange(70, 200)
        self.font_scale_slider.setSingleStep(5)
        self.font_scale_slider.setTickInterval(10)
        self.font_scale_slider.setToolTip("Schrift- und Steuerelementgröße in Prozent")
        self.font_scale_value = QLabel("100 %")
        font_row = QHBoxLayout()
        font_row.addWidget(self.font_scale_slider, 1)
        font_row.addWidget(self.font_scale_value)
        for spin in (self.window_width_spin, self.window_height_spin):
            spin.setMaximumWidth(150)
        display_form.addRow("Fenstergröße", size_row)
        display_form.addRow(self.apply_size_button)
        display_form.addRow("Voreinstellungen", preset_row)
        display_form.addRow(self.maximized_check)
        display_form.addRow("Schriftgröße", font_row)
        display_form.addRow("Popup-Breite", self.popup_width_spin)

        recognition_group = QGroupBox("Figurerkennung")
        recognition_form = QFormLayout(recognition_group)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Leer = mitgeliefertes HOG-SVM-Modell")
        model_browse = QPushButton("Modell…")
        model_browse.clicked.connect(self._browse_model)
        self.occupancy_spin = QDoubleSpinBox()
        self.occupancy_spin.setRange(1.0, 100.0)
        self.occupancy_spin.setDecimals(1)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.05, 0.99)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setSingleStep(0.05)
        apply_recognition = QPushButton("Erkennungseinstellungen anwenden")
        apply_recognition.clicked.connect(self._apply_recognition_settings)
        for spin in (self.occupancy_spin, self.confidence_spin):
            spin.setMaximumWidth(180)
        recognition_form.addRow("Figurenmodell (XML)", self._with_button(self.model_path_edit, model_browse))
        recognition_form.addRow("Belegungsschwelle (Std.-Abw.)", self.occupancy_spin)
        recognition_form.addRow("Warnschwelle Konfidenz", self.confidence_spin)
        recognition_form.addRow(apply_recognition)

        overlay_group = QGroupBox("Overlay und Brettrahmen")
        overlay_form = QFormLayout(overlay_group)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(60, 200)
        self.frame_check = QCheckBox("Roten Rahmen um das Spielfeld zeichnen")
        self.frame_check.toggled.connect(self._frame_settings_changed)
        self.frame_width_spin = QSpinBox()
        self.frame_width_spin.setRange(1, 24)
        self.frame_width_spin.setSuffix(" px")
        self.frame_width_spin.valueChanged.connect(self._frame_settings_changed)
        self.frame_color_button = QPushButton("#ff2d2d")
        self.frame_color_button.setMaximumWidth(240)
        self.frame_color_button.clicked.connect(self._choose_frame_color)
        self.coordinate_scale_spin = QSpinBox()
        self.coordinate_scale_spin.setRange(0, 400)
        self.coordinate_scale_spin.setSuffix(" %")
        self.coordinate_scale_spin.setSpecialValueText("automatisch")
        self.coordinate_scale_spin.setToolTip(
            "Nur nötig, wenn Pfeile bei Windows-Skalierung (125/150/200 %) versetzt wirken. "
            "0 = automatische DPI-Erkennung."
        )
        self.coordinate_scale_spin.valueChanged.connect(self._frame_settings_changed)
        self.hotkey_edit = QLineEdit()
        apply_hotkey = QPushButton("Hotkey anwenden")
        apply_hotkey.clicked.connect(self._apply_hotkey)
        self.btn_overlay = QPushButton("Overlay ein/aus")
        self.btn_overlay.clicked.connect(self._toggle_overlay)
        self.btn_test_overlay = QPushButton("Overlay testen (ohne Engine)")
        self.btn_test_overlay.clicked.connect(self._test_overlay)
        for spin in (self.frame_width_spin, self.coordinate_scale_spin):
            spin.setMaximumWidth(180)
        overlay_form.addRow("Deckkraft %", self.opacity_slider)
        overlay_form.addRow("Skalierung Panel %", self.scale_slider)
        overlay_form.addRow(self.frame_check)
        overlay_form.addRow("Rahmenbreite", self.frame_width_spin)
        overlay_form.addRow("Rahmenfarbe", self.frame_color_button)
        overlay_form.addRow("DPI-Korrektur", self.coordinate_scale_spin)
        overlay_form.addRow("Globaler Hotkey", self._with_button(self.hotkey_edit, apply_hotkey))
        overlay_form.addRow(self.btn_overlay)
        overlay_form.addRow(self.btn_test_overlay)
        overlay_form.addRow(
            QLabel("Panel am Titel verschieben, an ◢ skalieren. Pfeile bleiben klickdurchlässig.")
        )

        outer.addWidget(engine_group)
        outer.addWidget(display_group)
        outer.addWidget(recognition_group)
        outer.addWidget(overlay_group)
        outer.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _training_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        self.board_widget = BoardWidget()
        self.board_widget.confidence_threshold = self.config.min_piece_confidence
        self.board_widget.square_clicked.connect(self._on_square_clicked)
        side = QVBoxLayout()
        self.fen_edit = QTextEdit()
        self.fen_edit.setFixedHeight(70)
        fen_buttons = QHBoxLayout()
        import_fen = QPushButton("FEN importieren")
        import_fen.clicked.connect(self._import_fen)
        export_fen = QPushButton("FEN exportieren")
        export_fen.clicked.connect(self._export_fen)
        fen_buttons.addWidget(import_fen)
        fen_buttons.addWidget(export_fen)
        self.turn_combo = QComboBox()
        self.turn_combo.addItems(["Weiß am Zug", "Schwarz am Zug"])
        self.castling_edit = QLineEdit("-")
        self.castling_edit.setMaxLength(4)
        self.castling_edit.setToolTip("Rochaderechte: Kombination aus KQkq oder -")
        self.ep_edit = QLineEdit("-")
        self.ep_edit.setToolTip("En-passant-Zielfeld, z. B. e6, oder -")
        apply_state = QPushButton("Zugrecht / Rochade / En passant übernehmen")
        apply_state.clicked.connect(self._apply_board_state)
        self.piece_combo = QComboBox()
        for label, symbol in PIECE_CHOICES:
            self.piece_combo.addItem(label, symbol)
        apply_fen = QPushButton("FEN übernehmen")
        apply_fen.clicked.connect(self._apply_fen)
        start_pos = QPushButton("Grundstellung")
        start_pos.clicked.connect(self._reset_start)
        clear_board = QPushButton("Brett leeren")
        clear_board.clicked.connect(self._clear_board)
        flip_board = QPushButton("Brett drehen")
        flip_board.clicked.connect(self.board_widget.flip)
        self.btn_analyze_training = QPushButton("Trainingsstellung analysieren")
        self.btn_analyze_training.clicked.connect(self._analyze_training)
        self.selected_square_label = QLabel("Kein Feld ausgewählt.")
        self.validity_label = QLabel("–")
        self.validity_label.setWordWrap(True)
        self.detection_details = QLabel("Noch keine Bilderkennung.")
        self.detection_details.setWordWrap(True)
        self.detection_table = QTableWidget(0, 4)
        self.detection_table.setHorizontalHeaderLabels(["Feld", "Figur", "Score", "Quelle"])
        self.detection_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.detection_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.detection_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.detection_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.detection_table.setMaximumHeight(220)
        self.detection_table.cellClicked.connect(self._detection_row_clicked)

        side.addWidget(QLabel("FEN (vollständig, inklusive Zugrecht und Rochade)"))
        side.addWidget(self.fen_edit)
        side.addLayout(fen_buttons)
        side.addWidget(apply_fen)
        side.addWidget(self.turn_combo)
        state_row = QHBoxLayout()
        state_row.addWidget(QLabel("Rochade"))
        state_row.addWidget(self.castling_edit)
        state_row.addWidget(QLabel("EP"))
        state_row.addWidget(self.ep_edit)
        side.addLayout(state_row)
        side.addWidget(apply_state)
        side.addWidget(self.selected_square_label)
        side.addWidget(QLabel("Manuelle Korrektur"))
        side.addWidget(self.piece_combo)
        apply_piece = QPushButton("Auswahl auf Feld anwenden")
        apply_piece.clicked.connect(self._apply_piece_correction)
        side.addWidget(apply_piece)
        board_buttons = QHBoxLayout()
        board_buttons.addWidget(start_pos)
        board_buttons.addWidget(clear_board)
        board_buttons.addWidget(flip_board)
        side.addLayout(board_buttons)
        side.addWidget(self.btn_analyze_training)
        side.addWidget(self.validity_label)
        side.addWidget(QLabel("Erkennungs-Konfidenzen (Score, keine kalibrierte Wahrscheinlichkeit)"))
        side.addWidget(self.detection_table)
        side.addWidget(self.detection_details)
        side.addStretch()
        wrap = QWidget()
        wrap.setLayout(side)
        layout.addWidget(self.board_widget, 3)
        layout.addWidget(wrap, 2)
        return widget

    def _analysis_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        engine_row = QHBoxLayout()
        self.analysis_engine_label = QLabel("Engine: nicht gestartet.")
        self.analysis_engine_label.setWordWrap(True)
        start_engine = QPushButton("Engine starten")
        start_engine.clicked.connect(self._start_engine)
        open_settings = QPushButton("Engine-Einstellungen")
        open_settings.clicked.connect(lambda: self.tabs.setCurrentWidget(self.engine_tab))
        engine_row.addWidget(self.analysis_engine_label, 1)
        engine_row.addWidget(start_engine)
        engine_row.addWidget(open_settings)
        layout.addLayout(engine_row)
        self.analysis_summary = QLabel("Noch keine Analyse. Bewertungen werden aus Weiß-Sicht angezeigt.")
        self.analysis_summary.setWordWrap(True)
        self.analysis_tree = QTreeWidget()
        self.analysis_tree.setColumnCount(5)
        self.analysis_tree.setHeaderLabels(["PV", "Bewertung (Weiß)", "Tiefe", "Bester Zug", "Variante (SAN)"])
        header = self.analysis_tree.header()
        for column in (0, 1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.analysis_tree.setAlternatingRowColors(True)
        export_analysis = QPushButton("Aktuelles Analyseergebnis als Text exportieren")
        export_analysis.clicked.connect(self._export_analysis)
        layout.addWidget(self.analysis_summary)
        layout.addWidget(self.analysis_tree)
        layout.addWidget(export_analysis)
        layout.addWidget(QLabel("PV1–PV5 sind unabhängige beste Zugfolgen; Pfeile verwenden dieselbe Farbreihenfolge."))
        return widget

    def _pgn_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        row = QHBoxLayout()
        load_btn = QPushButton("PGN laden")
        load_btn.clicked.connect(self._load_pgn)
        export_btn = QPushButton("PGN exportieren")
        export_btn.clicked.connect(self._export_pgn)
        self.btn_analyze_pgn = QPushButton("Aktuelle PGN-Stellung analysieren")
        self.btn_analyze_pgn.clicked.connect(self._analyze_pgn)
        row.addWidget(load_btn)
        row.addWidget(export_btn)
        row.addWidget(self.btn_analyze_pgn)
        self.pgn_game_combo = QComboBox()
        self.pgn_game_combo.currentIndexChanged.connect(self._pgn_game_changed)
        self.pgn_list = QListWidget()
        self.pgn_list.currentRowChanged.connect(self._pgn_row_changed)
        self.pgn_info = QLabel("Keine Partie geladen. Nur abgeschlossene Partien können im PGN-Modus analysiert werden.")
        self.pgn_info.setWordWrap(True)
        layout.addLayout(row)
        layout.addWidget(self.pgn_game_combo)
        layout.addWidget(self.pgn_list)
        layout.addWidget(self.pgn_info)
        return widget

    def _history_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.history_view = QTextEdit()
        self.history_view.setReadOnly(True)
        layout.addWidget(self.history_view)
        return widget

    @staticmethod
    def _with_button(editor: QWidget, button: QPushButton) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(editor)
        row.addWidget(button)
        wrap = QWidget()
        wrap.setLayout(row)
        return wrap

    # ---------------------------------------------------------- Einstellungen
    def _load_settings_into_ui(self) -> None:
        self.engine_path.setText(self.config.engine.executable)
        self.depth_spin.setValue(self.config.engine.depth)
        self.time_spin.setValue(self.config.engine.move_time_ms)
        self.threads_spin.setValue(self.config.engine.threads)
        self.hash_spin.setValue(self.config.engine.hash_mb)
        self.multipv_spin.setValue(self.config.engine.multipv)
        self.opacity_slider.setValue(round(self.config.overlay.opacity * 100))
        self.scale_slider.setValue(self.config.overlay.scale_percent)
        self.hotkey_edit.setText(self.config.overlay.hotkey)
        self.orientation_combo.setCurrentText(self.config.capture.orientation)
        self.warp_size_spin.setValue(self.config.capture.warp_size)
        self.model_path_edit.setText(self.config.model_path)
        self.occupancy_spin.setValue(self.config.occupancy_std_threshold)
        self.confidence_spin.setValue(self.config.min_piece_confidence)

        self.frame_check.blockSignals(True)
        self.frame_width_spin.blockSignals(True)
        self.coordinate_scale_spin.blockSignals(True)
        self.frame_check.setChecked(self.config.overlay.show_frame)
        self.frame_width_spin.setValue(self.config.overlay.frame_width)
        self.coordinate_scale_spin.setValue(self.config.overlay.coordinate_scale_percent)
        self.frame_check.blockSignals(False)
        self.frame_width_spin.blockSignals(False)
        self.coordinate_scale_spin.blockSignals(False)
        self._frame_color = self.config.overlay.frame_color
        self.frame_color_button.setText(self._frame_color)
        self.frame_color_button.setStyleSheet(
            f"background:{self._frame_color}; color:#111; font-weight:700;"
        )

        self.auto_frame_check.blockSignals(True)
        self.auto_frame_check.setChecked(self.config.capture.auto_frame)
        self.auto_frame_check.blockSignals(False)

        for widget in (self.window_width_spin, self.window_height_spin):
            widget.blockSignals(True)
        self.window_width_spin.setValue(self.config.ui.window_width)
        self.window_height_spin.setValue(self.config.ui.window_height)
        for widget in (self.window_width_spin, self.window_height_spin):
            widget.blockSignals(False)
        self.maximized_check.blockSignals(True)
        self.maximized_check.setChecked(self.config.ui.start_maximized)
        self.maximized_check.blockSignals(False)
        if hasattr(self, "maximized_action"):
            self.maximized_action.blockSignals(True)
            self.maximized_action.setChecked(self.config.ui.start_maximized)
            self.maximized_action.blockSignals(False)
        self.popup_width_spin.blockSignals(True)
        self.popup_width_spin.setValue(self.config.ui.helper_width)
        self.popup_width_spin.blockSignals(False)
        self.font_scale_slider.blockSignals(True)
        self.font_scale_slider.setValue(self.config.ui.font_scale_percent)
        self.font_scale_slider.blockSignals(False)
        self.font_scale_value.setText(f"{self.config.ui.font_scale_percent} %")

        if self.config.capture.region:
            self.use_region.setChecked(True)
            self.region_left.setValue(self.config.capture.region["left"])
            self.region_top.setValue(self.config.capture.region["top"])
            self.region_width.setValue(self.config.capture.region["width"])
            self.region_height.setValue(self.config.capture.region["height"])
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        self.scale_slider.valueChanged.connect(self._scale_changed)
        self.font_scale_slider.valueChanged.connect(self._font_slider_changed)

    def _sync_engine_settings(self) -> None:
        self.config.engine.executable = self.engine_path.text().strip()
        self.config.engine.depth = self.depth_spin.value()
        self.config.engine.move_time_ms = self.time_spin.value()
        self.config.engine.threads = self.threads_spin.value()
        self.config.engine.hash_mb = self.hash_spin.value()
        self.config.engine.multipv = self.multipv_spin.value()
        self.config.overlay.opacity = self.opacity_slider.value() / 100.0
        self.config.overlay.scale_percent = self.scale_slider.value()
        self.config.overlay.hotkey = self.hotkey_edit.text().strip() or self.config.overlay.hotkey
        self.config.overlay.show_frame = self.frame_check.isChecked()
        self.config.overlay.frame_width = self.frame_width_spin.value()
        self.config.overlay.frame_color = getattr(self, "_frame_color", self.config.overlay.frame_color)
        self.config.overlay.coordinate_scale_percent = self.coordinate_scale_spin.value()
        self.config.capture.orientation = self.orientation_combo.currentText()
        self.config.capture.warp_size = self.warp_size_spin.value()
        self.config.capture.auto_frame = self.auto_frame_check.isChecked()
        self.config.ui.window_width = self.window_width_spin.value()
        self.config.ui.window_height = self.window_height_spin.value()
        self.config.ui.font_scale_percent = self.font_scale_slider.value()
        self.config.ui.helper_width = self.helper.panel_width()
        self.config.ui.start_maximized = self.maximized_check.isChecked()
        monitor_data = self.monitor_combo.currentData()
        self.config.capture.monitor_index = int(monitor_data) if monitor_data is not None else 1
        if self.use_region.isChecked():
            self.config.capture.region = {
                "left": self.region_left.value(),
                "top": self.region_top.value(),
                "width": self.region_width.value(),
                "height": self.region_height.value(),
            }
        else:
            self.config.capture.region = None
        self.config.capture.corners = deepcopy(self.corners) if len(self.corners) == 4 else None
        self.config.model_path = self.model_path_edit.text().strip()
        self.config.occupancy_std_threshold = self.occupancy_spin.value()
        self.config.min_piece_confidence = self.confidence_spin.value()
        self.config.side_to_move = "w" if self.turn_combo.currentIndex() == 0 else "b"

    # ------------------------------------------------------ Fenster & Schrift
    def _initial_geometry(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(0, 0, 1380, 900)
        width = min(self.config.ui.window_width, available.width())
        height = min(self.config.ui.window_height, available.height())
        self.resize(width, height)
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def _update_size_fields(self, width: int, height: int, maximized: bool) -> None:
        for widget in (self.window_width_spin, self.window_height_spin, self.maximized_check):
            widget.blockSignals(True)
        self.window_width_spin.setValue(max(720, width))
        self.window_height_spin.setValue(max(520, height))
        self.maximized_check.setChecked(maximized)
        for widget in (self.window_width_spin, self.window_height_spin, self.maximized_check):
            widget.blockSignals(False)
        if hasattr(self, "maximized_action"):
            self.maximized_action.blockSignals(True)
            self.maximized_action.setChecked(maximized)
            self.maximized_action.blockSignals(False)
        self.config.ui.window_width = self.window_width_spin.value()
        self.config.ui.window_height = self.window_height_spin.value()
        self.config.ui.start_maximized = maximized

    def _capture_window_size(self) -> None:
        if self.isMaximized():
            self._update_size_fields(
                self.config.ui.window_width, self.config.ui.window_height, True
            )
            self.statusBar().showMessage("Startgröße gespeichert: maximiert.")
        else:
            self._update_size_fields(self.width(), self.height(), False)
            self.statusBar().showMessage(
                f"Fenstergröße gespeichert: {self.width()} × {self.height()}."
            )
        self._refresh_overview()

    def _apply_size_preset(self, width: int, height: int) -> None:
        self._update_size_fields(width, height, False)
        if self.isMaximized():
            self.showNormal()
        screen = self.screen() or QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(0, 0, width, height)
        self.resize(min(width, available.width()), min(height, available.height()))
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())
        self._refresh_overview()
        self.statusBar().showMessage(f"Fenstergröße: {width} × {height}.")

    def _apply_size_from_fields(self) -> None:
        self._apply_size_preset(self.window_width_spin.value(), self.window_height_spin.value())
        self._save_config()

    def _set_start_maximized(self, checked: bool) -> None:
        self.config.ui.start_maximized = bool(checked)
        for widget in (self.maximized_check, getattr(self, "maximized_action", None)):
            if widget is None:
                continue
            widget.blockSignals(True)
            widget.setChecked(bool(checked))
            widget.blockSignals(False)
        self._refresh_overview()

    def _font_slider_changed(self, value: int) -> None:
        self._apply_font_scale(value, from_slider=True)

    def _adjust_font_scale(self, delta: int) -> None:
        self._apply_font_scale(self.config.ui.font_scale_percent + delta)

    def _apply_popup_width(self, value: int) -> None:
        self.popup_width_spin.setValue(self.helper.apply_width(value))

    def _apply_font_scale(self, percent: int, announce: bool = True, from_slider: bool = False) -> None:
        percent = clamp_font_scale(percent)
        self.config.ui.font_scale_percent = percent
        application = QApplication.instance()
        if application is not None:
            application.setStyleSheet(build_stylesheet(percent))
        self.font_scale_slider.blockSignals(True)
        self.font_scale_slider.setValue(percent)
        self.font_scale_slider.blockSignals(False)
        self.font_scale_value.setText(f"{percent} %")
        if announce:
            self.statusBar().showMessage(f"Schriftgröße: {percent} %.")
        self._refresh_overview()

    def _show_version_info(self) -> None:
        history = "".join(
            f"<li><b>{version}</b> – {description}</li>" for version, description in VERSION_HISTORY
        )
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Version / Info – {version_label()}")
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)
        heading = QLabel(f"<h2>{full_version_label()}</h2>")
        layout.addWidget(heading)
        text = QLabel(
            "Versionsschema: Jede neue veröffentlichte Fassung erhält die nächste Nummer "
            "(v1, v2, v3 …). Die Nummer steht im Fenstertitel, im Start-Tab und hier."
            f"<ul>{history}</ul>"
            "<p>Stockfish ist nicht enthalten und wird separat unter "
            "<a href='https://stockfishchess.org/download/'>stockfishchess.org/download</a> bezogen.</p>"
        )
        text.setWordWrap(True)
        text.setOpenExternalLinks(True)
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _apply_recognition_settings(self) -> None:
        self._sync_engine_settings()
        self.board_widget.confidence_threshold = self.config.min_piece_confidence
        self.board_widget.update()
        self.recognition_worker.request_configure.emit(
            self.config.occupancy_std_threshold,
            self.config.min_piece_confidence,
            self.config.model_path,
        )
        self.statusBar().showMessage("Erkennungseinstellungen angewendet; Session-Templates bleiben bei gleichem Modell erhalten.")

    # --------------------------------------------------------------- Capture
    def _refresh_monitors(self) -> None:
        self.monitor_combo.clear()
        try:
            monitors = list_monitors()
        except Exception as exc:
            logger.warning("Monitorliste nicht verfügbar: %s", exc)
            self.monitor_combo.addItem(f"Kein Bildschirmzugriff: {exc}", 1)
            return
        for info in monitors:
            self.monitor_combo.addItem(info.label, info.index)
        index = self.monitor_combo.findData(self.config.capture.monitor_index)
        self.monitor_combo.setCurrentIndex(index if index >= 0 else 0)

    def _toggle_preview(self) -> None:
        self.live_preview = not self.live_preview
        self.btn_preview.setText("Vorschau stoppen" if self.live_preview else "Vorschau starten")
        if self.live_preview:
            self.preview_timer.start()
            self._tick_preview()
        else:
            self.preview_timer.stop()
            self.capture_status.setText("Vorschau gestoppt. Ein vorhandener Snapshot bleibt eingefroren.")

    def _tick_preview(self) -> None:
        if self.live_preview and not self.capture_in_flight:
            self._request_capture("preview")

    def _take_snapshot(self) -> None:
        if self.live_preview:
            self._toggle_preview()
        self._request_capture("snapshot")

    def _request_capture(self, purpose: str) -> None:
        if self.capture_in_flight:
            self.capture_status.setText("Eine Bildschirmaufnahme läuft bereits.")
            return
        self._sync_engine_settings()
        self.capture_in_flight = True
        self.capture_status.setText("Bildschirmaufnahme läuft im Hintergrund …")
        self._update_action_states()
        self.capture_worker.request_grab.emit(
            self.config.capture.monitor_index,
            deepcopy(self.config.capture.region),
            purpose,
        )

    def _on_capture(self, frame: np.ndarray, origin: tuple[float, float], purpose: str) -> None:
        self.capture_in_flight = False
        self.capture_origin = (float(origin[0]), float(origin[1]))
        self.display_frame = frame
        if purpose == "snapshot":
            self.last_frame = frame.copy()
            self.last_warped = None
            self.snapshot_recognized = False
            self.snapshot_confirmed = self.confirm_box.isChecked()
            self._clear_detection_state()
            self.capture_status.setText(
                "Snapshot eingefroren. Jetzt vier Ecken prüfen, Orientierung wählen und erkennen."
            )
            if self.config.capture.auto_frame:
                detected = self._detect_frame_corners(silent=True)
                if detected:
                    self.capture_status.setText(
                        "Snapshot eingefroren. Brett automatisch rot eingerahmt – "
                        "Rahmen prüfen und danach erkennen."
                    )
                else:
                    self.capture_status.setText(
                        "Snapshot eingefroren. Brett nicht sicher gefunden: vier Ecken klicken "
                        "oder ein Rechteck aufziehen."
                    )
        elif self.live_preview and len(self.corners) == 4:
            try:
                warped, _ = warp_board(frame, self.corners, self.config.capture.warp_size)
                delta = board_mean_abs_diff(self.preview_warped, warped)
                self.preview_warped = warped
                self.capture_status.setText(
                    f"Vorschau aktiv. Änderung: {delta:.1f}. Keine automatische Erkennung oder Engine-Analyse."
                )
            except ValueError as exc:
                self.capture_status.setText(f"Kalibrierung prüfen: {exc}")
        else:
            self.capture_status.setText("Vorschau aktiv; keine automatische Engine-Aktualisierung.")
        if purpose == "snapshot" and self._chain_recognize:
            self._chain_recognize = False
            if len(self.corners) == 4:
                self.helper.set_step(1, "Brett erkannt. Figuren werden im Hintergrund erkannt …", busy=True)
                self._recognize_snapshot()
            else:
                self.helper.set_step(
                    0,
                    "Brett nicht automatisch gefunden. Bitte den Ausschnitt enger um das "
                    "Spielfeld ziehen oder in der App vier Ecken klicken.",
                    can_analyze=False,
                )
        self.calib_view.set_frame(self._annotate_corners(self.display_frame))
        self._update_overlay_frame()
        self._update_action_states()

    def _on_capture_error(self, message: str, purpose: str) -> None:
        self.capture_in_flight = False
        self._chain_recognize = False
        if purpose == "preview" and self.live_preview:
            self._toggle_preview()
        self.capture_status.setText(f"Aufnahmefehler: {message}")
        self.helper.set_step(0, f"Aufnahmefehler: {message.splitlines()[0]}")
        self._update_action_states()

    def _annotate_corners(self, frame: np.ndarray) -> np.ndarray:
        visual = frame.copy()
        red = (45, 45, 255)
        if len(self.corners) == 4:
            points = np.array(self.corners, dtype=np.int32).reshape(-1, 1, 2)
            overlay = visual.copy()
            cv2.fillPoly(overlay, [points], (40, 40, 220))
            visual = cv2.addWeighted(overlay, 0.10, visual, 0.90, 0)
            cv2.polylines(visual, [points], True, red, 4)
        for index, (x, y) in enumerate(self.corners):
            cv2.circle(visual, (round(x), round(y)), 9, red, -1)
            cv2.circle(visual, (round(x), round(y)), 4, (255, 255, 255), -1)
            cv2.putText(
                visual,
                str(index + 1),
                (round(x) + 12, round(y) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                red,
                2,
            )
        return visual

    def _set_corners(self, corners: list[list[float]], source: str = "") -> None:
        try:
            validate_corners(corners)
        except ValueError as exc:
            self.capture_status.setText(f"Rahmen ungültig: {exc}")
            return
        self.corners = [[float(point[0]), float(point[1])] for point in corners]
        self.config.capture.corners = deepcopy(self.corners)
        if self.display_frame is not None:
            self.calib_view.set_frame(self._annotate_corners(self.display_frame))
        self._update_overlay_frame()
        if source:
            self.capture_status.setText(source)
        self._update_action_states()

    def _detect_frame_corners(self, silent: bool = False) -> bool:
        frame = self.last_frame if self.last_frame is not None else self.display_frame
        if frame is None:
            if not silent:
                self.capture_status.setText(
                    "Bitte zuerst einen Snapshot aufnehmen, um das Brett automatisch einzurahmen."
                )
            return False
        corners = detect_board_corners(frame)
        if corners is None:
            if not silent:
                self.capture_status.setText(
                    "Kein Brett gefunden. Bitte vier Ecken klicken oder ein Rechteck aufziehen."
                )
            return False
        self._set_corners(
            corners,
            "Brett automatisch rot eingerahmt. Sitzt der Rahmen nicht, Ecken neu setzen."
            if not silent
            else "",
        )
        return True

    def _auto_frame_and_show(self) -> None:
        self.tabs.setCurrentWidget(self.capture_tab)
        self._detect_frame_corners()

    def _auto_frame_toggled(self, checked: bool) -> None:
        self.config.capture.auto_frame = bool(checked)
        self.statusBar().showMessage(
            "Automatischer roter Brettrahmen aktiviert." if checked else "Automatischer Rahmen aus."
        )
        self._refresh_overview()

    def _set_rectangle_corners(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self._set_corners(
            rectangle_to_corners(x0, y0, x1, y1),
            "Rechteck als Brettrahmen übernommen. Bei Bedarf Ecken einzeln nachjustieren.",
        )

    def _add_corner(self, x: float, y: float) -> None:
        if len(self.corners) >= 4:
            self.corners = []
        self.corners.append([float(x), float(y)])
        if self.display_frame is not None:
            self.calib_view.set_frame(self._annotate_corners(self.display_frame))
        self.capture_status.setText(f"{len(self.corners)} / 4 Ecken gesetzt.")
        self._sync_engine_settings()
        self._update_overlay_frame()
        self._update_action_states()

    def _reset_corners(self) -> None:
        self.corners = []
        self.preview_warped = None
        self.snapshot_recognized = False
        self.config.capture.corners = None
        if self.display_frame is not None:
            self.calib_view.set_frame(self._annotate_corners(self.display_frame))
        self.overlay.update_frame(None, self.capture_origin)
        self.capture_status.setText(
            "Rahmen gelöscht; Snapshot muss nach neuer Kalibrierung erneut erkannt werden."
        )
        self._update_action_states()

    # ------------------------------------------------------------ Recognition
    def _recognize_snapshot(self) -> None:
        if self.last_frame is None:
            QMessageBox.information(self, "Erkennung", "Bitte zuerst einen Snapshot aufnehmen.")
            return
        if len(self.corners) != 4:
            QMessageBox.information(self, "Erkennung", "Bitte vier Brettecken kalibrieren.")
            return
        if self.recognition_in_flight:
            self.capture_status.setText("Die Figurenerkennung läuft bereits.")
            return
        try:
            self.pending_warped, _ = warp_board(
                self.last_frame, self.corners, self.config.capture.warp_size
            )
        except (ValueError, cv2.error) as exc:
            QMessageBox.warning(self, "Kalibrierung", str(exc))
            return
        self.recognition_in_flight = True
        self.snapshot_recognized = False
        self.capture_status.setText("64 Felder werden im Hintergrund klassifiziert …")
        self._update_action_states()
        turn = chess.WHITE if self.turn_combo.currentIndex() == 0 else chess.BLACK
        self.recognition_worker.request_recognize.emit(
            self.pending_warped.copy(), self.orientation_combo.currentText(), turn
        )

    def _on_recognition(self, result: RecognitionResult) -> None:
        self.recognition_in_flight = False
        self.last_warped = self.pending_warped
        self.pending_warped = None
        self.detections = result.detections
        self.snapshot_recognized = True
        self._set_mode("frozen_snapshot")
        self._set_board(result.board, clear_detections=False)
        self.board_widget.set_detections(self.detections)
        self._populate_detection_table()
        ok, issues = result.validity
        low = [
            item
            for item in self.detections
            if item.occupied and item.confidence < self.config.min_piece_confidence
        ]
        messages = list(issues)
        if low:
            names = ", ".join(chess.square_name(item.square) for item in low[:12])
            messages.append(f"Unsichere belegte Felder: {names}. Bitte prüfen/korrigieren.")
        if result.model_error:
            messages.append("Modell: " + result.model_error)
        self._set_validity_messages(ok and not low, messages)
        self.guard.record(result.board.board_fen())
        self.capture_status.setText(
            "Erkennung abgeschlossen. Konfidenzen prüfen; keine Auto-Analyse."
        )
        unsure = len(low)
        summary = f"{len(self.detections)} Felder geprüft"
        if unsure:
            summary += f", {unsure} unsicher"
        self.helper.set_step(
            1,
            f"Erkennung fertig ({summary}). Jetzt analysieren oder in der App korrigieren.",
            can_analyze=True,
        )
        self._update_action_states()

    def _on_recognition_error(self, message: str) -> None:
        self.recognition_in_flight = False
        self.snapshot_recognized = False
        self.pending_warped = None
        self.capture_status.setText("Erkennungsfehler: " + message.split("\n", 1)[0])
        self.helper.set_step(1, "Erkennungsfehler: " + message.split("\n", 1)[0], can_analyze=False)
        self._update_action_states()

    def _on_confirm_changed(self) -> None:
        self.snapshot_confirmed = self.confirm_box.isChecked()

    # -------------------------------------------------------------- Position
    def _set_board(self, board: chess.Board, clear_detections: bool = True) -> None:
        self.board = board.copy(stack=False)
        self.board_widget.set_board(self.board)
        if clear_detections:
            self._clear_detection_state()
        self.fen_edit.setPlainText(self.board.fen())
        self.turn_combo.blockSignals(True)
        self.turn_combo.setCurrentIndex(0 if self.board.turn == chess.WHITE else 1)
        self.turn_combo.blockSignals(False)
        self.castling_edit.setText(self.board.castling_xfen())
        self.ep_edit.setText(chess.square_name(self.board.ep_square) if self.board.ep_square is not None else "-")
        self._refresh_validity()

    def _clear_detection_state(self) -> None:
        self.detections = []
        if hasattr(self, "detection_table"):
            self.detection_table.setRowCount(0)
            self.detection_details.setText("Noch keine Bilderkennung.")
            self.board_widget.clear_detections()

    def _refresh_validity(self) -> None:
        ok, issues = describe_validity(self.board)
        self._set_validity_messages(ok, issues)

    def _set_validity_messages(self, ok: bool, messages: list[str]) -> None:
        self.validity_label.setText("Stellung gültig." if ok else "\n".join(messages) or "Stellung prüfen.")
        self.validity_label.setStyleSheet("color:#79d99a;" if ok else "color:#f0b35e;")

    def _apply_fen(self) -> None:
        try:
            board = parse_fen(self.fen_edit.toPlainText().strip())
        except ValueError as exc:
            QMessageBox.warning(self, "FEN", str(exc))
            return
        self.snapshot_recognized = False
        self._set_mode("training")
        self._set_board(board)
        self._update_action_states()

    def _import_fen(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "FEN importieren", "", "FEN (*.fen *.txt);;Alle Dateien (*)")
        if not path:
            return
        try:
            board = load_fen(Path(path))
        except ValueError as exc:
            QMessageBox.warning(self, "FEN-Import", str(exc))
            return
        self.snapshot_recognized = False
        self._set_mode("training")
        self._set_board(board)
        self.tabs.setCurrentWidget(self.training_tab)
        self.statusBar().showMessage(f"FEN importiert: {path}")

    def _export_fen(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "FEN exportieren", "stellung.fen", "FEN (*.fen);;Text (*.txt)")
        if not path:
            return
        try:
            save_fen(Path(path), self.board)
        except ValueError as exc:
            QMessageBox.warning(self, "FEN-Export", str(exc))
            return
        self.statusBar().showMessage(f"FEN exportiert: {path}")

    def _reset_start(self) -> None:
        self.snapshot_recognized = False
        self._set_mode("training")
        self._set_board(chess.Board())
        self._update_action_states()

    def _clear_board(self) -> None:
        self.snapshot_recognized = False
        self._set_mode("training")
        self._set_board(chess.Board(None))
        self._update_action_states()

    def _apply_board_state(self) -> None:
        try:
            board = update_board_state(
                self.board,
                turn=chess.WHITE if self.turn_combo.currentIndex() == 0 else chess.BLACK,
                castling_fen=self.castling_edit.text(),
                ep_fen=self.ep_edit.text(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Stellungsdaten", str(exc))
            return
        self._set_board(board, clear_detections=False)
        if self.mode == "frozen_snapshot":
            self.guard.record(board.board_fen())
        self.statusBar().showMessage("Zugrecht, Rochade und En-passant-Feld aktualisiert.")

    def _on_square_clicked(self, square: int) -> None:
        name = chess.square_name(square)
        piece = self.board.piece_at(square)
        self.selected_square_label.setText(f"Ausgewählt: {name} ({piece.symbol() if piece else 'leer'})")
        detection = next((item for item in self.detections if item.square == square), None)
        if detection is not None:
            alternatives = ", ".join(
                f"{symbol} {score:.0%}" for symbol, score in detection.alternatives.items()
            )
            self.detection_details.setText(
                f"{name}: Belegung={detection.occupied}, Score={detection.confidence:.1%}, "
                f"Quelle={detection.source}. Alternativen: {alternatives or 'keine'}."
            )
        self.statusBar().showMessage(f"Feld {name} ausgewählt")

    def _apply_piece_correction(self) -> None:
        square = self.board_widget.selected
        if square is None:
            QMessageBox.information(self, "Korrektur", "Bitte ein Feld auf dem Brett anklicken.")
            return
        symbol = self.piece_combo.currentData()
        self.board = apply_manual_piece(self.board, square, symbol)
        if self.mode == "frozen_snapshot" and self.last_warped is not None:
            col, row = square_to_image_rc(square, self.orientation_combo.currentText())
            tile = extract_square(self.last_warped, col, row).copy()
            self.recognition_worker.request_correction.emit(tile, symbol)
            piece = self.board.piece_at(square)
            for index, detection in enumerate(self.detections):
                if detection.square == square:
                    self.detections[index] = Detection(
                        square=square,
                        piece=piece,
                        confidence=1.0,
                        occupied=piece is not None,
                        source="manual",
                    )
                    break
            self.guard.record(self.board.board_fen())
        self._set_board(self.board, clear_detections=False)
        self.board_widget.set_detections(self.detections)
        self._populate_detection_table()
        self._on_square_clicked(square)
        self._update_action_states()

    def _populate_detection_table(self) -> None:
        self.detection_table.setRowCount(len(self.detections))
        for row, detection in enumerate(self.detections):
            confidence_color = QColor(230, 235, 240)
            if detection.occupied and detection.confidence < self.config.min_piece_confidence:
                confidence_color = QColor(239, 100, 90)
            foreground = QBrush(confidence_color)
            values = [
                chess.square_name(detection.square),
                detection.piece.symbol() if detection.piece else ("?" if detection.occupied else "·"),
                f"{detection.confidence:.1%}",
                detection.source,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, int(detection.square))
                item.setForeground(foreground)
                item.setToolTip(
                    ", ".join(f"{key}: {score:.1%}" for key, score in detection.alternatives.items())
                    or "Keine SVM-Alternativen"
                )
                self.detection_table.setItem(row, column, item)

    def _detection_row_clicked(self, row: int, column: int) -> None:
        item = self.detection_table.item(row, 0)
        if item is None:
            return
        square = int(item.data(Qt.ItemDataRole.UserRole))
        self.board_widget.selected = square
        self.board_widget.update()
        self._on_square_clicked(square)

    # ---------------------------------------------------------------- Engine
    def _analyze_snapshot(self) -> None:
        if not self.snapshot_recognized or self.mode != "frozen_snapshot" or self.last_warped is None:
            QMessageBox.information(
                self, "Snapshot", "Bitte zuerst den aktuellen Snapshot erkennen und die Stellung prüfen."
            )
            return
        self._run_analysis(self.board)

    def _analyze_training(self) -> None:
        self._set_mode("training")
        self._run_analysis(self.board)

    def _analyze_pgn(self) -> None:
        game = self._current_pgn_game()
        if game is None:
            QMessageBox.information(self, "PGN", "Bitte zuerst eine PGN-Datei laden.")
            return
        if not game_is_completed(game):
            QMessageBox.warning(
                self,
                "PGN noch nicht abgeschlossen",
                "Diese Partie hat Ergebnis '*'. Der PGN-Analysemodus ist nur für abgeschlossene Partien freigegeben.",
            )
            return
        self._set_mode("pgn_completed")
        self._run_analysis(self.board)

    def _run_analysis(self, board: chess.Board) -> None:
        allowed, reason = can_analyze(
            self.mode,
            snapshot_confirmed=self.snapshot_confirmed,
            live_guard=self.guard if self.mode == "frozen_snapshot" else None,
        )
        if not allowed:
            QMessageBox.warning(self, "Analyse gesperrt", reason)
            return
        valid, issues = describe_validity(board)
        if not valid:
            QMessageBox.warning(self, "Ungültige Stellung", "\n".join(issues))
            return
        if self.analysis_in_flight:
            self.statusBar().showMessage("Eine Engine-Analyse läuft bereits.")
            return
        self._sync_engine_settings()
        if not self.engine_ready:
            QMessageBox.information(self, "Engine", "Bitte zuerst die Engine starten.")
            return
        self.analysis_in_flight = True
        self.statusBar().showMessage("Engine analysiert im Hintergrund …")
        self._update_action_states()
        self.worker.request_analyse.emit(board.copy(stack=False), deepcopy(self.config.engine))

    def _on_analysis(self, result) -> None:
        self.analysis_in_flight = False
        self.last_result = result
        self.analysis_tree.clear()
        for line in result.lines:
            values = [
                str(line.multipv),
                line.score_text,
                str(line.depth) if line.depth is not None else "–",
                line.best_move_san,
                line.pv_san,
            ]
            item = QTreeWidgetItem(values)
            item.setToolTip(4, "\n".join(line.pv_uci))
            self.analysis_tree.addTopLevelItem(item)
        self.analysis_summary.setText(
            f"{result.engine_id} · {result.check} · Bewertung aus Weiß-Sicht · Modus: {self.mode}"
        )
        history_lines = [
            f"[{datetime.now():%H:%M:%S}] {result.engine_id} | {result.check} | {result.fen}",
            *[f"PV{line.multipv} {line.score_text}: {line.pv_san}" for line in result.lines],
        ]
        if result.hanging:
            history_lines.append("Hängende Figuren: " + ", ".join(result.hanging))
        self.history_view.append("\n".join(history_lines) + "\n")
        if result.lines:
            self.helper.set_step(
                2,
                f"Analyse fertig: {result.lines[0].score_text} · {result.lines[0].best_move_san}",
                can_analyze=True,
            )
        else:
            self.helper.set_step(2, "Analyse fertig, aber keine Zugfolge erhalten.", can_analyze=True)
        current = result.fen == self.board.fen()
        if current:
            self.board_widget.set_analysis(result.lines)
            calibrated = self.corners if len(self.corners) == 4 else None
            self.overlay.apply_settings(self.config.overlay)
            if self.overlay.user_hidden:
                self.overlay.user_hidden = False
                self.config.overlay.visible = True
                self.overlay.present()
            self.overlay.set_analysis(
                result,
                calibrated,
                self.orientation_combo.currentText(),
                self.config.capture.warp_size,
                origin=self.capture_origin,
            )
            self.statusBar().showMessage(
                f"Analyse fertig: {result.lines[0].score_text}" if result.lines else "Analyse fertig; keine Zugfolge."
            )
        else:
            self.statusBar().showMessage("Analyse einer früheren Stellung fertig; Overlay wegen Stellungswechsel nicht aktualisiert.")
        self._update_action_states()

    def _on_engine_error(self, message: str) -> None:
        self.analysis_in_flight = False
        self.engine_start_in_flight = False
        self.engine_ready = False
        self._update_engine_hint()
        logger.error(message)
        self.engine_status.setText("Engine-Fehler: " + message.split("\n", 1)[0])
        self._pending_analysis = False
        self.helper.set_step(2, "Engine-Fehler: " + message.split("\n", 1)[0])
        QMessageBox.warning(self, "Engine-Fehler", message.split("\n", 1)[0])
        self.statusBar().showMessage("Engine-Fehler; bitte Pfad/Einstellungen prüfen und Engine neu starten.")
        self._update_action_states()

    def _update_engine_hint(self) -> None:
        if not hasattr(self, "analysis_engine_label"):
            return
        if self.engine_ready:
            text = "Engine: bereit."
        elif self.engine_start_in_flight:
            text = "Engine: startet …"
        else:
            text = "Engine: nicht gestartet – bitte vor der Analyse starten."
        self.analysis_engine_label.setText(text)

    def _on_engine_started(self, name: str) -> None:
        self.engine_ready = True
        self.engine_start_in_flight = False
        self._update_engine_hint()
        self.engine_status.setText(f"Engine bereit: {name}")
        self.statusBar().showMessage(f"Engine bereit: {name}")
        self._update_action_states()
        if self._pending_analysis:
            self._pending_analysis = False
            if self.snapshot_recognized:
                self._helper_analyze()
            else:
                self.helper.set_step(
                    1, "Engine bereit. Bitte Snapshot + erkennen ausführen.", can_analyze=False
                )

    def _on_engine_stopped(self) -> None:
        self.engine_ready = False
        self.engine_start_in_flight = False
        self._update_engine_hint()
        self.engine_status.setText("Engine gestoppt.")
        self.helper.set_step(2, "Engine gestoppt. Für die Analyse erneut starten.", can_analyze=True)
        self.statusBar().showMessage("Engine gestoppt")
        self._update_action_states()

    def _start_engine(self) -> None:
        if self.engine_start_in_flight:
            return
        self._sync_engine_settings()
        self.engine_ready = False
        self.engine_start_in_flight = True
        self._update_engine_hint()
        self.engine_status.setText("Engine wird gestartet …")
        self._update_action_states()
        self.worker.request_start.emit(deepcopy(self.config.engine), None)

    def _stop_engine(self) -> None:
        self.engine_status.setText("Engine-Stopp angefordert …")
        self.worker.request_stop.emit()

    def _browse_engine(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Stockfish auswählen", "", "Programme (*.exe);;Alle Dateien (*)"
        )
        if path:
            self.engine_path.setText(path)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "OpenCV-SVM-Modell auswählen", "", "OpenCV XML (*.xml);;Alle Dateien (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    # ------------------------------------------------------- Einrichtungs-Popup
    def _place_helper(self) -> None:
        x, y = self._helper_offset
        self.helper.adjustSize()
        if x is None or y is None or x < 0 or y < 0:
            self.helper.place_default()
        else:
            self.helper.move(int(x), int(y))

    def _helper_moved(self, x: int, y: int) -> None:
        self._helper_offset = (int(x), int(y))
        self.config.ui.helper_x = int(x)
        self.config.ui.helper_y = int(y)

    def _helper_resized(self, width: int) -> None:
        value = self.helper.apply_width(width)
        self.config.ui.helper_width = value
        if hasattr(self, "popup_width_spin"):
            self.popup_width_spin.blockSignals(True)
            self.popup_width_spin.setValue(value)
            self.popup_width_spin.blockSignals(False)
        self.statusBar().showMessage(f"Popup-Breite: {value} px.")

    def _popup_width_changed(self, value: int) -> None:
        width = self.helper.apply_width(value)
        self.config.ui.helper_width = width
        if self.helper.isVisible():
            self.helper.adjustSize()
        self._refresh_overview()

    def _helper_hidden(self) -> None:
        self.config.ui.helper_visible = False
        if hasattr(self, "helper_action"):
            self.helper_action.blockSignals(True)
            self.helper_action.setChecked(False)
            self.helper_action.blockSignals(False)
        self.statusBar().showMessage(
            "Einrichtungs-Popup ausgeblendet. Start-Tab → Popup einblenden holt es zurück."
        )
        self._refresh_overview()

    def _show_helper(self) -> None:
        self.config.ui.helper_visible = True
        self._place_helper()
        self.helper.show_helper()
        ensure_topmost(self.helper)
        if hasattr(self, "helper_action"):
            self.helper_action.blockSignals(True)
            self.helper_action.setChecked(True)
            self.helper_action.blockSignals(False)
        self._refresh_overview()

    def _toggle_helper_from_menu(self, checked: bool) -> None:
        if checked and not self.helper.isVisible():
            self._show_helper()
        elif not checked and self.helper.isVisible():
            self.helper.hide()
            self.config.ui.helper_visible = False
            self._refresh_overview()

    def _toggle_helper(self) -> None:
        if self.helper.isVisible():
            self.helper.hide()
            self.config.ui.helper_visible = False
        else:
            self._show_helper()
        self._refresh_overview()

    def _helper_snapshot(self) -> None:
        self._chain_recognize = True
        self.helper.set_step(1, "Snapshot wird aufgenommen …", busy=True, can_analyze=False)
        self._take_snapshot()

    def _helper_open_settings(self) -> None:
        self.tabs.setCurrentWidget(self.engine_tab)
        if self.isMinimized():
            self.showNormal()
        self.raise_()
        self.activateWindow()

    def _helper_analyze(self) -> None:
        if not self.snapshot_recognized or self.last_warped is None:
            self.helper.set_step(
                1,
                "Bitte zuerst Snapshot + erkennen ausführen, danach analysieren.",
                can_analyze=False,
            )
            return
        if not self.engine_ready:
            self._pending_analysis = True
            self.helper.set_step(2, "Engine wird gestartet …", busy=True, can_analyze=False)
            self._start_engine()
            return
        self.helper.set_step(2, "Engine analysiert die Stellung …", busy=True, can_analyze=False)
        self._analyze_snapshot()

    # ------------------------------------------------------- Ausschnitt per Maus
    @staticmethod
    def _screen_rects() -> list[tuple[float, float, float, float, float]]:
        rects: list[tuple[float, float, float, float, float]] = []
        for screen in QGuiApplication.screens():
            geometry = screen.geometry()
            rects.append(
                (
                    float(geometry.x()),
                    float(geometry.y()),
                    float(geometry.width()),
                    float(geometry.height()),
                    float(screen.devicePixelRatio()) or 1.0,
                )
            )
        return rects

    def _select_region_with_mouse(self) -> None:
        if self.selector is None:
            self.selector = RegionSelector(None)
            self.selector.selected.connect(self._on_region_selected)
            self.selector.cancelled.connect(self._on_region_cancelled)
        self._helper_was_visible = self.helper.isVisible()
        if self._helper_was_visible:
            self.helper.hide()
        self.selector.start()
        self.statusBar().showMessage(
            "Ausschnitt wählen: Rahmen mit der Maus um das Spielfeld ziehen (Esc bricht ab)."
        )

    def _on_region_cancelled(self) -> None:
        if getattr(self, "_helper_was_visible", False):
            self._show_helper()
        self.helper.set_step(
            self.helper.step, "Auswahl abgebrochen. Es wurde nichts geändert."
        )
        self.statusBar().showMessage("Auswahl abgebrochen.")

    def _on_region_selected(self, rect: QRect) -> None:
        region = logical_rect_to_region(
            (rect.x(), rect.y(), rect.width(), rect.height()), self._screen_rects()
        )
        if getattr(self, "_helper_was_visible", False):
            self._show_helper()
        if region is None:
            self.helper.set_step(
                0,
                "Der Rahmen war zu klein. Bitte erneut einen Bereich um das Spielfeld ziehen.",
                can_analyze=False,
            )
            return
        self.config.capture.region = dict(region)
        self.use_region.blockSignals(True)
        self.use_region.setChecked(True)
        self.use_region.blockSignals(False)
        for widget, value in (
            (self.region_left, region["left"]),
            (self.region_top, region["top"]),
            (self.region_width, region["width"]),
            (self.region_height, region["height"]),
        ):
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)
        self.corners = []
        self.preview_warped = None
        self.snapshot_recognized = False
        self.config.capture.corners = None
        self.capture_origin = (float(region["left"]), float(region["top"]))
        self.overlay.update_frame(None, self.capture_origin)
        self._refresh_region_label()
        self.capture_status.setText(
            "Ausschnitt gespeichert. Snapshot wird aufgenommen und das Brett erkannt."
        )
        self._refresh_overview()
        self.helper.set_step(
            1,
            "Ausschnitt gespeichert. Snapshot wird aufgenommen und das Brett erkannt …",
            busy=True,
            can_analyze=False,
        )
        self._chain_recognize = True
        self._take_snapshot()

    def _refresh_region_label(self) -> None:
        if not hasattr(self, "region_label"):
            return
        region = self.config.capture.region
        if not region:
            self.region_label.setText(
                "Kein eigener Ausschnitt – es wird der gewählte Monitor komplett aufgenommen."
            )
            return
        self.region_label.setText(
            f"Ausschnitt: {region['width']} × {region['height']} px "
            f"ab {region['left']} / {region['top']} (physische Pixel)"
        )

    # ---------------------------------------------------------------- Overlay
    def _opacity_changed(self, value: int) -> None:
        self.config.overlay.opacity = value / 100.0
        self.overlay.apply_settings(self.config.overlay)

    def _scale_changed(self, value: int) -> None:
        self.config.overlay.scale_percent = value
        self.overlay.apply_settings(self.config.overlay)

    def _overlay_panel_changed(self, settings: OverlaySettings) -> None:
        self.config.overlay.panel_offset_x = settings.panel_offset_x
        self.config.overlay.panel_offset_y = settings.panel_offset_y
        self.config.overlay.panel_width = settings.panel_width
        self.config.overlay.panel_height = settings.panel_height
        self.config.overlay.scale_percent = settings.scale_percent

    def _frame_settings_changed(self) -> None:
        if not hasattr(self, "frame_check"):
            return
        self.config.overlay.show_frame = self.frame_check.isChecked()
        self.config.overlay.frame_width = self.frame_width_spin.value()
        self.config.overlay.coordinate_scale_percent = self.coordinate_scale_spin.value()
        self.overlay.apply_settings(self.config.overlay)
        self.overlay.update_frame(
            self.corners if len(self.corners) == 4 else None,
            self.capture_origin,
            self.orientation_combo.currentText(),
            self.config.capture.warp_size,
        )
        self._refresh_overview()

    def _choose_frame_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._frame_color), self, "Rahmenfarbe wählen")
        if not color.isValid():
            return
        self._frame_color = color.name()
        self.config.overlay.frame_color = self._frame_color
        self.frame_color_button.setText(self._frame_color)
        self.frame_color_button.setStyleSheet(
            f"background:{self._frame_color}; color:#111; font-weight:700;"
        )
        self.overlay.apply_settings(self.config.overlay)
        self.overlay.update_frame(
            self.corners if len(self.corners) == 4 else None,
            self.capture_origin,
            self.orientation_combo.currentText(),
            self.config.capture.warp_size,
        )

    def _update_overlay_frame(self) -> None:
        """Roter Rahmen folgt der aktuellen Kalibrierung – auch ohne Analyse."""
        if not hasattr(self, "overlay"):
            return
        self.overlay.update_frame(
            self.corners if len(self.corners) == 4 else None,
            self.capture_origin,
            self.orientation_combo.currentText() if hasattr(self, "orientation_combo") else "white_bottom",
            self.config.capture.warp_size,
        )

    def _test_overlay(self) -> None:
        self._sync_engine_settings()
        if self.overlay.user_hidden:
            self.overlay.user_hidden = False
            self.config.overlay.visible = True
        self.overlay.cover_virtual_desktop()
        self.overlay.present()
        self.overlay.apply_settings(self.config.overlay)
        self._update_overlay_frame()
        self.overlay.show_test()
        ensure_topmost(self.overlay)
        ensure_topmost(self.helper)
        self._refresh_overview()
        message = (
            "Overlay-Test aktiv. "
            + ("Rahmen und Demo-Pfeil werden angezeigt." if len(self.corners) == 4
               else "Panel sichtbar; für Rahmen und Pfeile zuerst das Brett einrahmen.")
        )
        logger.info("Overlay-Test: %s", self.overlay.describe_state())
        self.statusBar().showMessage(message, 10_000)

    def _toggle_overlay(self) -> None:
        self.overlay.user_hidden = not self.overlay.user_hidden
        self.config.overlay.visible = not self.overlay.user_hidden
        if self.overlay.user_hidden:
            self.overlay.hide()
        else:
            self.overlay.present()
            self.overlay.set_click_through(True)
            self._update_overlay_frame()
        self._refresh_overview()
        self.statusBar().showMessage("Overlay aus" if self.overlay.user_hidden else "Overlay ein")

    def _install_hotkey(self) -> bool:
        app = QGuiApplication.instance()
        if self.hotkey_filter is not None:
            if app is not None:
                app.removeNativeEventFilter(self.hotkey_filter)
            self.hotkey_filter.unregister()
            self.hotkey_filter = None
        try:
            hotkey_filter = WindowsHotkeyFilter(self.config.overlay.hotkey)
            hotkey_filter.register()
            hotkey_filter.activated.connect(self._toggle_overlay)
            if app is None:
                raise RuntimeError("Qt-Anwendung ist nicht initialisiert.")
            app.installNativeEventFilter(hotkey_filter)
            self.hotkey_filter = hotkey_filter
            return True
        except Exception as exc:
            logger.warning("Hotkey nicht verfügbar: %s", exc)
            self.statusBar().showMessage(f"Hotkey nicht registriert: {exc}", 8000)
            return False

    def _apply_hotkey(self) -> None:
        old_hotkey = self.config.overlay.hotkey
        candidate = self.hotkey_edit.text().strip()
        self.config.overlay.hotkey = candidate
        if not self._install_hotkey():
            self.config.overlay.hotkey = old_hotkey
            self.hotkey_edit.setText(old_hotkey)
            self._install_hotkey()
            QMessageBox.warning(self, "Hotkey", f"Hotkey konnte nicht angewendet werden: {candidate}")
        else:
            self.statusBar().showMessage(f"Hotkey aktiv: {candidate}")

    # ------------------------------------------------------------------- PGN
    def _load_pgn(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "PGN laden", "", "PGN (*.pgn);;Alle Dateien (*)")
        if not path:
            return
        try:
            games = load_pgn_games(Path(path), limit=50)
        except ValueError as exc:
            QMessageBox.warning(self, "PGN", str(exc))
            return
        self.pgn_games = games
        self.pgn_game_combo.blockSignals(True)
        self.pgn_game_combo.clear()
        for index, game in enumerate(games, start=1):
            self.pgn_game_combo.addItem(game_label(game, index))
        self.pgn_game_combo.blockSignals(False)
        self.pgn_game_combo.setCurrentIndex(0)
        self._pgn_game_changed(0)
        self.tabs.setCurrentWidget(self.pgn_tab)
        self.statusBar().showMessage(f"{len(games)} PGN-Partie(n) geladen: {path}")

    def _current_pgn_game(self) -> chess.pgn.Game | None:
        index = self.pgn_game_combo.currentIndex()
        return self.pgn_games[index] if 0 <= index < len(self.pgn_games) else None

    def _pgn_game_changed(self, index: int) -> None:
        if not 0 <= index < len(self.pgn_games):
            return
        game = self.pgn_games[index]
        self.pgn_positions = game_positions(game)
        self.pgn_list.clear()
        for position_index, (san, _comment, _board) in enumerate(self.pgn_positions):
            self.pgn_list.addItem(f"{position_index:03d}  {san}")
        headers = game.headers
        completion = "abgeschlossen" if game_is_completed(game) else "NICHT abgeschlossen – Analyse gesperrt"
        self.pgn_info.setText(
            f"{headers.get('Event', 'Partie')}: {headers.get('White', '?')} – {headers.get('Black', '?')} "
            f"({headers.get('Result', '*')}), {completion}. Hauptvariante; Kommentare erscheinen in der Statuszeile."
        )
        self._set_mode("pgn_completed")
        self.pgn_list.setCurrentRow(0)
        self._pgn_row_changed(0)
        self._update_action_states()

    def _pgn_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.pgn_positions):
            return
        san, comment, board = self.pgn_positions[row]
        self.snapshot_recognized = False
        self._set_board(board)
        extra = f" — {comment}" if comment else ""
        self.statusBar().showMessage(f"PGN {san}{extra}")
        self._set_mode("pgn_completed")
        self._update_action_states()

    def _export_pgn(self) -> None:
        game = self._current_pgn_game() or game_from_board(self.board)
        path, _ = QFileDialog.getSaveFileName(self, "PGN exportieren", "partie.pgn", "PGN (*.pgn)")
        if not path:
            return
        try:
            save_pgn_game(Path(path), game)
        except ValueError as exc:
            QMessageBox.warning(self, "PGN-Export", str(exc))
            return
        self.statusBar().showMessage(f"PGN exportiert: {path}")

    def _export_analysis(self) -> None:
        if self.last_result is None:
            QMessageBox.information(self, "Analyse", "Noch kein Analyseergebnis vorhanden.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Analyse exportieren", "analyse.txt", "Text (*.txt)")
        if not path:
            return
        result = self.last_result
        lines = [
            "Stockfish Vision Chess AI – Trainingsanalyse",
            f"Engine: {result.engine_id}",
            f"FEN: {result.fen}",
            f"Status: {result.check}",
            "Bewertungen aus Weiß-Sicht; positive Werte favorisieren Weiß.",
            "",
            *[
                f"PV{line.multipv} | {line.score_text} | Tiefe {line.depth or '–'} | {line.pv_san}"
                for line in result.lines
            ],
        ]
        try:
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Analyse-Export", str(exc))
            return
        self.statusBar().showMessage(f"Analyse exportiert: {path}")

    # --------------------------------------------------------------- Lifecycle
    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._update_mode_banner()

    def _update_mode_banner(self) -> None:
        labels = {
            "training": "Eigene Trainingsstellung",
            "pgn_completed": "PGN-Nachbereitung abgeschlossener Partien",
            "frozen_snapshot": "Manuell eingefrorener Snapshot",
        }
        self.mode_banner.setText(
            f"TRAININGS- / ANALYSEMODUS · {labels.get(self.mode, 'Gesperrt')} · "
            "keine automatische Live-Engine-Hilfe in laufenden Online-Partien"
        )

    def _update_action_states(self) -> None:
        if not hasattr(self, "btn_snapshot"):
            return
        self.btn_snapshot.setEnabled(not self.capture_in_flight)
        self.btn_recognize.setEnabled(
            not self.recognition_in_flight and self.last_frame is not None and len(self.corners) == 4
        )
        self.btn_analyze_snap.setEnabled(
            self.snapshot_recognized and not self.analysis_in_flight and self.mode == "frozen_snapshot"
        )
        self.btn_analyze_training.setEnabled(not self.analysis_in_flight)
        self.btn_analyze_pgn.setEnabled(not self.analysis_in_flight)
        self.btn_engine_start.setEnabled(not self.engine_start_in_flight)
        self._refresh_overview()

    def _show_fairness(self) -> None:
        QMessageBox.information(
            self,
            "Trainingsmodus",
            "Das Programm analysiert ausschließlich manuell ausgelöste Trainingsstellungen, "
            "bestätigte eingefrorene Snapshots und abgeschlossene PGN-Partien. Es gibt keine "
            "automatische Live-Erkennung mit Engine-Ausgabe und keine Anti-Cheat-Umgehung.",
        )

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._ui_geometry_applied:
            self._ui_geometry_applied = True
            if self.config.ui.start_maximized:
                self.showMaximized()
            else:
                self._initial_geometry()
        self._refresh_overview()

    def _save_config(self) -> None:
        if not self.isMaximized():
            self._capture_window_size()
        try:
            self._persist_config()
        except OSError as exc:
            logger.exception("Konfiguration konnte nicht gespeichert werden")
            QMessageBox.warning(self, "Konfiguration", str(exc))
            return
        self.statusBar().showMessage("Konfiguration gespeichert.")

    def _persist_config(self) -> None:
        if self.isMaximized():
            self.config.ui.start_maximized = True
        else:
            self.config.ui.start_maximized = self.maximized_check.isChecked()
        self.config.ui.window_width = self.window_width_spin.value()
        self.config.ui.window_height = self.window_height_spin.value()
        self._sync_engine_settings()
        self.config.last_fen = self.board.fen()
        save_config(self.config)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.preview_timer.stop()
        if not self.isMaximized() and self.isVisible():
            for widget in (self.window_width_spin, self.window_height_spin):
                widget.blockSignals(True)
            self.window_width_spin.setValue(max(720, self.width()))
            self.window_height_spin.setValue(max(520, self.height()))
            for widget in (self.window_width_spin, self.window_height_spin):
                widget.blockSignals(False)
            self.config.ui.window_width = self.window_width_spin.value()
            self.config.ui.window_height = self.window_height_spin.value()
        app = QGuiApplication.instance()
        if self.hotkey_filter is not None:
            try:
                if app is not None:
                    app.removeNativeEventFilter(self.hotkey_filter)
                self.hotkey_filter.unregister()
            except Exception:
                logger.exception("Hotkey konnte nicht abgemeldet werden")
            self.hotkey_filter = None
        try:
            self._persist_config()
        except OSError:
            logger.exception("Konfiguration beim Beenden konnte nicht gespeichert werden")
        self.worker.request_stop.emit()
        self.engine_thread.quit()
        if not self.engine_thread.wait(5000):
            logger.warning("Engine-Thread beendet nicht rechtzeitig; er wird abgebrochen.")
            self.engine_thread.terminate()
            self.engine_thread.wait(1000)
        for thread in (self.capture_thread, self.recognition_thread):
            thread.quit()
            if not thread.wait(5000):
                logger.warning("Hintergrund-Thread beendet nicht rechtzeitig; er wird abgebrochen.")
                thread.terminate()
                thread.wait(1000)
        self.topmost.stop()
        if self.selector is not None:
            self.selector.close()
            self.selector = None
        self.helper.close()
        self.overlay.close()
        event.accept()

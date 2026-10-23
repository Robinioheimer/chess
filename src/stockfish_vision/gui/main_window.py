from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import chess
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QCloseEvent, QColor, QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
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

from stockfish_vision.board.geometry import board_mean_abs_diff, extract_square, square_to_image_rc, warp_board
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
from stockfish_vision.gui.worker import CaptureWorker, EngineWorker, RecognitionResult, RecognitionWorker
from stockfish_vision.overlay.hotkey import WindowsHotkeyFilter
from stockfish_vision.overlay.window import OverlayWindow
from stockfish_vision.policy import LiveGameGuard, can_analyze
from stockfish_vision.recognition.pieces import Detection

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
    corner_clicked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(480, 320)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#111; color:#ccc;")
        self.setText("Noch kein Screenshot.")
        self._image: np.ndarray | None = None
        self._qimage: QImage | None = None

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

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self._image is None or self.pixmap() is None:
            return
        pixmap = self.pixmap()
        pw, ph = pixmap.width(), pixmap.height()
        x0 = (self.width() - pw) // 2
        y0 = (self.height() - ph) // 2
        mx, my = int(event.position().x()), int(event.position().y())
        if not (x0 <= mx < x0 + pw and y0 <= my < y0 + ph):
            return
        height, width = self._image.shape[:2]
        self.corner_clicked.emit((mx - x0) / pw * width, (my - y0) / ph * height)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.setWindowTitle("Stockfish Vision Chess AI – Trainingsmodus")
        self.resize(1380, 900)

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
        if config.overlay.visible:
            self.overlay.show()
        self.overlay.set_click_through(True)

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
        self._set_board(self.board)
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
        self.capture_tab = self._capture_tab()
        self.training_tab = self._training_tab()
        self.analysis_tab = self._analysis_tab()
        self.engine_tab = self._engine_tab()
        self.pgn_tab = self._pgn_tab()
        self.tabs.addTab(self.capture_tab, "Aufnahme")
        self.tabs.addTab(self.training_tab, "Training & Korrektur")
        self.tabs.addTab(self.analysis_tab, "Analyse")
        self.tabs.addTab(self.engine_tab, "Engine & Overlay")
        self.tabs.addTab(self.pgn_tab, "PGN")
        self.tabs.addTab(self._history_tab(), "Historie")
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

        help_menu = self.menuBar().addMenu("Hilfe")
        fairness_act = QAction("Trainingsmodus und Fairness", self)
        fairness_act.triggered.connect(self._show_fairness)
        help_menu.addAction(fairness_act)

    def _capture_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        splitter = QSplitter()
        left = QWidget()
        form = QVBoxLayout(left)
        self.use_region = QCheckBox("Manuellen Bildschirmbereich verwenden")
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
        region_row = QHBoxLayout()
        for label, spin in (
            ("L", self.region_left),
            ("T", self.region_top),
            ("B", self.region_width),
            ("H", self.region_height),
        ):
            region_row.addWidget(QLabel(label))
            region_row.addWidget(spin)
        self.monitor_combo = QComboBox()
        refresh = QPushButton("Monitore aktualisieren")
        refresh.clicked.connect(self._refresh_monitors)
        self.btn_preview = QPushButton("Vorschau starten")
        self.btn_preview.clicked.connect(self._toggle_preview)
        self.btn_snapshot = QPushButton("Snapshot aufnehmen && Vorschau einfrieren")
        self.btn_snapshot.clicked.connect(self._take_snapshot)
        self.btn_clear_corners = QPushButton("Ecken zurücksetzen")
        self.btn_clear_corners.clicked.connect(self._reset_corners)
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["white_bottom", "white_top", "white_left", "white_right"])
        self.warp_size_spin = QSpinBox()
        self.warp_size_spin.setRange(128, 1600)
        self.warp_size_spin.setSingleStep(64)
        self.confirm_box = QCheckBox(
            "Ich bestätige: eigene Trainingsstellung oder abgeschlossene Partie "
            "(keine laufende Online-Partie)."
        )
        self.confirm_box.stateChanged.connect(self._on_confirm_changed)
        self.btn_recognize = QPushButton("Snapshot im Hintergrund erkennen")
        self.btn_recognize.clicked.connect(self._recognize_snapshot)
        self.btn_analyze_snap = QPushButton("Geprüften Snapshot analysieren")
        self.btn_analyze_snap.clicked.connect(self._analyze_snapshot)
        self.btn_review = QPushButton("Erkennung prüfen / Felder korrigieren")
        self.btn_review.clicked.connect(lambda: self.tabs.setCurrentWidget(self.training_tab))
        self.capture_status = QLabel("Keine Aufnahme. Vorschau und Erkennung starten keine Engine automatisch.")
        self.capture_status.setWordWrap(True)
        form.addWidget(QLabel("Monitor"))
        form.addWidget(self.monitor_combo)
        form.addWidget(refresh)
        form.addWidget(self.use_region)
        form.addLayout(region_row)
        form.addWidget(self.btn_preview)
        form.addWidget(self.btn_snapshot)
        form.addWidget(QLabel("Brettorientierung"))
        form.addWidget(self.orientation_combo)
        form.addWidget(QLabel("Erkennungsgröße (quadratisch)"))
        form.addWidget(self.warp_size_spin)
        form.addWidget(self.btn_clear_corners)
        form.addWidget(self.confirm_box)
        form.addWidget(self.btn_recognize)
        form.addWidget(self.btn_review)
        form.addWidget(self.btn_analyze_snap)
        form.addWidget(self.capture_status)
        form.addStretch()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(
            QLabel("Kalibrierung: vier Ecken im Uhrzeigersinn klicken (oben links zuerst).")
        )
        self.calib_view = CalibrationView()
        self.calib_view.corner_clicked.connect(self._add_corner)
        right_layout.addWidget(self.calib_view)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        return widget

    def _engine_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)
        self.engine_path = QLineEdit()
        self.engine_path.setPlaceholderText("Pfad zur separat installierten stockfish.exe oder STOCKFISH_PATH")
        browse = QPushButton("Durchsuchen…")
        browse.clicked.connect(self._browse_engine)
        path_wrap = self._with_button(self.engine_path, browse)
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

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(60, 200)
        self.hotkey_edit = QLineEdit()
        apply_hotkey = QPushButton("Hotkey anwenden")
        apply_hotkey.clicked.connect(self._apply_hotkey)
        self.btn_overlay = QPushButton("Overlay ein/aus")
        self.btn_overlay.clicked.connect(self._toggle_overlay)

        layout.addRow("Stockfish.exe", path_wrap)
        layout.addRow("Suchtiefe", self.depth_spin)
        layout.addRow("Zeitbudget", self.time_spin)
        layout.addRow("Threads", self.threads_spin)
        layout.addRow("Hash", self.hash_spin)
        layout.addRow("Beste Zugfolgen (MultiPV)", self.multipv_spin)
        layout.addRow(self.btn_engine_start, self.btn_engine_stop)
        layout.addRow("Status", self.engine_status)
        layout.addRow("Figurenmodell (XML)", self._with_button(self.model_path_edit, model_browse))
        layout.addRow("Belegungsschwelle (Std.-Abw.)", self.occupancy_spin)
        layout.addRow("Warnschwelle Konfidenz", self.confidence_spin)
        layout.addRow(apply_recognition)
        layout.addRow("Overlay-Deckkraft %", self.opacity_slider)
        layout.addRow("Overlay-Skalierung %", self.scale_slider)
        layout.addRow("Globaler Hotkey", self._with_button(self.hotkey_edit, apply_hotkey))
        layout.addRow(self.btn_overlay)
        layout.addRow(QLabel("Panel am Titel verschieben, an ◢ skalieren. Pfeile bleiben klickdurchlässig."))
        return widget

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
        self.analysis_summary = QLabel("Noch keine Analyse. Bewertungen werden aus Weiß-Sicht angezeigt.")
        self.analysis_summary.setWordWrap(True)
        self.analysis_tree = QTreeWidget()
        self.analysis_tree.setColumnCount(5)
        self.analysis_tree.setHeaderLabels(["PV", "Bewertung (Weiß)", "Tiefe", "Bester Zug", "Variante (SAN)"])
        self.analysis_tree.header().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
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

    # -------------------------------------------------------------- Settings
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
        if self.config.capture.region:
            self.use_region.setChecked(True)
            self.region_left.setValue(self.config.capture.region["left"])
            self.region_top.setValue(self.config.capture.region["top"])
            self.region_width.setValue(self.config.capture.region["width"])
            self.region_height.setValue(self.config.capture.region["height"])
        self.opacity_slider.valueChanged.connect(self._opacity_changed)
        self.scale_slider.valueChanged.connect(self._scale_changed)

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
        self.config.capture.orientation = self.orientation_combo.currentText()
        self.config.capture.warp_size = self.warp_size_spin.value()
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
        self.calib_view.set_frame(self._annotate_corners(self.display_frame))
        self._update_action_states()

    def _on_capture_error(self, message: str, purpose: str) -> None:
        self.capture_in_flight = False
        if purpose == "preview" and self.live_preview:
            self._toggle_preview()
        self.capture_status.setText(f"Aufnahmefehler: {message}")
        self._update_action_states()

    def _annotate_corners(self, frame: np.ndarray) -> np.ndarray:
        visual = frame.copy()
        for index, (x, y) in enumerate(self.corners):
            cv2.circle(visual, (round(x), round(y)), 8, (0, 220, 120), -1)
            cv2.putText(
                visual,
                str(index + 1),
                (round(x) + 10, round(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 120),
                2,
            )
        if len(self.corners) == 4:
            points = np.array(self.corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(visual, [points], True, (0, 200, 255), 2)
        return visual

    def _add_corner(self, x: float, y: float) -> None:
        if len(self.corners) >= 4:
            self.corners = []
        self.corners.append([float(x), float(y)])
        if self.display_frame is not None:
            self.calib_view.set_frame(self._annotate_corners(self.display_frame))
        self.capture_status.setText(f"{len(self.corners)} / 4 Ecken gesetzt.")
        self._sync_engine_settings()

    def _reset_corners(self) -> None:
        self.corners = []
        self.preview_warped = None
        self.snapshot_recognized = False
        self.config.capture.corners = None
        if self.display_frame is not None:
            self.calib_view.set_frame(self.display_frame)
        self.capture_status.setText("Ecken gelöscht; Snapshot muss nach neuer Kalibrierung erneut erkannt werden.")
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
            "Erkennung abgeschlossen. Konfidenzen unter Training & Korrektur prüfen; keine Auto-Analyse."
        )
        self._update_action_states()

    def _on_recognition_error(self, message: str) -> None:
        self.recognition_in_flight = False
        self.snapshot_recognized = False
        self.pending_warped = None
        self.capture_status.setText("Erkennungsfehler: " + message.split("\n", 1)[0])
        QMessageBox.warning(self, "Erkennungsfehler", message.split("\n", 1)[0])
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
        current = result.fen == self.board.fen()
        if current:
            self.board_widget.set_analysis(result.lines)
            calibrated = self.corners if self.mode == "frozen_snapshot" and len(self.corners) == 4 else None
            self.overlay.apply_settings(self.config.overlay)
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
        logger.error(message)
        self.engine_status.setText("Engine-Fehler: " + message.split("\n", 1)[0])
        QMessageBox.warning(self, "Engine-Fehler", message.split("\n", 1)[0])
        self.statusBar().showMessage("Engine-Fehler; bitte Pfad/Einstellungen prüfen und Engine neu starten.")
        self._update_action_states()

    def _on_engine_started(self, name: str) -> None:
        self.engine_ready = True
        self.engine_start_in_flight = False
        self.engine_status.setText(f"Engine bereit: {name}")
        self.statusBar().showMessage(f"Engine bereit: {name}")
        self._update_action_states()

    def _on_engine_stopped(self) -> None:
        self.engine_ready = False
        self.engine_start_in_flight = False
        self.engine_status.setText("Engine gestoppt.")
        self.statusBar().showMessage("Engine gestoppt")
        self._update_action_states()

    def _start_engine(self) -> None:
        if self.engine_start_in_flight:
            return
        self._sync_engine_settings()
        self.engine_ready = False
        self.engine_start_in_flight = True
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

    def _toggle_overlay(self) -> None:
        self.overlay.user_hidden = not self.overlay.user_hidden
        self.config.overlay.visible = not self.overlay.user_hidden
        if self.overlay.user_hidden:
            self.overlay.hide()
        else:
            self.overlay.cover_virtual_desktop()
            self.overlay.show()
            self.overlay.set_click_through(True)
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

    def _show_fairness(self) -> None:
        QMessageBox.information(
            self,
            "Trainingsmodus",
            "Das Programm analysiert ausschließlich manuell ausgelöste Trainingsstellungen, "
            "bestätigte eingefrorene Snapshots und abgeschlossene PGN-Partien. Es gibt keine "
            "automatische Live-Erkennung mit Engine-Ausgabe und keine Anti-Cheat-Umgehung.",
        )

    def _save_config(self) -> None:
        try:
            self._persist_config()
        except OSError as exc:
            logger.exception("Konfiguration konnte nicht gespeichert werden")
            QMessageBox.warning(self, "Konfiguration", str(exc))
            return
        self.statusBar().showMessage("Konfiguration gespeichert.")

    def _persist_config(self) -> None:
        self._sync_engine_settings()
        self.config.last_fen = self.board.fen()
        save_config(self.config)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.preview_timer.stop()
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
        self.overlay.close()
        event.accept()

from __future__ import annotations

import logging
from pathlib import Path

import chess
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QAction, QImage, QPixmap, QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)

from stockfish_vision.board.geometry import (
    board_mean_abs_diff,
    extract_square,
    square_to_image_rc,
    warp_board,
)
from stockfish_vision.capture.screen import ScreenCapture, list_monitors
from stockfish_vision.chesslogic.fenutil import apply_manual_piece, parse_fen
from stockfish_vision.chesslogic.pgnutil import game_positions, load_pgn_games
from stockfish_vision.config import AppConfig, save_config
from stockfish_vision.gui.board_widget import BoardWidget
from stockfish_vision.gui.worker import EngineWorker
from stockfish_vision.overlay.hotkey import WindowsHotkeyFilter
from stockfish_vision.overlay.window import OverlayWindow
from stockfish_vision.policy import LiveGameGuard, can_analyze
from stockfish_vision.recognition.pieces import PieceClassifier

logger = logging.getLogger(__name__)

PIECE_CHOICES = ["(leer)", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


class CalibrationView(QLabel):
    corner_clicked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(480, 320)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#111; color:#ccc;")
        self.setText("Noch kein Screenshot.")
        self._image: np.ndarray | None = None

    def set_frame(self, bgr: np.ndarray) -> None:
        self._image = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.setPixmap(
            QPixmap.fromImage(qimage).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._image is None or self.pixmap() is None:
            return
        pix = self.pixmap()
        pw, ph = pix.width(), pix.height()
        x0 = (self.width() - pw) // 2
        y0 = (self.height() - ph) // 2
        mx, my = int(event.position().x()), int(event.position().y())
        if not (x0 <= mx < x0 + pw and y0 <= my < y0 + ph):
            return
        nx = (mx - x0) / pw
        ny = (my - y0) / ph
        h, w = self._image.shape[:2]
        self.corner_clicked.emit(nx * w, ny * h)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.setWindowTitle("Stockfish Vision Chess AI")
        self.resize(1280, 840)
        self.capture = ScreenCapture()
        self.classifier = PieceClassifier(
            occupancy_threshold=config.occupancy_std_threshold,
            min_confidence=config.min_piece_confidence,
        )
        self.guard = LiveGameGuard()
        self.snapshot_confirmed = False
        self.mode = "training"
        self.live_preview = False
        self.last_warped: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self.corners: list[list[float]] = list(config.capture.corners or [])
        self.board = parse_fen(config.last_fen)
        self.pgn_positions: list[tuple[str, str, chess.Board]] = []
        self.overlay = OverlayWindow()
        self.overlay.apply_settings(config.overlay)
        self.overlay.cover_virtual_desktop()
        self.overlay.show()
        self.overlay.set_click_through(True)
        self.capture_origin = (0.0, 0.0)

        self.engine_ready = False
        self.engine_thread = QThread(self)
        self.worker = EngineWorker()
        self.worker.moveToThread(self.engine_thread)
        self.worker.request_start.connect(self.worker.start_engine, Qt.ConnectionType.QueuedConnection)
        self.worker.request_analyse.connect(self.worker.analyse, Qt.ConnectionType.QueuedConnection)
        self.worker.request_stop.connect(self.worker.stop_engine, Qt.ConnectionType.QueuedConnection)
        self.engine_thread.start()
        self.worker.finished.connect(self._on_analysis)
        self.worker.failed.connect(self._on_engine_error)
        self.worker.started.connect(self._on_engine_started)
        self.worker.stopped.connect(self._on_engine_stopped)

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(700)
        self.preview_timer.timeout.connect(self._tick_preview)

        self._build_ui()
        self._refresh_monitors()
        self.board_widget.set_board(self.board)
        self._load_settings_into_ui()
        self._install_hotkey()
        self.statusBar().showMessage("Bereit. Live-Online-Hilfe ist deaktiviert.")

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._capture_tab(), "Aufnahme")
        tabs.addTab(self._engine_tab(), "Engine")
        tabs.addTab(self._training_tab(), "Training")
        tabs.addTab(self._pgn_tab(), "PGN")
        tabs.addTab(self._history_tab(), "Historie")
        self.setCentralWidget(tabs)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("Datei")
        save_act = QAction("Konfiguration speichern", self)
        save_act.triggered.connect(self._save_config)
        quit_act = QAction("Beenden", self)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(save_act)
        file_menu.addAction(quit_act)

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
        for spin in (self.region_left, self.region_top, self.region_width, self.region_height):
            spin.setRange(0, 16000)
        self.region_width.setValue(800)
        self.region_height.setValue(800)
        region_row = QHBoxLayout()
        region_row.addWidget(QLabel("L"))
        region_row.addWidget(self.region_left)
        region_row.addWidget(QLabel("T"))
        region_row.addWidget(self.region_top)
        region_row.addWidget(QLabel("B"))
        region_row.addWidget(self.region_width)
        region_row.addWidget(QLabel("H"))
        region_row.addWidget(self.region_height)
        self.monitor_combo = QComboBox()
        self.btn_preview = QPushButton("Vorschau starten")
        self.btn_preview.clicked.connect(self._toggle_preview)
        self.btn_snapshot = QPushButton("Snapshot aufnehmen")
        self.btn_snapshot.clicked.connect(self._take_snapshot)
        self.btn_clear_corners = QPushButton("Ecken zurücksetzen")
        self.btn_clear_corners.clicked.connect(self._reset_corners)
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(
            ["white_bottom", "white_top", "white_left", "white_right"]
        )
        self.confirm_box = QCheckBox(
            "Ich bestätige: eigene Trainingsstellung oder abgeschlossene Partie "
            "(keine laufende Online-Partie)."
        )
        self.confirm_box.stateChanged.connect(self._on_confirm_changed)
        self.btn_recognize = QPushButton("Snapshot erkennen")
        self.btn_recognize.clicked.connect(self._recognize_snapshot)
        self.btn_analyze_snap = QPushButton("Snapshot analysieren")
        self.btn_analyze_snap.clicked.connect(self._analyze_snapshot)
        self.capture_status = QLabel("Keine Aufnahme.")
        self.capture_status.setWordWrap(True)
        form.addWidget(QLabel("Monitor"))
        form.addWidget(self.monitor_combo)
        form.addWidget(self.use_region)
        form.addLayout(region_row)
        form.addWidget(self.btn_preview)
        form.addWidget(self.btn_snapshot)
        form.addWidget(QLabel("Brettorientierung"))
        form.addWidget(self.orientation_combo)
        form.addWidget(self.btn_clear_corners)
        form.addWidget(self.confirm_box)
        form.addWidget(self.btn_recognize)
        form.addWidget(self.btn_analyze_snap)
        form.addWidget(self.capture_status)
        form.addStretch()

        right = QWidget()
        rlayout = QVBoxLayout(right)
        rlayout.addWidget(QLabel("Kalibrierung: vier Ecken im Uhrzeigersinn klicken (oben links zuerst)."))
        self.calib_view = CalibrationView()
        self.calib_view.corner_clicked.connect(self._add_corner)
        rlayout.addWidget(self.calib_view)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        return widget

    def _engine_tab(self) -> QWidget:
        widget = QWidget()
        layout = QFormLayout(widget)
        self.engine_path = QLineEdit()
        browse = QPushButton("Durchsuchen…")
        browse.clicked.connect(self._browse_engine)
        path_row = QHBoxLayout()
        path_row.addWidget(self.engine_path)
        path_row.addWidget(browse)
        path_wrap = QWidget()
        path_wrap.setLayout(path_row)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 40)
        self.time_spin = QSpinBox()
        self.time_spin.setRange(50, 20000)
        self.time_spin.setSuffix(" ms")
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        self.hash_spin = QSpinBox()
        self.hash_spin.setRange(16, 4096)
        self.hash_spin.setSuffix(" MB")
        self.multipv_spin = QSpinBox()
        self.multipv_spin.setRange(1, 5)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.hotkey_edit = QLineEdit()
        self.btn_engine_start = QPushButton("Engine starten")
        self.btn_engine_start.clicked.connect(self._start_engine)
        self.btn_engine_stop = QPushButton("Engine stoppen")
        self.btn_engine_stop.clicked.connect(lambda: self.worker.request_stop.emit())
        self.btn_overlay = QPushButton("Overlay umschalten")
        self.btn_overlay.clicked.connect(self._toggle_overlay)
        layout.addRow("Stockfish.exe", path_wrap)
        layout.addRow("Suchtiefe", self.depth_spin)
        layout.addRow("Zeitbudget", self.time_spin)
        layout.addRow("Threads", self.threads_spin)
        layout.addRow("Hash", self.hash_spin)
        layout.addRow("Varianten (MultiPV)", self.multipv_spin)
        layout.addRow("Overlay-Transparenz %", self.opacity_slider)
        layout.addRow("Hotkey", self.hotkey_edit)
        layout.addRow(self.btn_engine_start, self.btn_engine_stop)
        layout.addRow(self.btn_overlay)
        return widget

    def _training_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        self.board_widget = BoardWidget()
        self.board_widget.square_clicked.connect(self._on_square_clicked)
        side = QVBoxLayout()
        self.fen_edit = QTextEdit()
        self.fen_edit.setFixedHeight(70)
        self.turn_combo = QComboBox()
        self.turn_combo.addItems(["Weiß am Zug", "Schwarz am Zug"])
        self.piece_combo = QComboBox()
        self.piece_combo.addItems(PIECE_CHOICES)
        apply_fen = QPushButton("FEN übernehmen")
        apply_fen.clicked.connect(self._apply_fen)
        start_pos = QPushButton("Grundstellung")
        start_pos.clicked.connect(self._reset_start)
        clear_board = QPushButton("Brett leeren")
        clear_board.clicked.connect(self._clear_board)
        analyze_train = QPushButton("Trainingsstellung analysieren")
        analyze_train.clicked.connect(self._analyze_training)
        self.validity_label = QLabel("–")
        self.validity_label.setWordWrap(True)
        side.addWidget(QLabel("FEN"))
        side.addWidget(self.fen_edit)
        side.addWidget(apply_fen)
        side.addWidget(self.turn_combo)
        side.addWidget(QLabel("Manuelle Figur für ausgewähltes Feld"))
        side.addWidget(self.piece_combo)
        apply_piece = QPushButton("Feld korrigieren")
        apply_piece.clicked.connect(self._apply_piece_correction)
        side.addWidget(apply_piece)
        side.addWidget(start_pos)
        side.addWidget(clear_board)
        side.addWidget(analyze_train)
        side.addWidget(self.validity_label)
        side.addStretch()
        wrap = QWidget()
        wrap.setLayout(side)
        layout.addWidget(self.board_widget, 3)
        layout.addWidget(wrap, 2)
        return widget

    def _pgn_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        row = QHBoxLayout()
        load_btn = QPushButton("PGN laden")
        load_btn.clicked.connect(self._load_pgn)
        analyze_btn = QPushButton("Aktuelle PGN-Stellung analysieren")
        analyze_btn.clicked.connect(self._analyze_pgn)
        row.addWidget(load_btn)
        row.addWidget(analyze_btn)
        self.pgn_list = QListWidget()
        self.pgn_list.currentRowChanged.connect(self._pgn_row_changed)
        self.pgn_info = QLabel("Keine Partie geladen.")
        self.pgn_info.setWordWrap(True)
        layout.addLayout(row)
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

    def _load_settings_into_ui(self) -> None:
        self.engine_path.setText(self.config.engine.executable)
        self.depth_spin.setValue(self.config.engine.depth)
        self.time_spin.setValue(self.config.engine.move_time_ms)
        self.threads_spin.setValue(self.config.engine.threads)
        self.hash_spin.setValue(self.config.engine.hash_mb)
        self.multipv_spin.setValue(self.config.engine.multipv)
        self.opacity_slider.setValue(int(self.config.overlay.opacity * 100))
        self.hotkey_edit.setText(self.config.overlay.hotkey)
        self.orientation_combo.setCurrentText(self.config.capture.orientation)
        self.fen_edit.setPlainText(self.config.last_fen)
        if self.config.capture.region:
            self.use_region.setChecked(True)
            self.region_left.setValue(int(self.config.capture.region.get("left", 0)))
            self.region_top.setValue(int(self.config.capture.region.get("top", 0)))
            self.region_width.setValue(int(self.config.capture.region.get("width", 800)))
            self.region_height.setValue(int(self.config.capture.region.get("height", 800)))
        self.opacity_slider.valueChanged.connect(self._opacity_changed)

    def _install_hotkey(self) -> None:
        self.hotkey_filter = None
        try:
            self.hotkey_filter = WindowsHotkeyFilter(self.config.overlay.hotkey)
            self.hotkey_filter.register()
            self.hotkey_filter.activated.connect(self._toggle_overlay)
            self.app_instance = self.window().windowHandle()
            from PySide6.QtGui import QGuiApplication

            QGuiApplication.instance().installNativeEventFilter(self.hotkey_filter)
        except Exception as exc:
            logger.warning("Hotkey nicht verfügbar: %s", exc)
            self.statusBar().showMessage(f"Hotkey nicht registriert: {exc}", 8000)

    def _refresh_monitors(self) -> None:
        self.monitor_combo.clear()
        try:
            monitors = list_monitors()
        except Exception as exc:
            self.monitor_combo.addItem(f"Fehler: {exc}", 1)
            return
        for info in monitors:
            self.monitor_combo.addItem(info.label, info.index)
        idx = self.monitor_combo.findData(self.config.capture.monitor_index)
        if idx >= 0:
            self.monitor_combo.setCurrentIndex(idx)

    def _sync_engine_settings(self) -> None:
        self.config.engine.executable = self.engine_path.text().strip()
        self.config.engine.depth = self.depth_spin.value()
        self.config.engine.move_time_ms = self.time_spin.value()
        self.config.engine.threads = self.threads_spin.value()
        self.config.engine.hash_mb = self.hash_spin.value()
        self.config.engine.multipv = self.multipv_spin.value()
        self.config.overlay.opacity = self.opacity_slider.value() / 100.0
        self.config.overlay.hotkey = self.hotkey_edit.text().strip() or self.config.overlay.hotkey
        self.config.capture.orientation = self.orientation_combo.currentText()
        self.config.capture.monitor_index = int(self.monitor_combo.currentData() or 1)
        if self.use_region.isChecked() and self.region_width.value() > 0 and self.region_height.value() > 0:
            self.config.capture.region = {
                "left": self.region_left.value(),
                "top": self.region_top.value(),
                "width": self.region_width.value(),
                "height": self.region_height.value(),
            }
        else:
            self.config.capture.region = None
        if self.corners:
            self.config.capture.corners = self.corners

    def _toggle_preview(self) -> None:
        self.live_preview = not self.live_preview
        self.btn_preview.setText("Vorschau stoppen" if self.live_preview else "Vorschau starten")
        if self.live_preview:
            self.preview_timer.start()
        else:
            self.preview_timer.stop()

    def _tick_preview(self) -> None:
        try:
            frame = self._grab()
        except Exception as exc:
            self.capture_status.setText(str(exc))
            return
        self.last_frame = frame
        if len(self.corners) == 4:
            warped, _ = warp_board(frame, self.corners, self.config.capture.warp_size)
            delta = board_mean_abs_diff(self.last_warped, warped)
            self.last_warped = warped
            self.capture_status.setText(
                f"Vorschau aktiv. Änderung: {delta:.1f}. "
                "Engine läuft nicht automatisch mit (kein Live-Assist)."
            )
            self.calib_view.set_frame(self._annotate_corners(frame))
        else:
            self.calib_view.set_frame(self._annotate_corners(frame))

    def _grab(self) -> np.ndarray:
        self._sync_engine_settings()
        region = self.config.capture.region
        if region:
            self.capture_origin = (float(region["left"]), float(region["top"]))
        else:
            monitors = list_monitors()
            match = next((item for item in monitors if item.index == self.config.capture.monitor_index), None)
            if match:
                self.capture_origin = (float(match.left), float(match.top))
        return self.capture.grab(self.config.capture.monitor_index, region)

    def _annotate_corners(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        for i, (x, y) in enumerate(self.corners):
            cv2.circle(vis, (int(x), int(y)), 8, (0, 220, 120), -1)
            cv2.putText(vis, str(i + 1), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)
        if len(self.corners) == 4:
            pts = np.array(self.corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True, (0, 200, 255), 2)
        return vis

    def _add_corner(self, x: float, y: float) -> None:
        if len(self.corners) >= 4:
            self.corners = []
        self.corners.append([float(x), float(y)])
        if self.last_frame is not None:
            self.calib_view.set_frame(self._annotate_corners(self.last_frame))
        self.capture_status.setText(f"{len(self.corners)} / 4 Ecken gesetzt.")

    def _reset_corners(self) -> None:
        self.corners = []
        self.capture_status.setText("Ecken gelöscht.")

    def _take_snapshot(self) -> None:
        try:
            frame = self._grab()
        except Exception as exc:
            QMessageBox.warning(self, "Aufnahme", str(exc))
            return
        self.last_frame = frame
        self.calib_view.set_frame(self._annotate_corners(frame))
        self.capture_status.setText("Snapshot gespeichert. Brett bleibt eingefroren.")
        self.snapshot_confirmed = self.confirm_box.isChecked()

    def _on_confirm_changed(self) -> None:
        self.snapshot_confirmed = self.confirm_box.isChecked()

    def _recognize_snapshot(self) -> None:
        if self.last_frame is None:
            QMessageBox.information(self, "Erkennung", "Bitte zuerst einen Snapshot aufnehmen.")
            return
        if len(self.corners) != 4:
            QMessageBox.information(self, "Erkennung", "Bitte vier Brettecken kalibrieren.")
            return
        warped, _ = warp_board(self.last_frame, self.corners, self.config.capture.warp_size)
        self.last_warped = warped
        turn = chess.WHITE if self.turn_combo.currentIndex() == 0 else chess.BLACK
        board, detections, validity = self.classifier.detect_board(
            warped, self.orientation_combo.currentText(), turn
        )
        self.board = board
        self.board_widget.set_board(board)
        self.board_widget.confidences = {int(d.square): d.confidence for d in detections}
        ok, issues = validity
        low = [d for d in detections if d.occupied and d.confidence < self.config.min_piece_confidence]
        messages = []
        if not ok:
            messages.extend(issues)
        if low:
            names = ", ".join(chess.square_name(d.square) for d in low[:8])
            messages.append(f"Unsichere Felder: {names}. Bitte manuell korrigieren.")
        if self.classifier.model_error:
            messages.append("Hinweis Modell: " + self.classifier.model_error)
        self.validity_label.setText("OK." if not messages else "\n".join(messages))
        self.fen_edit.setPlainText(board.fen())
        self.mode = "frozen_snapshot"
        placement = board.board_fen()
        self.guard.record(placement)
        self.capture_status.setText("Erkennung abgeschlossen. Keine automatische Engine-Aktualisierung.")

    def _analyze_snapshot(self) -> None:
        allowed, reason = can_analyze(
            "frozen_snapshot",
            snapshot_confirmed=self.snapshot_confirmed,
            live_guard=self.guard,
        )
        if not allowed:
            QMessageBox.warning(self, "Analyse gesperrt", reason)
            return
        self.mode = "frozen_snapshot"
        self._run_analysis(self.board)

    def _analyze_training(self) -> None:
        self.mode = "training"
        self._run_analysis(self.board)

    def _analyze_pgn(self) -> None:
        self.mode = "pgn_completed"
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
        self._sync_engine_settings()
        if not self.engine_ready:
            QMessageBox.information(self, "Engine", "Bitte zuerst die Engine starten.")
            return
        self.statusBar().showMessage("Analysiere …")
        self.worker.request_analyse.emit(board.copy(), self.config.engine)

    def _on_analysis(self, result) -> None:
        lines = [
            f"{result.engine_id} | {result.check}",
            *[f"PV{line.multipv} {line.score_text}: {line.pv_san}" for line in result.lines],
        ]
        if result.hanging:
            lines.append("Taktische Drohungen (hängende Figuren): " + ", ".join(result.hanging))
        text = "\n".join(lines)
        self.history_view.append(text + "\n")
        self.statusBar().showMessage(result.lines[0].score_text if result.lines else "Analyse fertig")
        self.overlay.apply_settings(self.config.overlay)
        self.overlay.set_analysis(
            result,
            self.corners if len(self.corners) == 4 else None,
            self.orientation_combo.currentText(),
            self.config.capture.warp_size,
            origin=self.capture_origin,
        )

    def _on_engine_error(self, message: str) -> None:
        logger.error(message)
        QMessageBox.warning(self, "Engine-Fehler", message.split("\n", 1)[0])
        self.statusBar().showMessage("Engine-Fehler")

    def _on_engine_started(self, name: str) -> None:
        self.engine_ready = True
        self.statusBar().showMessage(f"Engine bereit: {name}")

    @Slot()
    def _on_engine_stopped(self) -> None:
        self.engine_ready = False
        self.statusBar().showMessage("Engine gestoppt")

    def _start_engine(self) -> None:
        self._sync_engine_settings()
        self.worker.request_start.emit(self.config.engine, None)

    def _browse_engine(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Stockfish auswählen", "", "Programme (*.exe);;Alle Dateien (*)")
        if path:
            self.engine_path.setText(path)

    def _opacity_changed(self, value: int) -> None:
        self.config.overlay.opacity = value / 100.0
        self.overlay.apply_settings(self.config.overlay)

    def _toggle_overlay(self) -> None:
        self.overlay.user_hidden = not self.overlay.user_hidden
        if self.overlay.user_hidden:
            self.overlay.hide()
        else:
            self.overlay.cover_virtual_desktop()
            self.overlay.show()
            self.overlay.set_click_through(True)
        self.statusBar().showMessage("Overlay aus" if self.overlay.user_hidden else "Overlay ein")

    def _apply_fen(self) -> None:
        try:
            self.board = parse_fen(self.fen_edit.toPlainText().strip())
        except ValueError as exc:
            QMessageBox.warning(self, "FEN", str(exc))
            return
        self.mode = "training"
        self.board_widget.set_board(self.board)
        self.validity_label.setText("FEN übernommen.")

    def _reset_start(self) -> None:
        self.board = chess.Board()
        self.fen_edit.setPlainText(self.board.fen())
        self.board_widget.set_board(self.board)
        self.mode = "training"

    def _clear_board(self) -> None:
        self.board = chess.Board(None)
        self.board.clear_board()
        self.fen_edit.setPlainText(self.board.fen())
        self.board_widget.set_board(self.board)

    def _on_square_clicked(self, square: int) -> None:
        self.statusBar().showMessage(f"Feld {chess.square_name(square)} ausgewählt")

    def _apply_piece_correction(self) -> None:
        if self.board_widget.selected is None:
            QMessageBox.information(self, "Korrektur", "Bitte ein Feld auf dem Brett anklicken.")
            return
        choice = self.piece_combo.currentText()
        symbol = None if choice == "(leer)" else choice
        self.board = apply_manual_piece(self.board, self.board_widget.selected, symbol)
        if symbol and self.last_warped is not None:
            col, row = square_to_image_rc(self.board_widget.selected, self.orientation_combo.currentText())
            tile = extract_square(self.last_warped, col, row)
            self.classifier.remember_correction(tile, symbol)
        self.board_widget.set_board(self.board)
        self.fen_edit.setPlainText(self.board.fen())

    def _load_pgn(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "PGN laden", "", "PGN (*.pgn);;Alle Dateien (*)")
        if not path:
            return
        try:
            games = load_pgn_games(Path(path), limit=1)
        except ValueError as exc:
            QMessageBox.warning(self, "PGN", str(exc))
            return
        self.pgn_positions = game_positions(games[0])
        self.pgn_list.clear()
        headers = games[0].headers
        self.pgn_info.setText(
            f"{headers.get('White', '?')} – {headers.get('Black', '?')} "
            f"({headers.get('Result', '*')})"
        )
        for index, (san, _comment, _board) in enumerate(self.pgn_positions):
            self.pgn_list.addItem(f"{index:03d}  {san}")
        self.mode = "pgn_completed"

    def _pgn_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.pgn_positions):
            return
        san, comment, board = self.pgn_positions[row]
        self.board = board
        self.board_widget.set_board(board)
        self.fen_edit.setPlainText(board.fen())
        extra = f" — {comment}" if comment else ""
        self.statusBar().showMessage(f"PGN {san}{extra}")
        self.mode = "pgn_completed"

    def _save_config(self) -> None:
        self._sync_engine_settings()
        self.config.last_fen = self.board.fen()
        save_config(self.config)
        self.statusBar().showMessage("Konfiguration gespeichert.")

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self.preview_timer.stop()
        try:
            if self.hotkey_filter:
                self.hotkey_filter.unregister()
        except Exception:
            pass
        self.worker.request_stop.emit()
        self.engine_thread.quit()
        self.engine_thread.wait(3000)
        self.overlay.close()
        self._save_config()
        event.accept()

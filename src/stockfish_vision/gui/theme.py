"""Qt-Stylesheet mit einstellbarer Schriftgröße."""

from __future__ import annotations

BASE_FONT_SIZE = 13
MIN_FONT_SCALE = 70
MAX_FONT_SCALE = 200

STYLE = """
QWidget { background: #121417; color: #e8edf2; font-family: "Segoe UI"; font-size: __FONT_SIZE__px; }
QMainWindow { background: #121417; }
QTabWidget::pane { border: 1px solid #2a313a; }
QTabBar::tab { background: #1b2026; padding: 8px 16px; margin-right: 2px; }
QTabBar::tab:selected { background: #2e7d4f; }
QPushButton { background: #2e7d4f; border: none; padding: 8px 12px; border-radius: 6px; }
QPushButton:hover { background: #3b9a63; }
QPushButton:pressed { background: #246540; }
QPushButton:disabled { background: #343a40; color: #89929b; }
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTableWidget, QTreeWidget {
  background: #1b2026; border: 1px solid #343d48; border-radius: 4px; padding: 4px;
}
QHeaderView::section { background: #20262d; padding: 5px; border: none; }
QLabel { color: #d5dde5; }
QGroupBox {
  border: 1px solid #2a313a; border-radius: 6px; margin-top: 14px; padding-top: 8px;
  font-weight: 600; color: #a8e6bf;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QStatusBar { background: #0e1114; color: #9fb0c0; }
QCheckBox { spacing: 8px; }
QSlider::groove:horizontal { height: 6px; background: #2a313a; }
QSlider::handle:horizontal { background: #3b9a63; width: 14px; margin: -5px 0; border-radius: 7px; }
QScrollArea { border: none; }
QToolTip { background: #1b2026; color: #e8edf2; border: 1px solid #343d48; }
"""


def clamp_font_scale(percent: int | float) -> int:
    try:
        value = int(percent)
    except (TypeError, ValueError):
        return 100
    return max(MIN_FONT_SCALE, min(MAX_FONT_SCALE, value))


def font_pixel_size(percent: int | float = 100) -> int:
    scale = clamp_font_scale(percent) / 100.0
    return max(9, round(BASE_FONT_SIZE * scale))


def build_stylesheet(percent: int | float = 100) -> str:
    return STYLE.replace("__FONT_SIZE__", str(font_pixel_size(percent)))

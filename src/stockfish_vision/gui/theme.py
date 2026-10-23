"""Qt-Stylesheet: helles Design mit einstellbarer Schriftgröße.

Weißer Hintergrund, schwarzer Text, graue Akzente. Nur Warn- und Fehlerzustände
nutzen zusätzlich Rot/Orange, damit sie sofort auffallen.
"""

from __future__ import annotations

BASE_FONT_SIZE = 13
MIN_FONT_SCALE = 70
MAX_FONT_SCALE = 200

# Farbpalette (zentral, damit auch Code mit eigenen Stylesheets sie nutzen kann)
BACKGROUND = "#ffffff"
SURFACE = "#f7f7f8"
SURFACE_ALT = "#efeff1"
BORDER = "#cfd1d4"
BORDER_STRONG = "#a9adb3"
TEXT = "#111214"
TEXT_MUTED = "#5b6067"
ACCENT = "#6b7280"
ACCENT_HOVER = "#7b8290"
ACCENT_PRESSED = "#565c66"
SUCCESS = "#1f7a3f"
WARNING = "#9a5b00"
ERROR = "#b3261e"

STYLE = """
QWidget {
  background: #ffffff; color: #111214;
  font-family: "Segoe UI"; font-size: __FONT_SIZE__px;
}
QMainWindow, QDialog { background: #ffffff; }
QTabWidget::pane { border: 1px solid #cfd1d4; background: #ffffff; }
QTabBar::tab {
  background: #f1f2f4; color: #3a3f46; padding: 8px 16px; margin-right: 2px;
  border-top-left-radius: 6px; border-top-right-radius: 6px;
}
QTabBar::tab:selected { background: #ffffff; color: #111214; font-weight: 600; border-bottom: 2px solid #6b7280; }
QTabBar::tab:hover { background: #e7e8eb; }
QPushButton {
  background: #eceef1; color: #111214; border: 1px solid #cfd1d4;
  padding: 8px 12px; border-radius: 6px;
}
QPushButton:hover { background: #e1e4e8; border-color: #b8bcc2; }
QPushButton:pressed { background: #d6d9de; }
QPushButton:disabled { background: #f4f5f6; color: #9aa0a6; border-color: #e2e4e7; }
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox,
QListWidget, QTableWidget, QTreeWidget {
  background: #ffffff; color: #111214; border: 1px solid #cfd1d4;
  border-radius: 4px; padding: 4px;
  selection-background-color: #d8dbe0; selection-color: #111214;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus, QListWidget:focus, QTableWidget:focus,
QTreeWidget:focus { border-color: #8a9099; }
QComboBox QAbstractItemView { background: #ffffff; color: #111214; selection-background-color: #e5e7eb; selection-color: #111214; }
QHeaderView::section {
  background: #f1f2f4; color: #33383e; padding: 5px;
  border: none; border-right: 1px solid #dfe1e4; border-bottom: 1px solid #cfd1d4;
}
QTableWidget { gridline-color: #e3e5e8; alternate-background-color: #fafafb; }
QLabel { color: #111214; background: transparent; }
QGroupBox {
  border: 1px solid #cfd1d4; border-radius: 8px; margin-top: 16px;
  padding-top: 10px; font-weight: 600; color: #33383e; background: #ffffff;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #33383e; }
QCheckBox { spacing: 8px; color: #111214; }
QCheckBox::indicator {
  width: 15px; height: 15px; border: 1px solid #9aa0a6; border-radius: 3px; background: #ffffff;
}
QCheckBox::indicator:checked { background: #6b7280; border-color: #6b7280; }
QRadioButton { spacing: 8px; }
QStatusBar { background: #f4f5f6; color: #4b5158; border-top: 1px solid #dfe1e4; }
QMenuBar { background: #ffffff; color: #111214; }
QMenuBar::item:selected { background: #e9ebee; }
QMenu { background: #ffffff; color: #111214; border: 1px solid #cfd1d4; }
QMenu::item:selected { background: #e9ebee; }
QSlider::groove:horizontal { height: 6px; background: #e2e4e7; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #c3c7cd; border-radius: 3px; }
QSlider::handle:horizontal {
  background: #6b7280; width: 14px; margin: -5px 0; border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: #565c66; }
QScrollArea { border: none; background: #ffffff; }
QScrollBar:vertical { background: #f4f5f6; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #c8ccd2; min-height: 24px; border-radius: 6px; }
QScrollBar::handle:vertical:hover { background: #b2b7be; }
QScrollBar:horizontal { background: #f4f5f6; height: 12px; }
QScrollBar::handle:horizontal { background: #c8ccd2; min-width: 24px; border-radius: 6px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QToolTip { background: #ffffff; color: #111214; border: 1px solid #cfd1d4; padding: 4px; }
QSplitter::handle { background: #e9ebee; }
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

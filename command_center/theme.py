# =====================================================
# Phase 8 -- Professional Polish. One dark "operations center" QSS
# stylesheet, applied once at the top level (command_center.main_window.
# MainWindow.__init__) so it cascades to every existing and new panel
# with no per-widget changes. Builds on the palette this codebase
# already established rather than inventing a new one: the charcoal
# panel background (#2b2b2b) designer/widgets/bottom_info_bar.py already
# uses, and the severity colors (safe/moderate/elevated/critical)
# command_center/building_view.py already uses for zone/exit/stair
# coloring -- restated here as named constants so both modules read from
# the same source of truth without importing PyQt widget code across
# layers.
# =====================================================

BACKGROUND = "#1e1f22"
PANEL_BACKGROUND = "#2b2b2b"
PANEL_BORDER = "#3c3f41"
TEXT = "#e6e6e6"
MUTED_TEXT = "#9aa0a6"
ACCENT = "#4a90d9"
SAFE = "#428563"
MODERATE = "#e09c2d"
ELEVATED = "#d65b2d"
CRITICAL = "#b02121"

COMMAND_CENTER_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-size: 10pt;
}}

QGroupBox {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {PANEL_BORDER};
    border-radius: 4px;
    margin-top: 14px;
    padding: 8px 6px 6px 6px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: {ACCENT};
}}

QLabel {{
    color: {TEXT};
    background: transparent;
}}

QFrame#StatTile {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {PANEL_BORDER};
    border-radius: 4px;
}}

QLabel#StatTileCaption {{
    color: {MUTED_TEXT};
    font-size: 8pt;
}}

QTabWidget::pane {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {PANEL_BORDER};
    top: -1px;
}}

QTabBar::tab {{
    background-color: {BACKGROUND};
    color: {MUTED_TEXT};
    padding: 6px 14px;
    border: 1px solid {PANEL_BORDER};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background-color: {PANEL_BACKGROUND};
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}

QTabBar::tab:hover {{
    color: {TEXT};
}}

QTableWidget {{
    background-color: {PANEL_BACKGROUND};
    alternate-background-color: #313335;
    gridline-color: {PANEL_BORDER};
    border: 1px solid {PANEL_BORDER};
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}

QHeaderView::section {{
    background-color: #313335;
    color: {TEXT};
    padding: 4px;
    border: none;
    border-bottom: 1px solid {PANEL_BORDER};
    font-weight: bold;
}}

QPushButton {{
    background-color: #3c3f41;
    color: {TEXT};
    border: 1px solid {PANEL_BORDER};
    border-radius: 3px;
    padding: 4px 12px;
}}

QPushButton:hover {{
    background-color: #47494b;
    border-color: {ACCENT};
}}

QPushButton:pressed {{
    background-color: {ACCENT};
}}

QComboBox, QDoubleSpinBox {{
    background-color: #3c3f41;
    color: {TEXT};
    border: 1px solid {PANEL_BORDER};
    border-radius: 3px;
    padding: 2px 6px;
}}

QSlider::groove:horizontal {{
    background: {PANEL_BORDER};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}

QMenuBar {{
    background-color: {PANEL_BACKGROUND};
    color: {TEXT};
}}

QMenuBar::item:selected {{
    background-color: {ACCENT};
}}

QMenu {{
    background-color: {PANEL_BACKGROUND};
    color: {TEXT};
    border: 1px solid {PANEL_BORDER};
}}

QMenu::item:selected {{
    background-color: {ACCENT};
}}

QSplitter::handle {{
    background-color: {PANEL_BORDER};
}}
"""

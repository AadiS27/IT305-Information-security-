"""
Styles - UI Styling (Separate from logic for maintainability)
Modern, clean design with professional color scheme
"""

class Styles:
    """
    Centralized stylesheet management.
    Performance benefit: Single stylesheet application is faster than individual widget styling.
    """
    
    @staticmethod
    def get_main_stylesheet():
        """
        Return main application stylesheet with modern design.
        Uses flat design principles and consistent color palette.
        """
        return """
        /* Main Window */
        QMainWindow {
            background-color: #2c3e50;
        }
        
        /* Menu Bar */
        QMenuBar {
            background-color: #34495e;
            color: #ecf0f1;
            border-bottom: 2px solid #1abc9c;
            padding: 5px;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 5px 10px;
            border-radius: 3px;
        }
        
        QMenuBar::item:selected {
            background-color: #1abc9c;
        }
        
        QMenu {
            background-color: #34495e;
            color: #ecf0f1;
            border: 1px solid #1abc9c;
        }
        
        QMenu::item:selected {
            background-color: #1abc9c;
        }
        
        /* Status Bar */
        QStatusBar {
            background-color: #34495e;
            color: #ecf0f1;
            border-top: 2px solid #1abc9c;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 13px;
            font-weight: bold;
            min-height: 35px;
        }
        
        QPushButton:hover {
            background-color: #2980b9;
        }
        
        QPushButton:pressed {
            background-color: #21618c;
        }
        
        QPushButton:disabled {
            background-color: #7f8c8d;
            color: #bdc3c7;
        }
        
        /* Group Boxes */
        QGroupBox {
            background-color: #34495e;
            border: 2px solid #1abc9c;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 15px;
            font-weight: bold;
            color: #ecf0f1;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 10px;
            background-color: #1abc9c;
            color: white;
            border-radius: 3px;
        }
        
        /* Labels */
        QLabel {
            color: #ecf0f1;
            font-size: 12px;
        }
        
        /* Text Edit / Log Area */
        QTextEdit {
            background-color: #1e272e;
            color: #ecf0f1;
            border: 2px solid #34495e;
            border-radius: 5px;
            padding: 5px;
            font-family: Consolas, monospace;
            selection-background-color: #3498db;
        }
        
        /* Frames / Panels */
        QFrame {
            background-color: #2c3e50;
            border-radius: 5px;
        }
        
        /* Splitter */
        QSplitter::handle {
            background-color: #1abc9c;
        }
        
        QSplitter::handle:horizontal {
            width: 3px;
        }
        
        QSplitter::handle:vertical {
            height: 3px;
        }
        
        /* Scroll Bars */
        QScrollBar:vertical {
            background-color: #2c3e50;
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #3498db;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #2980b9;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QScrollBar:horizontal {
            background-color: #2c3e50;
            height: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #3498db;
            border-radius: 6px;
            min-width: 20px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #2980b9;
        }
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        
        /* Input Dialogs */
        QInputDialog {
            background-color: #34495e;
        }
        
        QLineEdit {
            background-color: #1e272e;
            color: #ecf0f1;
            border: 2px solid #3498db;
            border-radius: 5px;
            padding: 8px;
            font-size: 12px;
        }
        
        QLineEdit:focus {
            border: 2px solid #1abc9c;
        }
        
        /* Message Boxes */
        QMessageBox {
            background-color: #34495e;
        }
        
        QMessageBox QLabel {
            color: #ecf0f1;
            font-size: 13px;
        }
        
        QMessageBox QPushButton {
            min-width: 80px;
        }
        """

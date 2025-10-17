"""
Modern PyQt5 GUI for Face Authentication & Cryptography System
Entry point that launches the application
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow

def main():
    """Initialize and run the application"""
    # Enable high DPI scaling for modern displays
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern cross-platform style
    
    # Set application metadata
    app.setApplicationName("Face Auth Crypto")
    app.setOrganizationName("Security Systems")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

"""
Main Window - View Component (MVC Pattern)
Handles UI layout and delegates actions to the controller
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTextEdit, QStatusBar, 
                             QMenuBar, QAction, QFileDialog, QMessageBox,
                             QGroupBox, QGridLayout, QSplitter, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
from .controller import AppController
from .styles import Styles

class MainWindow(QMainWindow):
    """
    Main application window following MVC pattern.
    Separates UI concerns from business logic for maintainability.
    """
    
    # Custom signals for non-blocking operations
    status_update = pyqtSignal(str, str)  # message, level (info/warning/error)
    
    def __init__(self):
        super().__init__()
        
        # Initialize controller (handles all business logic)
        self.controller = AppController(self)
        
        # Connect signals
        self.status_update.connect(self._update_status)
        
        # Setup UI
        self._init_ui()
        self._apply_styles()
        
        # Start timer for session updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_session_info)
        self.update_timer.start(1000)  # Update every second
        
    def _init_ui(self):
        """Initialize all UI components - organized and modular"""
        self.setWindowTitle("Face Authentication & Cryptography System")
        self.setGeometry(100, 100, 1200, 700)
        self.setMinimumSize(900, 600)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - User Actions
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Information Display
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (40% left, 60% right)
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready", 3000)
        
        # Create permanent status labels
        self.auth_status_label = QLabel("Not Authenticated")
        self.auth_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.status_bar.addPermanentWidget(self.auth_status_label)
        
    def _create_menu_bar(self):
        """Create menu bar with organized actions"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        encrypt_action = QAction('Encrypt File', self)
        encrypt_action.setShortcut('Ctrl+E')
        encrypt_action.triggered.connect(self.controller.encrypt_file)
        file_menu.addAction(encrypt_action)
        
        decrypt_action = QAction('Decrypt File', self)
        decrypt_action.setShortcut('Ctrl+D')
        decrypt_action.triggered.connect(self.controller.decrypt_file)
        file_menu.addAction(decrypt_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # User menu
        user_menu = menubar.addMenu('&User')
        
        register_action = QAction('Register New User', self)
        register_action.triggered.connect(self.controller.register_user)
        user_menu.addAction(register_action)
        
        authenticate_action = QAction('Authenticate', self)
        authenticate_action.setShortcut('Ctrl+A')
        authenticate_action.triggered.connect(self.controller.authenticate_user)
        user_menu.addAction(authenticate_action)
        
        logout_action = QAction('Logout', self)
        logout_action.triggered.connect(self.controller.logout)
        user_menu.addAction(logout_action)
        
        # Models menu
        models_menu = menubar.addMenu('&Models')
        
        train_action = QAction('Train Recognition Models', self)
        train_action.triggered.connect(self.controller.train_models)
        models_menu.addAction(train_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
    def _create_left_panel(self):
        """Create left panel with user actions - organized by category"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("User Actions")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Authentication group
        auth_group = self._create_auth_group()
        layout.addWidget(auth_group)
        
        # File Operations group
        file_group = self._create_file_operations_group()
        layout.addWidget(file_group)
        
        # Folder Operations group
        folder_group = self._create_folder_operations_group()
        layout.addWidget(folder_group)
        
        # Model Management group
        model_group = self._create_model_group()
        layout.addWidget(model_group)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        return panel
        
    def _create_auth_group(self):
        """Create authentication controls group"""
        group = QGroupBox("Authentication")
        layout = QVBoxLayout()
        
        register_btn = QPushButton("Register New User")
        register_btn.clicked.connect(self.controller.register_user)
        layout.addWidget(register_btn)
        
        auth_btn = QPushButton("Authenticate User")
        auth_btn.clicked.connect(self.controller.authenticate_user)
        layout.addWidget(auth_btn)
        
        logout_btn = QPushButton("Logout")
        logout_btn.clicked.connect(self.controller.logout)
        layout.addWidget(logout_btn)
        
        group.setLayout(layout)
        return group
        
    def _create_file_operations_group(self):
        """Create file operations group"""
        group = QGroupBox("File Operations")
        layout = QVBoxLayout()
        
        encrypt_file_btn = QPushButton("Encrypt File")
        encrypt_file_btn.clicked.connect(self.controller.encrypt_file)
        layout.addWidget(encrypt_file_btn)
        
        decrypt_file_btn = QPushButton("Decrypt File")
        decrypt_file_btn.clicked.connect(self.controller.decrypt_file)
        layout.addWidget(decrypt_file_btn)
        
        group.setLayout(layout)
        return group
        
    def _create_folder_operations_group(self):
        """Create folder operations group"""
        group = QGroupBox("Folder Operations")
        layout = QVBoxLayout()
        
        encrypt_folder_btn = QPushButton("Encrypt Folder")
        encrypt_folder_btn.clicked.connect(self.controller.encrypt_folder)
        layout.addWidget(encrypt_folder_btn)
        
        decrypt_folder_btn = QPushButton("Decrypt Folder")
        decrypt_folder_btn.clicked.connect(self.controller.decrypt_folder)
        layout.addWidget(decrypt_folder_btn)
        
        group.setLayout(layout)
        return group
        
    def _create_model_group(self):
        """Create model management group"""
        group = QGroupBox("Model Management")
        layout = QVBoxLayout()
        
        train_btn = QPushButton("Train Models")
        train_btn.clicked.connect(self.controller.train_models)
        layout.addWidget(train_btn)
        
        view_info_btn = QPushButton("View System Info")
        view_info_btn.clicked.connect(self.controller.view_system_info)
        layout.addWidget(view_info_btn)
        
        group.setLayout(layout)
        return group
        
    def _create_right_panel(self):
        """Create right panel for information display"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("System Information & Logs")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Session info
        self.session_info_label = QLabel("Session: Not Started")
        self.session_info_label.setWordWrap(True)
        self.session_info_label.setStyleSheet("padding: 10px; background-color: #34495e; color: white; border-radius: 5px;")
        layout.addWidget(self.session_info_label)
        
        # Log area
        log_label = QLabel("Activity Log:")
        log_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)
        
        return panel
        
    def _apply_styles(self):
        """Apply modern styles to the application"""
        self.setStyleSheet(Styles.get_main_stylesheet())
        
    def _update_status(self, message, level="info"):
        """Update status bar with color-coded messages"""
        if level == "error":
            color = "#e74c3c"
        elif level == "warning":
            color = "#f39c12"
        elif level == "success":
            color = "#27ae60"
        else:
            color = "#3498db"
            
        self.status_bar.setStyleSheet(f"color: {color};")
        self.status_bar.showMessage(message, 5000)
        
    def _update_session_info(self):
        """Update session information display"""
        info = self.controller.get_session_info()
        self.session_info_label.setText(info)
        
        # Update auth status
        if self.controller.is_authenticated():
            self.auth_status_label.setText(f"✓ {self.controller.get_authenticated_user()}")
            self.auth_status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        else:
            self.auth_status_label.setText("Not Authenticated")
            self.auth_status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            
    def append_log(self, message, level="info"):
        """Append message to log area with timestamp and color"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if level == "error":
            color = "#e74c3c"
            prefix = "✗"
        elif level == "warning":
            color = "#f39c12"
            prefix = "⚠"
        elif level == "success":
            color = "#27ae60"
            prefix = "✓"
        else:
            color = "#3498db"
            prefix = "ℹ"
            
        formatted_msg = f'<span style="color: {color};">[{timestamp}] {prefix} {message}</span>'
        self.log_text.append(formatted_msg)
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Face Auth Crypto",
            "<h2>Face Authentication & Cryptography System</h2>"
            "<p>Version 1.0</p>"
            "<p>A secure system combining biometric face recognition "
            "with file encryption.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Multi-model face recognition (SVM, RF, NN)</li>"
            "<li>Per-user encryption keys</li>"
            "<li>Secure file and folder operations</li>"
            "</ul>"
        )
        
    def closeEvent(self, event):
        """Handle window close event"""
        reply = QMessageBox.question(
            self,
            'Confirm Exit',
            'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Cleanup
            if self.controller.is_authenticated():
                self.controller.logout()
            event.accept()
        else:
            event.ignore()

"""
Controller - Business Logic Component (MVC Pattern)
Bridges UI and model, delegates to existing system components
"""

import os
import sys
from pathlib import Path
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox

# Add parent directory to path to import existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face_recognizer_opencv import FaceRecognizer
from src.crypto_manager import CryptoManager
from src.face_capture import FaceCapture


class WorkerThread(QThread):
    """
    Worker thread for long-running operations to keep GUI responsive.
    Performance optimization: prevents UI freezing during ML training or encryption.
    """
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(str)  # progress message
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        """Execute the function in background thread"""
        try:
            result = self.func(*self.args, **self.kwargs)
            if result is False:
                self.finished.emit(False, "Operation failed")
            else:
                self.finished.emit(True, "Operation completed successfully")
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")


class AppController(QObject):
    """
    Main controller handling all business logic.
    MVC Performance Benefits:
    - Separates UI from logic for easier testing and maintenance
    - Enables threaded operations without blocking UI
    - Centralizes state management for consistency
    """
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        # Initialize components
        self.face_recognizer = FaceRecognizer()
        self.crypto_manager = CryptoManager()
        self.face_capture = FaceCapture()  # Add face capture instance
        
        # Session state
        self.authenticated_user = None
        self.session_start_time = None
        
        # Paths
        self.base_path = Path(__file__).parent.parent
        self.face_data_path = self.base_path / "face_data"
        self.face_key_path = self.base_path / "face_key"
        self.encrypted_path = self.base_path / "encrypted_files"
        
        # Create directories if needed
        self.face_data_path.mkdir(exist_ok=True)
        self.face_key_path.mkdir(exist_ok=True)
        self.encrypted_path.mkdir(exist_ok=True)
        
    def register_user(self):
        """Register a new user with face capture"""
        username, ok = QInputDialog.getText(
            self.main_window,
            "Register User",
            "Enter username:"
        )
        
        if not ok or not username:
            return
            
        username = username.strip()
        if not username:
            self._show_error("Username cannot be empty")
            return
            
        user_dir = self.face_data_path / username
        if user_dir.exists():
            self._show_error(f"User '{username}' already exists")
            return
            
        self._log(f"Starting registration for user: {username}")
        
        try:
            # Capture face samples (auto-capture enabled)
            # Note: capture_face_samples creates the directory itself
            self._show_info(
                "Face Capture",
                "Auto-capture will begin in 2 seconds.\n"
                "Position your face in the frame.\n"
                "Multiple images will be captured automatically.\n\n"
                "Press 'q' to finish."
            )
            
            sample_count = self.face_capture.capture_face_samples(username, num_samples=20, save_dir=str(self.face_data_path))
            
            if sample_count < 2:
                # Clean up directory if registration failed
                import shutil
                if user_dir.exists():
                    shutil.rmtree(user_dir)
                self._show_error(f"Registration failed: Only {sample_count} samples captured. Need at least 2.")
                self._log(f"Registration failed: insufficient samples ({sample_count})", "error")
                return
                
            self._log(f"Successfully registered user: {username} ({sample_count} samples)", "success")
            self._show_success("Registration", f"User '{username}' registered successfully with {sample_count} face samples")
            
        except Exception as e:
            self._log(f"Registration failed: {str(e)}", "error")
            self._show_error(f"Registration failed: {str(e)}")
            
    def authenticate_user(self):
        """Authenticate user via face recognition"""
        if self.authenticated_user:
            self._show_warning("Already authenticated as: " + self.authenticated_user)
            return
            
        self._log("Starting authentication...")
        
        # Check if models are loaded
        models_dir = self.base_path / "models"
        if not models_dir.exists() or not self.face_recognizer._models_exist():
            self._show_error(
                "Models not found!\n"
                "Please train the models first:\n"
                "1. Register at least 2 users\n"
                "2. Click 'Train Models'\n"
                "3. Then try authentication again"
            )
            self._log("Authentication aborted: Models not trained", "warning")
            return
        
        # Load models if not already loaded
        if self.face_recognizer.svm_model is None:
            self._log("Loading trained models...")
            if not self.face_recognizer.load_models():
                self._show_error("Failed to load models. Please train models first.")
                return
            self._log("Models loaded successfully")
        
        try:
            # Capture face for authentication (auto-capture enabled)
            self._show_info(
                "Face Authentication",
                "Auto-capture will begin after face is stable.\n"
                "Position your face in the frame and hold steady.\n"
                "Capture will happen automatically after 1.5 seconds."
            )
            
            temp_path = self.base_path / "temp_auth.jpg"
            face_image = self.face_capture.capture_single_face()
            
            if face_image is None:
                self._show_error("Failed to capture face")
                return
            
            # Save the captured face to temp file
            import cv2
            cv2.imwrite(str(temp_path), face_image)
                
            # Recognize face (returns tuple: username, confidence)
            self._log("Recognizing face using ensemble models...")
            result = self.face_recognizer.recognize_face(str(temp_path), method='ensemble')
            
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            
            # Handle result
            if result is None or len(result) != 2:
                self._log("Authentication failed: Invalid recognition result", "error")
                self._show_error("Authentication failed: Error in face recognition")
                return
            
            username, confidence = result
                
            if not username or username == "Unknown":
                # Check if confidence was close to threshold
                if confidence >= 0.35:  # Close to threshold
                    self._log(f"Authentication failed: Unknown user (confidence: {confidence:.2f} - close to threshold)", "warning")
                    self._show_error(
                        f"Authentication failed: User not recognized\n"
                        f"Confidence: {confidence:.2%}\n\n"
                        f"Suggestions:\n"
                        f"• Ensure good lighting (similar to registration)\n"
                        f"• Face camera directly\n"
                        f"• Try again or re-register with more samples (15+)\n"
                        f"• Consider re-training models"
                    )
                else:
                    self._log(f"Authentication failed: Unknown user (confidence: {confidence:.2f})", "error")
                    self._show_error(
                        f"Authentication failed: User not recognized\n"
                        f"Confidence: {confidence:.2%}\n\n"
                        f"Low confidence suggests:\n"
                        f"• You may not be registered in the system\n"
                        f"• Very different lighting/angle from registration\n"
                        f"• Register as a new user first"
                    )
                return
            
            self._log(f"Face recognized as: {username} (confidence: {confidence:.2f})")
                
            # Generate or load user key
            self._generate_user_key(username)
            self._load_user_face_key(username)
            
            # Update session
            self.authenticated_user = username
            from datetime import datetime
            self.session_start_time = datetime.now()
            
            self._log(f"Successfully authenticated as: {username}", "success")
            self._show_success("Authentication", f"Welcome, {username}!\nConfidence: {confidence:.2%}")
            
        except Exception as e:
            self._log(f"Authentication error: {str(e)}", "error")
            self._show_error(f"Authentication failed: {str(e)}")
            
    def logout(self):
        """Logout current user"""
        if not self.authenticated_user:
            self._show_warning("No user is currently authenticated")
            return
            
        user = self.authenticated_user
        self.authenticated_user = None
        self.session_start_time = None
        
        self._log(f"User {user} logged out", "success")
        self._show_info("Logout", "Logged out successfully")
        
    def encrypt_file(self):
        """Encrypt a selected file"""
        if not self._check_authentication():
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select File to Encrypt",
            str(self.base_path),
            "All Files (*.*)"
        )
        
        if not file_path:
            return
            
        self._log(f"Encrypting file: {file_path}")
        
        try:
            encrypted_path = self.crypto_manager.encrypt_file(file_path, delete_original=True)
            self._log(f"File encrypted successfully: {encrypted_path}", "success")
            self._show_success("Encryption", f"File encrypted successfully!\nLocation: {encrypted_path}")
        except Exception as e:
            self._log(f"Encryption failed: {str(e)}", "error")
            self._show_error(f"Encryption failed: {str(e)}")
            
    def decrypt_file(self):
        """Decrypt a selected file"""
        if not self._check_authentication():
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select File to Decrypt",
            str(self.encrypted_path),
            "Encrypted Files (*.encrypted);;All Files (*.*)"
        )
        
        if not file_path:
            return
            
        self._log(f"Decrypting file: {file_path}")
        
        try:
            decrypted_path = self.crypto_manager.decrypt_file(file_path)
            self._log(f"File decrypted successfully: {decrypted_path}", "success")
            self._show_success("Decryption", f"File decrypted successfully!\nLocation: {decrypted_path}")
        except Exception as e:
            self._log(f"Decryption failed: {str(e)}", "error")
            self._show_error(f"Decryption failed: {str(e)}")
            
    def encrypt_folder(self):
        """Encrypt all files in a selected folder"""
        if not self._check_authentication():
            return
            
        folder_path = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select Folder to Encrypt",
            str(self.base_path)
        )
        
        if not folder_path:
            return
            
        self._log(f"Encrypting folder: {folder_path}")
        
        # Use worker thread for long operation
        def encrypt_operation():
            self.crypto_manager.create_encrypted_folder(folder_path)
            return True
            
        self._run_threaded_operation(
            encrypt_operation,
            "Folder Encryption",
            "Folder encrypted successfully!"
        )
        
    def decrypt_folder(self):
        """Decrypt all files in a selected folder"""
        if not self._check_authentication():
            return
            
        folder_path = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select Encrypted Folder to Decrypt",
            str(self.base_path)
        )
        
        if not folder_path:
            return
            
        if not folder_path.endswith("_encrypted"):
            self._show_warning("Please select an encrypted folder (ends with '_encrypted')")
            return
            
        self._log(f"Decrypting folder: {folder_path}")
        
        # Use worker thread for long operation
        def decrypt_operation():
            self.crypto_manager.decrypt_folder(folder_path)
            return True
            
        self._run_threaded_operation(
            decrypt_operation,
            "Folder Decryption",
            "Folder decrypted successfully!"
        )
        
    def train_models(self):
        """Train face recognition models"""
        self._log("Starting model training...")
        
        # Check if we have enough users with valid face data
        users = []
        insufficient_users = []
        for d in self.face_data_path.iterdir():
            if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('_'):
                # Check if directory has face images
                images = list(d.glob("*.jpg")) + list(d.glob("*.png"))
                if len(images) >= 2:  # Need at least 2 images per user
                    users.append((d.name, len(images)))
                elif len(images) == 1:
                    insufficient_users.append((d.name, len(images)))
        
        # Check for users with insufficient samples
        if insufficient_users:
            user_list = ", ".join([f"{name} ({count} sample)" for name, count in insufficient_users])
            self._show_error(
                f"Some users don't have enough face samples!\n\n"
                f"Users with insufficient samples:\n{user_list}\n\n"
                f"REQUIREMENT: Each user needs at least 2 face samples.\n\n"
                f"Please:\n"
                f"1. Delete these users' folders, OR\n"
                f"2. Re-register them to capture more samples"
            )
            self._log(f"Training aborted: Users with insufficient samples: {user_list}", "error")
            return
        
        if len(users) < 2:
            self._show_error(
                f"Need at least 2 registered users with face data to train models.\n"
                f"Currently have {len(users)} valid user(s).\n\n"
                f"REQUIREMENT: Minimum 2 users with 2+ samples each.\n\n"
                f"Please register more users with face samples."
            )
            self._log(f"Training aborted: Only {len(users)} valid user(s) found", "warning")
            return
        
        user_names = [name for name, count in users]
        self._log(f"Found {len(users)} users for training:")
        for name, count in users:
            self._log(f"  - {name}: {count} samples")
            
        # Use worker thread for training (can be slow)
        def train_operation():
            success = self.face_recognizer.train_models(data_dir=str(self.face_data_path))
            return success
            
        # Create a custom callback for training completion
        def on_training_complete(success, msg):
            if success:
                self._log("Model training completed successfully!", "success")
                self._log(f"Trained with {len(users)} users: {', '.join(user_names)}", "success")
                
                # Verify models were saved
                models_dir = self.base_path / "models"
                if models_dir.exists():
                    model_files = list(models_dir.glob("*.pkl"))
                    self._log(f"Saved {len(model_files)} model files", "success")
                    
                self._show_success(
                    "Model Training Complete",
                    f"Models trained successfully with {len(users)} users!\n\n"
                    f"Users trained: {', '.join(user_names)}\n\n"
                    f"You can now authenticate using face recognition."
                )
            else:
                self._log("Model training failed!", "error")
                self._show_error(
                    f"Training failed!\n\n"
                    f"Common causes:\n"
                    f"1. Some users have only 1 face sample (need 2+)\n"
                    f"2. Poor quality images\n"
                    f"3. Corrupted image files\n\n"
                    f"Please check logs for details."
                )
        
        # Start training in background
        self.worker = WorkerThread(train_operation)
        self.worker.finished.connect(on_training_complete)
        self.worker.start()
        self._log("Training models in background... (this may take 15-30 seconds)")
        self._show_info("Training Started", "Training models in background.\nThis may take 15-30 seconds.\nGUI will remain responsive.")
        
    def view_system_info(self):
        """Display system information"""
        users_info = []
        for d in self.face_data_path.iterdir():
            if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('_'):
                images = list(d.glob("*.jpg")) + list(d.glob("*.png"))
                users_info.append((d.name, len(images)))
        
        models_exist = (self.base_path / "models").exists()
        
        users_list = ''.join(f'<li>{user} ({count} samples)</li>' for user, count in users_info)
        
        info = f"""
<h3>System Information</h3>
<p><b>Registered Users:</b> {len(users_info)}</p>
<ul>
{users_list if users_info else '<li>No users registered</li>'}
</ul>
<p><b>Models Trained:</b> {'Yes' if models_exist else 'No'}</p>
<p><b>Current User:</b> {self.authenticated_user or 'None'}</p>
<p><b>Min Users for Training:</b> 2 users with 2+ samples each</p>
"""
        
        QMessageBox.information(self.main_window, "System Information", info)
        
    def is_authenticated(self):
        """Check if a user is authenticated"""
        return self.authenticated_user is not None
        
    def get_authenticated_user(self):
        """Get current authenticated username"""
        return self.authenticated_user or "None"
        
    def get_session_info(self):
        """Get session information string"""
        if not self.authenticated_user:
            return "Session: Not Started"
            
        from datetime import datetime
        duration = datetime.now() - self.session_start_time
        minutes, seconds = divmod(duration.seconds, 60)
        
        return (
            f"<b>User:</b> {self.authenticated_user}<br>"
            f"<b>Session Duration:</b> {minutes}m {seconds}s<br>"
            f"<b>Key Loaded:</b> Yes"
        )
        
    def _check_authentication(self):
        """Check if user is authenticated, show error if not"""
        if not self.authenticated_user:
            self._show_error("Please authenticate first")
            return False
        return True
        
    def _generate_user_key(self, username):
        """Generate encryption key for user"""
        key_file = self.face_key_path / f"{username}.key"
        if not key_file.exists():
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            self._log(f"Generated encryption key for {username}")
            
    def _load_user_face_key(self, username):
        """Load user's encryption key into crypto manager"""
        key_file = self.face_key_path / f"{username}.key"
        if key_file.exists():
            key = key_file.read_bytes()
            self.crypto_manager.key = key
            self._log(f"Loaded encryption key for {username}")
        else:
            raise FileNotFoundError(f"Encryption key not found for {username}")
            
    def _run_threaded_operation(self, func, title, success_msg):
        """Run operation in background thread"""
        self.worker = WorkerThread(func)
        self.worker.finished.connect(
            lambda success, msg: self._on_thread_finished(success, msg, title, success_msg)
        )
        self.worker.start()
        self._log(f"Running {title} in background...")
        
    def _on_thread_finished(self, success, msg, title, success_msg):
        """Handle thread completion"""
        if success:
            self._log(success_msg, "success")
            self._show_success(title, success_msg)
        else:
            self._log(msg, "error")
            self._show_error(msg)
            
    def _log(self, message, level="info"):
        """Log message to UI"""
        self.main_window.append_log(message, level)
        
    def _show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self.main_window, "Error", message)
        self.main_window.status_update.emit(message, "error")
        
    def _show_warning(self, message):
        """Show warning message"""
        QMessageBox.warning(self.main_window, "Warning", message)
        self.main_window.status_update.emit(message, "warning")
        
    def _show_success(self, title, message):
        """Show success message"""
        QMessageBox.information(self.main_window, title, message)
        self.main_window.status_update.emit(message, "success")
        
    def _show_info(self, title, message):
        """Show info message"""
        QMessageBox.information(self.main_window, title, message)
        self.main_window.status_update.emit(message, "info")

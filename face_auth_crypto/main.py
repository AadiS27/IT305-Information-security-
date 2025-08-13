"""
Face-based Authentication and Cryptography System
Main application that combines face recognition with file encryption
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.face_capture import FaceCapture
from src.face_recognizer_opencv import FaceRecognizer
from src.crypto_manager import CryptoManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_auth_crypto.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceAuthCryptoSystem:
    def __init__(self):
        """Initialize the face authentication crypto system"""
        self.face_capture = FaceCapture()
        self.face_recognizer = FaceRecognizer()
        self.crypto_manager = CryptoManager()
        
        self.authenticated_user = None
        self.session_start_time = None
        
        logger.info("Face Authentication Crypto System initialized")
    
    def register_user(self):
        """Register a new user by capturing face samples"""
        print("\n=== User Registration ===")
        user_name = input("Enter username for registration: ").strip()
        
        if not user_name:
            print("Invalid username!")
            return False
        
        # Check if user already exists
        existing_users = self.face_recognizer.get_user_list()
        if user_name in existing_users:
            overwrite = input(f"User '{user_name}' already exists. Overwrite? (y/n): ").lower()
            if overwrite != 'y':
                return False
        
        print(f"Registering user: {user_name}")
        print("We will capture multiple face samples for better recognition accuracy.")
        
        num_samples = int(input("Enter number of samples to capture (recommended: 20-30): ") or 20)
        
        # Capture face samples
        if self.face_capture.capture_face_samples(user_name, num_samples):
            print(f"Successfully registered user: {user_name}")
            
            # Retrain models with new user data
            retrain = input("Retrain recognition models now? (y/n): ").lower()
            if retrain == 'y':
                print("Retraining models...")
                if self.face_recognizer.train_models():
                    print("Models retrained successfully!")
                else:
                    print("Failed to retrain models. You can retrain later from the main menu.")
            
            return True
        else:
            print("Failed to register user")
            return False
    
    def authenticate_user(self):
        """Authenticate user using face recognition"""
        print("\n=== Face Authentication ===")
        print("Position your face in front of the camera for authentication.")
        
        # Load models if not already loaded
        if (self.face_recognizer.svm_model is None or 
            self.face_recognizer.rf_model is None or 
            self.face_recognizer.nn_model is None):
            print("Loading recognition models...")
            if not self.face_recognizer.load_models():
                print("Failed to load models. Please train models first.")
                return False
        
        # Capture face for authentication
        face_image = self.face_capture.capture_single_face(timeout=15)
        
        if face_image is None:
            print("Failed to capture face. Authentication failed.")
            return False
        
        print("Analyzing face...")
        
        # Try different recognition methods
        methods = ['ensemble', 'similarity', 'svm', 'rf', 'nn']
        results = []
        
        for method in methods:
            try:
                user_name, confidence = self.face_recognizer.recognize_face(face_image, method=method)
                results.append((method, user_name, confidence))
                print(f"{method.upper()}: {user_name or 'Unknown'} (confidence: {confidence:.4f})")
            except Exception as e:
                print(f"{method.upper()}: Error - {e}")
        
        # Use ensemble result as primary
        ensemble_result = results[0] if results else (None, None, 0.0)
        method, user_name, confidence = ensemble_result
        
        if user_name:
            self.authenticated_user = user_name
            self.session_start_time = datetime.now()
            print(f"\n✓ Authentication successful!")
            print(f"Welcome, {user_name}!")
            print(f"Confidence: {confidence:.4f}")
            return True
        else:
            print(f"\n✗ Authentication failed!")
            print(f"No matching user found (confidence: {confidence:.4f})")
            
            # Offer to adjust threshold
            if confidence > 0:
                adjust = input(f"Current threshold: {self.face_recognizer.recognition_threshold:.2f}. "
                             f"Adjust threshold to {confidence:.4f}? (y/n): ").lower()
                if adjust == 'y':
                    self.face_recognizer.set_recognition_threshold(confidence - 0.01)
                    print("Threshold adjusted. Please try authentication again.")
            
            return False
    
    def train_models(self):
        """Train face recognition models"""
        print("\n=== Training Face Recognition Models ===")
        
        # Check if face data exists
        if not os.path.exists("face_data"):
            print("No face data found. Please register users first.")
            return False
        
        print("Training machine learning models...")
        print("This may take a few minutes depending on the amount of data...")
        
        start_time = time.time()
        
        if self.face_recognizer.train_models():
            end_time = time.time()
            print(f"✓ Models trained successfully in {end_time - start_time:.2f} seconds!")
            
            # Display known users
            users = self.face_recognizer.get_user_list()
            print(f"Known users: {', '.join(users)}")
            
            return True
        else:
            print("✗ Failed to train models")
            return False
    
    def encrypt_file(self):
        """Encrypt a file"""
        if not self.authenticated_user:
            print("Please authenticate first!")
            return False
        
        print(f"\n=== File Encryption (User: {self.authenticated_user}) ===")
        
        file_path = input("Enter path to file to encrypt: ").strip()
        
        if not os.path.exists(file_path):
            print("File not found!")
            return False
        
        print("Encryption options:")
        print("1. Standard encryption")
        print("2. Face-authenticated encryption (requires face auth to decrypt)")
        
        choice = input("Choose option (1-2): ").strip()
        
        if choice == "1":
            if self.crypto_manager.encrypt_file(file_path):
                print(f"✓ File encrypted: {file_path}.encrypted")
                return True
            else:
                print("✗ Encryption failed")
                return False
                
        elif choice == "2":
            if self.crypto_manager.encrypt_with_face_auth(file_path, self.authenticated_user, self.face_recognizer):
                encrypted_file = file_path + f".{self.authenticated_user}.face_encrypted"
                print(f"✓ File encrypted with face authentication: {encrypted_file}")
                print(f"Only {self.authenticated_user} can decrypt this file")
                return True
            else:
                print("✗ Face-authenticated encryption failed")
                return False
        else:
            print("Invalid choice!")
            return False
    
    def decrypt_file(self):
        """Decrypt a file"""
        if not self.authenticated_user:
            print("Please authenticate first!")
            return False
        
        print(f"\n=== File Decryption (User: {self.authenticated_user}) ===")
        
        file_path = input("Enter path to encrypted file: ").strip()
        
        if not os.path.exists(file_path):
            print("File not found!")
            return False
        
        # Determine encryption type
        if ".face_encrypted" in file_path:
            # Face-authenticated encryption
            if self.crypto_manager.decrypt_with_face_auth(file_path, self.authenticated_user):
                print("✓ File decrypted successfully!")
                return True
            else:
                print("✗ Decryption failed - Access denied or corrupted file")
                return False
        else:
            # Standard encryption
            if self.crypto_manager.decrypt_file(file_path):
                output_file = file_path.replace('.encrypted', '') if file_path.endswith('.encrypted') else file_path + '.decrypted'
                print(f"✓ File decrypted: {output_file}")
                return True
            else:
                print("✗ Decryption failed")
                return False
    
    def encrypt_folder(self):
        """Encrypt an entire folder"""
        if not self.authenticated_user:
            print("Please authenticate first!")
            return False
        
        print(f"\n=== Folder Encryption (User: {self.authenticated_user}) ===")
        
        folder_path = input("Enter path to folder to encrypt: ").strip()
        
        if not os.path.exists(folder_path):
            print("Folder not found!")
            return False
        
        if not os.path.isdir(folder_path):
            print("Path is not a directory!")
            return False
        
        print("Encrypting folder...")
        if self.crypto_manager.create_encrypted_folder(folder_path):
            print(f"✓ Folder encrypted: {folder_path}_encrypted")
            return True
        else:
            print("✗ Folder encryption failed")
            return False
    
    def decrypt_folder(self):
        """Decrypt an entire folder"""
        if not self.authenticated_user:
            print("Please authenticate first!")
            return False
        
        print(f"\n=== Folder Decryption (User: {self.authenticated_user}) ===")
        
        folder_path = input("Enter path to encrypted folder: ").strip()
        
        if not os.path.exists(folder_path):
            print("Folder not found!")
            return False
        
        print("Decrypting folder...")
        if self.crypto_manager.decrypt_folder(folder_path):
            output_folder = folder_path.replace('_encrypted', '') if folder_path.endswith('_encrypted') else folder_path + '_decrypted'
            print(f"✓ Folder decrypted: {output_folder}")
            return True
        else:
            print("✗ Folder decryption failed")
            return False
    
    def view_system_info(self):
        """Display system information"""
        print("\n=== System Information ===")
        
        # User info
        if self.authenticated_user:
            print(f"Authenticated User: {self.authenticated_user}")
            print(f"Session Duration: {datetime.now() - self.session_start_time}")
        else:
            print("Authentication Status: Not authenticated")
        
        # Known users
        users = self.face_recognizer.get_user_list()
        print(f"Registered Users: {', '.join(users) if users else 'None'}")
        
        # Recognition threshold
        print(f"Recognition Threshold: {self.face_recognizer.recognition_threshold:.3f}")
        
        # Model status
        models_loaded = (
            self.face_recognizer.svm_model is not None and
            self.face_recognizer.rf_model is not None and
            self.face_recognizer.nn_model is not None
        )
        print(f"Models Status: {'Loaded' if models_loaded else 'Not loaded'}")
        
        # Data directories
        print(f"Face Data Directory: {'face_data' if os.path.exists('face_data') else 'Not found'}")
        print(f"Models Directory: {'models' if os.path.exists('models') else 'Not found'}")
        
        # System files
        print(f"Crypto Key File: {'crypto.key' if os.path.exists('crypto.key') else 'Not found'}")
        print(f"Log File: {'face_auth_crypto.log' if os.path.exists('face_auth_crypto.log') else 'Not found'}")
    
    def adjust_settings(self):
        """Adjust system settings"""
        print("\n=== System Settings ===")
        
        print("1. Adjust recognition threshold")
        print("2. Test camera")
        print("3. View face data statistics")
        print("4. Clean up old files")
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice == "1":
            current_threshold = self.face_recognizer.recognition_threshold
            print(f"Current recognition threshold: {current_threshold:.3f}")
            print("Lower threshold = more lenient (may allow wrong users)")
            print("Higher threshold = more strict (may reject correct users)")
            
            try:
                new_threshold = float(input("Enter new threshold (0.0-1.0): "))
                if 0.0 <= new_threshold <= 1.0:
                    self.face_recognizer.set_recognition_threshold(new_threshold)
                    print(f"Threshold updated to {new_threshold:.3f}")
                else:
                    print("Threshold must be between 0.0 and 1.0")
            except ValueError:
                print("Invalid threshold value")
        
        elif choice == "2":
            print("Starting camera test...")
            self.face_capture.preview_camera()
        
        elif choice == "3":
            if os.path.exists("face_data"):
                print("\nFace Data Statistics:")
                for user_dir in os.listdir("face_data"):
                    user_path = os.path.join("face_data", user_dir)
                    if os.path.isdir(user_path):
                        image_count = len([f for f in os.listdir(user_path) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        print(f"  {user_dir}: {image_count} images")
            else:
                print("No face data directory found")
        
        elif choice == "4":
            print("Cleaning up temporary files...")
            # Clean up any temporary files
            for file in os.listdir("."):
                if file.endswith((".tmp", ".temp")) or file.startswith("temp_"):
                    try:
                        os.remove(file)
                        print(f"Removed: {file}")
                    except:
                        pass
            print("Cleanup completed")
    
    def logout(self):
        """Logout current user"""
        if self.authenticated_user:
            session_duration = datetime.now() - self.session_start_time
            print(f"\nLogging out {self.authenticated_user}")
            print(f"Session duration: {session_duration}")
            
            self.authenticated_user = None
            self.session_start_time = None
            print("Logged out successfully")
        else:
            print("No user currently authenticated")
    
    def run(self):
        """Main application loop"""
        print("="*60)
        print("    FACE-BASED AUTHENTICATION & CRYPTOGRAPHY SYSTEM")
        print("="*60)
        print("Welcome to the secure face authentication system!")
        print("This system uses advanced face recognition to protect your files.")
        
        while True:
            print(f"\n{'='*40}")
            if self.authenticated_user:
                print(f"Authenticated as: {self.authenticated_user}")
            else:
                print("Status: Not authenticated")
            print(f"{'='*40}")
            
            print("\n1.  Register new user")
            print("2.  Authenticate user")
            print("3.  Train recognition models")
            print("4.  Encrypt file")
            print("5.  Decrypt file")
            print("6.  Encrypt folder")
            print("7.  Decrypt folder")
            print("8.  View system information")
            print("9.  Adjust settings")
            print("10. Logout")
            print("11. Exit")
            
            choice = input("\nEnter your choice (1-11): ").strip()
            
            try:
                if choice == "1":
                    self.register_user()
                elif choice == "2":
                    self.authenticate_user()
                elif choice == "3":
                    self.train_models()
                elif choice == "4":
                    self.encrypt_file()
                elif choice == "5":
                    self.decrypt_file()
                elif choice == "6":
                    self.encrypt_folder()
                elif choice == "7":
                    self.decrypt_folder()
                elif choice == "8":
                    self.view_system_info()
                elif choice == "9":
                    self.adjust_settings()
                elif choice == "10":
                    self.logout()
                elif choice == "11":
                    self.logout()
                    print("\nThank you for using Face-based Authentication & Cryptography System!")
                    print("Stay secure! 🔒")
                    break
                else:
                    print("Invalid choice! Please enter a number between 1-11.")
                    
            except KeyboardInterrupt:
                print("\n\nOperation interrupted by user")
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"An error occurred: {e}")
                print("Please try again or contact support.")

if __name__ == "__main__":
    try:
        app = FaceAuthCryptoSystem()
        app.run()
    except KeyboardInterrupt:
        print("\n\nApplication terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        print("Please check the logs for more details.")

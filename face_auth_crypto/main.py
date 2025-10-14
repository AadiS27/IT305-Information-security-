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
    def _load_user_face_key(self, user_name):
        """Load the user's face key from face_key/{username}.key and set it in CryptoManager."""
        face_key_path = os.path.join("face_key", f"{user_name}.key")
        if not os.path.exists(face_key_path):
            print(f"Face key not found for user {user_name} at {face_key_path}")
            return False
        try:
            with open(face_key_path, 'rb') as f:
                key = f.read()
            self.crypto_manager.key = key
            from cryptography.fernet import Fernet
            self.crypto_manager.fernet = Fernet(key)
            logger.info(f"Loaded face key for user {user_name} from {face_key_path}")
            return True
        except Exception as e:
            print(f"Failed to load face key for user {user_name}: {e}")
            logger.error(f"Failed to load face key for user {user_name}: {e}")
            return False
    def __init__(self):
        """Initialize the face authentication crypto system"""
        self.face_capture = FaceCapture()
        self.face_recognizer = FaceRecognizer()
        self.crypto_manager = CryptoManager()
        
        self.authenticated_user = None
        self.session_start_time = None
        
        logger.info("Face Authentication Crypto System initialized")
    
    def _generate_user_key(self, user_name):
        """Generate a unique key for the authenticated user in face_key folder"""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Create face_key directory if it doesn't exist
            face_key_dir = "face_key"
            os.makedirs(face_key_dir, exist_ok=True)
            
            # Generate a unique key for this user
            user_key = Fernet.generate_key()
            
            # Save the key to face_key folder with user's name
            key_file_path = os.path.join(face_key_dir, f"{user_name}.key")
            
            with open(key_file_path, 'wb') as key_file:
                key_file.write(user_key)
            
            print(f"✓ User key generated: {key_file_path}")
            logger.info(f"Generated key for user {user_name} at {key_file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating user key: {e}")
            print(f"⚠ Warning: Could not generate user key - {e}")
            return False
    
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
        
        # Check if any users are registered
        registered_users = self.face_recognizer.get_registered_users()
        if not registered_users:
            print("No registered users found!")
            print("Please register users first using option 1.")
            return False
        
        print(f"Number of registered users: {len(registered_users)}")
        print("Position your face in front of the camera for authentication.")
        
        # Load models if not already loaded
        if (self.face_recognizer.svm_model is None or 
            self.face_recognizer.rf_model is None or 
            self.face_recognizer.nn_model is None):
            print("Loading recognition models...")
            if not self.face_recognizer.load_models():
                print("Failed to load models. Models may not be trained yet.")
                print("Please train models first using option 3.")
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
            # Generate user-specific key in face_key folder (if not already present)
            self._generate_user_key(user_name)
            # Load the user's face key for all encryption/decryption
            self._load_user_face_key(user_name)
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
        
        # Check for registered users
        registered_users = self.face_recognizer.get_registered_users()
        if not registered_users:
            print("No registered users found in face_data directory.")
            print("Please register users first using option 1.")
            return False
        
        # Check if models already exist and are up to date
        if self.face_recognizer._models_exist():
            print(f"Found existing trained models for users: {', '.join(registered_users)}")
            choice = input("Do you want to retrain the models? (y/n): ").strip().lower()
            if choice != 'y':
                print("Using existing models.")
                return True
        
        print("Training machine learning models...")
        print("This may take a few minutes depending on the amount of data...")
        
        start_time = time.time()
        
        if self.face_recognizer.train_models():
            end_time = time.time()
            training_time = end_time - start_time
            print(f"✓ Models trained successfully in {training_time:.2f} seconds!")
            
            # Display known users
            users = self.face_recognizer.get_user_list()
            print(f"Known users: {', '.join(users)}")
            
            # Display additional model performance information for presentation
            print("\n=== MODEL PERFORMANCE METRICS FOR PRESENTATION ===")
            print(f"Total training time: {training_time:.2f} seconds")
            
            # Show samples per user for presentation
            print("\nSamples per user:")
            total_samples = 0
            for user in users:
                user_dir = os.path.join("face_data", user)
                if os.path.isdir(user_dir):
                    image_count = len([f for f in os.listdir(user_dir) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  - {user}: {image_count} samples")
                    total_samples += image_count
            
            print(f"\nTotal training samples: {total_samples}")
            
            # Model-specific metrics (if available)
            if hasattr(self.face_recognizer, 'svm_accuracy'):
                print("\nModel Accuracy:")
                print(f"  - SVM: {self.face_recognizer.svm_accuracy:.4f}")
                print(f"  - Random Forest: {self.face_recognizer.rf_accuracy:.4f}")
                print(f"  - Neural Network: {self.face_recognizer.nn_accuracy:.4f}")
                print(f"  - Ensemble (weighted): {(self.face_recognizer.svm_accuracy*0.4 + self.face_recognizer.rf_accuracy*0.4 + self.face_recognizer.nn_accuracy*0.2):.4f}")
            
            # Display confusion matrices
            if hasattr(self.face_recognizer, 'svm_confusion_matrix') and self.face_recognizer.svm_confusion_matrix is not None:
                print("\n=== CONFUSION MATRICES ===")
                print("Confusion matrices show how well each model classifies different users.")
                print("Rows represent actual users, columns represent predicted users.")
                print("Diagonal values (top-left to bottom-right) show correct predictions.")
                
                # Get user names for better display
                label_map = {i: name for i, name in enumerate(self.face_recognizer.label_encoder.classes_)}
                
                # Display confusion matrices
                models = [
                    ("SVM Model", self.face_recognizer.svm_confusion_matrix),
                    ("Random Forest Model", self.face_recognizer.rf_confusion_matrix),
                    ("Neural Network Model", self.face_recognizer.nn_confusion_matrix)
                ]
                
                for model_name, conf_matrix in models:
                    if conf_matrix is not None:
                        print(f"\n{model_name} Confusion Matrix:")
                        # Create header with user names
                        header = "      " + " ".join([f"{label_map[i][:7]:8}" for i in range(len(label_map))])
                        print(header)
                        
                        # Print matrix with row labels
                        for i, row in enumerate(conf_matrix):
                            row_str = f"{label_map[i][:7]:7}"
                            for val in row:
                                row_str += f"{val:8}"
                            print(row_str)
            
            # Additional statistics for presentation
            print("\nSystem Statistics:")
            print(f"  - Face detection algorithm: Haar Cascade Classifier")
            print(f"  - Feature extraction: 128-dimension face encodings")
            print(f"  - Recognition threshold: {self.face_recognizer.recognition_threshold:.3f}")
            print(f"  - Model file size: {os.path.getsize('models/svm_model.pkl')/1024:.1f} KB (SVM), {os.path.getsize('models/rf_model.pkl')/1024:.1f} KB (RF), {os.path.getsize('models/nn_model.pkl')/1024:.1f} KB (NN)")
            
            # Max Voting System Description
            print("\n=== MAX VOTING SYSTEM DETAILS ===")
            print("The system uses ensemble learning with a max voting classifier:")
            print("\n1. Ensemble Learning with Max Voting:")
            print("   • Multiple models vote for the final classification decision")
            print("   • Each model independently predicts the user identity")
            print("   • The class with the maximum votes becomes the final prediction")
            print("   • In case of a tie, models with higher accuracy get precedence")
            
            print("\n2. Models in the Ensemble:")
            print("   • SVM Model:")
            print("     - Strengths: High accuracy with clear separation between classes")
            print("     - Use case: Creates optimal decision boundaries in high-dimensional face space")
            print(f"     - Current accuracy: {self.face_recognizer.svm_accuracy:.2%}")
            
            print("\n   • Random Forest Model:")
            print("     - Strengths: Robust to noise and outliers, handles non-linear patterns")
            print("     - Use case: Performs well with varied lighting conditions and minor face occlusions")
            print(f"     - Current accuracy: {self.face_recognizer.rf_accuracy:.2%}")
            
            print("\n   • Neural Network Model:")
            print("     - Strengths: Learns complex patterns, adapts to subtle differences")
            print("     - Use case: Captures nuanced facial features that other models might miss")
            print(f"     - Current accuracy: {self.face_recognizer.nn_accuracy:.2%}")
            
            print("\n3. Voting Process:")
            print("   • Each model predicts a class label (user identity)")
            print("   • Max Voting tallies all predictions")
            print("   • The class with the highest number of votes wins")
            print("   • Example: If SVM predicts User1, RF predicts User1, and NN predicts User2,")
            print("     the final result would be User1 (2 votes vs. 1 vote)")
            
            print("\n4. Confidence Calculation:")
            print("   • After determining the winning class by votes,")
            print("   • The system calculates a confidence score based on the models' certainty")
            print("   • Higher confidence scores indicate more reliable predictions")
            print(f"   • Recognition threshold: {self.face_recognizer.recognition_threshold:.2f}")
            print("   • If confidence < threshold, authentication is rejected")
            
            print("\n5. Benefits of Max Voting:")
            print("   • Higher accuracy: Combined knowledge of multiple algorithms")
            print("   • Reduced false positives: Models must agree for a strong prediction")
            print("   • Robustness: Less affected by individual model weaknesses")
            print("   • Generalization: Better performance on new, unseen faces")
            print("   • Reduced overfitting: Multiple models compensate for each other's biases")
            print("===================================================")
            
            print("\nRecommended next steps:")
            print("  1. Test authentication with different lighting conditions")
            print("  2. Try authentication with different facial expressions")
            print("  3. Measure authentication speed for real-time performance")
            print("===================================================")
            
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
        
        print("⚠️  WARNING: After encryption, the original file will be deleted for security!")
        print("Only the encrypted file will remain.")
        confirm = input("Do you want to continue? (y/n): ").lower()
        if confirm != 'y':
            print("Encryption cancelled.")
            return False
        
        print("\nEncryption options:")
        print("1. Standard encryption")
        print("2. Face-authenticated encryption (requires face auth to decrypt)")
        
        choice = input("Choose option (1-2): ").strip()
        
        if choice == "1":
            # Always use the authenticated user's face key for standard encryption
            if not self._load_user_face_key(self.authenticated_user):
                print("Cannot encrypt: face key not loaded.")
                return False
            if self.crypto_manager.encrypt_file(file_path, delete_original=True):
                print(f"✓ File encrypted: {file_path}.encrypted")
                print(f"✓ Original file deleted: {file_path}")
                return True
            else:
                print("✗ Encryption failed")
                return False
                
        elif choice == "2":
            if self.crypto_manager.encrypt_with_face_auth(file_path, self.authenticated_user, self.face_recognizer, delete_original=True):
                encrypted_file = file_path + f".{self.authenticated_user}.face_encrypted"
                print(f"✓ File encrypted with face authentication: {encrypted_file}")
                print(f"✓ Original file deleted: {file_path}")
                print(f"Only {self.authenticated_user} can decrypt this file")
                return True
            else:
                print("✗ Face-authenticated encryption failed")
                return False
        else:
            print("Invalid choice!")
            return False
    
    def decrypt_file(self):
        """Decrypt a file (face-authenticated: require live face capture)"""
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
            print("Show your face to the camera for decryption...")
            
            # Try multiple attempts to capture face
            max_attempts = 3
            face_image = None
            
            for attempt in range(max_attempts):
                print(f"Attempt {attempt + 1}/{max_attempts}")
                face_image = self.face_capture.capture_single_face(timeout=20)
                
                if face_image is not None:
                    print("Face captured successfully!")
                    break
                else:
                    print(f"Failed to capture face on attempt {attempt + 1}")
                    if attempt < max_attempts - 1:
                        retry = input("Try again? (y/n): ").lower()
                        if retry != 'y':
                            break
            
            if face_image is None:
                print("Failed to capture face after all attempts. Decryption aborted.")
                return False
            
            # Try different approaches to extract face features
            live_encoding = None
            
            # First, try direct feature extraction
            print("Extracting face features from captured image...")
            live_encoding = self.face_recognizer.extract_face_features(face_image)
            
            if live_encoding is None:
                print("Direct feature extraction failed. Trying with enhanced detection...")
                # Save the captured face temporarily and try again
                temp_path = "temp_face_for_decryption.jpg"
                try:
                    import cv2
                    cv2.imwrite(temp_path, face_image)
                    live_encoding = self.face_recognizer.extract_face_features(temp_path)
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Enhanced detection also failed: {e}")
            
            if live_encoding is None:
                print("Could not extract face features with any method.")
                print("This might be due to:")
                print("- Poor lighting conditions")
                print("- Face not clearly visible")
                print("- Camera angle or distance issues")
                fallback = input("Try using stored face data instead? (y/n): ").lower()
                
                if fallback == 'y':
                    # Use stored face data as fallback
                    user_encoding = self.face_recognizer.get_user_encoding(self.authenticated_user)
                    if user_encoding is not None:
                        print("Using stored face data for decryption...")
                        if self.crypto_manager.decrypt_with_face_auth(file_path, self.authenticated_user, self.face_recognizer, face_encoding=user_encoding):
                            print("✓ File decrypted successfully using stored face data!")
                            return True
                        else:
                            print("✗ Decryption failed - Access denied or corrupted file")
                            return False
                    else:
                        print("No stored face data found for user.")
                        return False
                else:
                    return False
            
            # Compare to all stored encodings for the user
            import numpy as np
            user_dir = os.path.join("face_data", self.authenticated_user)
            if not os.path.exists(user_dir):
                print(f"No face data found for user {self.authenticated_user}")
                return False
            
            best_dist = float('inf')
            best_encoding = None
            threshold = 0.6  # Increased threshold for more lenient matching
            valid_encodings = 0
            
            print("Comparing with stored face encodings...")
            for img_file in os.listdir(user_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(user_dir, img_file)
                    stored_encoding = self.face_recognizer.extract_face_features(img_path)
                    if stored_encoding is not None:
                        valid_encodings += 1
                        dist = np.linalg.norm(live_encoding - stored_encoding)
                        print(f"  Distance to {img_file}: {dist:.3f}")
                        if dist < best_dist:
                            best_dist = dist
                            best_encoding = stored_encoding
            
            print(f"Found {valid_encodings} valid stored encodings")
            print(f"Best match distance: {best_dist:.3f} (threshold: {threshold})")
            
            if best_encoding is not None and best_dist < threshold:
                print(f"Face match confirmed! Attempting decryption...")
                if self.crypto_manager.decrypt_with_face_auth(file_path, self.authenticated_user, self.face_recognizer, face_encoding=best_encoding):
                    print("✓ File decrypted successfully!")
                    return True
                else:
                    print("✗ Decryption failed - Access denied or corrupted file")
                    return False
            else:
                print(f"Face verification failed. Distance {best_dist:.3f} exceeds threshold {threshold}")
                adjust_threshold = input(f"Adjust threshold to {best_dist + 0.05:.3f} and try again? (y/n): ").lower()
                if adjust_threshold == 'y':
                    threshold = best_dist + 0.05
                    if best_encoding is not None and best_dist < threshold:
                        print(f"Trying with adjusted threshold {threshold:.3f}...")
                        if self.crypto_manager.decrypt_with_face_auth(file_path, self.authenticated_user, self.face_recognizer, face_encoding=best_encoding):
                            print("✓ File decrypted successfully!")
                            return True
                        else:
                            print("✗ Decryption failed - Access denied or corrupted file")
                            return False
                
                print("Decryption aborted - face verification failed.")
                return False
        else:
            # Standard encryption: use the authenticated user's face key
            if not self._load_user_face_key(self.authenticated_user):
                print("Cannot decrypt: face key not loaded.")
                return False
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
        
        print("⚠️  WARNING: After encryption, the original folder will be deleted for security!")
        print("Only the encrypted folder will remain.")
        confirm = input("Do you want to continue? (y/n): ").lower()
        if confirm != 'y':
            print("Folder encryption cancelled.")
            return False
        
        print("Encrypting folder...")
        if self.crypto_manager.create_encrypted_folder(folder_path, delete_original=True):
            print(f"✓ Folder encrypted: {folder_path}_encrypted")
            print(f"✓ Original folder deleted: {folder_path}")
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
        
        # Show system status on startup
        print(f"\n{'='*50}")
        print("SYSTEM STATUS")
        print(f"{'='*50}")
        
        # Check for registered users
        registered_users = self.face_recognizer.get_registered_users()
        if registered_users:
            print(f"✓ {len(registered_users)} registered user(s) found")
        else:
            print("⚠ No registered users found")
        
        # Check for trained models
        if self.face_recognizer._models_exist():
            print("✓ Trained models loaded successfully")
        else:
            print("⚠ No trained models found - training required")
        
        print(f"{'='*50}")
        
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
                    print("Stay secure! ")
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

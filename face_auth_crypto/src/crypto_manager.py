"""
Crypto Manager for Face-based Authentication System
Handles file encryption and decryption using AES
"""

import os
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoManager:
    def _derive_key_from_face_encoding(self, face_encoding):
        """Derive a 32-byte AES key from a face encoding (numpy array) using SHA-256."""
        import hashlib
        if hasattr(face_encoding, 'tobytes'):
            encoding_bytes = face_encoding.tobytes()
        else:
            encoding_bytes = bytes(face_encoding)
        key = hashlib.sha256(encoding_bytes).digest()  # 32 bytes for AES-256
        return base64.urlsafe_b64encode(key)
    def __init__(self):
        """Initialize crypto manager (no default key file)"""
        self.key = None
        self.fernet = None
    
    # generate_key and load_key methods removed (no longer needed)
    
    def derive_key_from_password(self, password, salt=None):
        """Derive encryption key from password"""
        try:
            if salt is None:
                salt = os.urandom(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            self.key = key
            self.fernet = Fernet(key)
            
            return salt
            
        except Exception as e:
            logger.error(f"Error deriving key from password: {e}")
            return None
    
    def encrypt_text(self, text):
        """Encrypt text data"""
        try:
            if self.fernet is None:
                logger.error("No encryption key available")
                return None
            
            encrypted_data = self.fernet.encrypt(text.encode())
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting text: {e}")
            return None
    
    def decrypt_text(self, encrypted_data):
        """Decrypt text data"""
        try:
            if self.fernet is None:
                logger.error("No encryption key available")
                return None
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            return decrypted_data.decode()
            
        except Exception as e:
            logger.error(f"Error decrypting text: {e}")
            return None
    
    def encrypt_file(self, input_file, output_file=None, delete_original=True):
        """Encrypt a file and optionally delete the original"""
        try:
            if self.fernet is None:
                logger.error("No encryption key available")
                return False
            
            if output_file is None:
                output_file = input_file + ".encrypted"
            
            # Read original file
            with open(input_file, 'rb') as f:
                file_data = f.read()
            
            # Encrypt data
            encrypted_data = self.fernet.encrypt(file_data)
            
            # Write encrypted file
            with open(output_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Delete original file if requested
            if delete_original:
                try:
                    os.remove(input_file)
                    logger.info(f"Original file deleted: {input_file}")
                except Exception as e:
                    logger.warning(f"Could not delete original file {input_file}: {e}")
            
            logger.info(f"File encrypted: {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error encrypting file: {e}")
            return False
    
    def decrypt_file(self, input_file, output_file=None):
        """Decrypt a file"""
        try:
            if self.fernet is None:
                logger.error("No encryption key available")
                return False
            
            if output_file is None:
                # Remove .encrypted extension if present
                if input_file.endswith('.encrypted'):
                    output_file = input_file[:-10]  # Remove '.encrypted'
                else:
                    output_file = input_file + ".decrypted"
            
            # Read encrypted file
            with open(input_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt data
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            # Write decrypted file
            with open(output_file, 'wb') as f:
                f.write(decrypted_data)
            
            logger.info(f"File decrypted: {input_file} -> {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error decrypting file: {e}")
            return False
    
    def secure_delete_file(self, filepath):
        """Securely delete a file by overwriting it"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"File does not exist: {filepath}")
                return False
            
            # Get file size
            filesize = os.path.getsize(filepath)
            
            # Overwrite file with random data multiple times
            with open(filepath, 'rb+') as f:
                for _ in range(3):  # 3 passes
                    f.seek(0)
                    f.write(os.urandom(filesize))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Delete the file
            os.remove(filepath)
            logger.info(f"File securely deleted: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error securely deleting file: {e}")
            return False
    
    def create_encrypted_folder(self, folder_path, delete_original=True):
        """Create an encrypted folder structure and optionally delete original"""
        try:
            encrypted_folder = folder_path + "_encrypted"
            os.makedirs(encrypted_folder, exist_ok=True)
            
            # Encrypt all files in the folder
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Create relative path structure in encrypted folder
                    rel_path = os.path.relpath(file_path, folder_path)
                    encrypted_file_path = os.path.join(encrypted_folder, rel_path + ".encrypted")
                    
                    # Create directories if needed
                    os.makedirs(os.path.dirname(encrypted_file_path), exist_ok=True)
                    
                    # Encrypt file (without deleting yet - we'll delete the whole folder later)
                    if not self.encrypt_file(file_path, encrypted_file_path, delete_original=False):
                        logger.error(f"Failed to encrypt {file_path}")
                        return False
            
            # Delete original folder if requested and encryption was successful
            if delete_original:
                try:
                    import shutil
                    shutil.rmtree(folder_path)
                    logger.info(f"Original folder deleted: {folder_path}")
                except Exception as e:
                    logger.warning(f"Could not delete original folder {folder_path}: {e}")
            
            logger.info(f"Folder encrypted: {folder_path} -> {encrypted_folder}")
            return True
            
        except Exception as e:
            logger.error(f"Error encrypting folder: {e}")
            return False
    
    def decrypt_folder(self, encrypted_folder_path, output_folder=None):
        """Decrypt an encrypted folder structure"""
        try:
            if output_folder is None:
                if encrypted_folder_path.endswith('_encrypted'):
                    output_folder = encrypted_folder_path[:-10]  # Remove '_encrypted'
                else:
                    output_folder = encrypted_folder_path + "_decrypted"
            
            os.makedirs(output_folder, exist_ok=True)
            
            # Decrypt all files in the encrypted folder
            for root, dirs, files in os.walk(encrypted_folder_path):
                for file in files:
                    if file.endswith('.encrypted'):
                        encrypted_file_path = os.path.join(root, file)
                        
                        # Create relative path structure in output folder
                        rel_path = os.path.relpath(encrypted_file_path, encrypted_folder_path)
                        rel_path = rel_path[:-10]  # Remove '.encrypted'
                        decrypted_file_path = os.path.join(output_folder, rel_path)
                        
                        # Create directories if needed
                        os.makedirs(os.path.dirname(decrypted_file_path), exist_ok=True)
                        
                        # Decrypt file
                        if not self.decrypt_file(encrypted_file_path, decrypted_file_path):
                            logger.error(f"Failed to decrypt {encrypted_file_path}")
                            return False
            
            logger.info(f"Folder decrypted: {encrypted_folder_path} -> {output_folder}")
            return True
            
        except Exception as e:
            logger.error(f"Error decrypting folder: {e}")
            return False
    
    def encrypt_with_face_auth(self, input_file, user_name, face_recognizer, delete_original=True):
        """Encrypt file with face authentication requirement (face-derived key) and optionally delete original"""
        try:
            # Get face encoding for the user
            face_encoding = face_recognizer.get_user_encoding(user_name)
            if face_encoding is None:
                logger.error(f"No face encoding found for user {user_name}")
                return False
            key = self._derive_key_from_face_encoding(face_encoding)
            fernet = Fernet(key)
            # Encrypt the file
            with open(input_file, 'rb') as f:
                file_data = f.read()
            encrypted_data = fernet.encrypt(file_data)
            encrypted_file = input_file + f".{user_name}.face_encrypted"
            with open(encrypted_file, 'wb') as f:
                f.write(encrypted_data)
            # Store a hash of the face-derived key in face_keys/<username>.keyhash
            import hashlib
            key_hash = hashlib.sha256(key).digest()
            import base64
            key_hash_b64 = base64.urlsafe_b64encode(key_hash).decode('utf-8')
            key_dir = os.path.join(os.path.dirname(__file__), '..', 'face_keys')
            os.makedirs(key_dir, exist_ok=True)
            key_file_path = os.path.join(key_dir, f"{user_name}.keyhash")
            with open(key_file_path, 'w') as f:
                f.write(key_hash_b64)
            # Create metadata
            metadata = {
                'encrypted_by': user_name,
                'encryption_method': 'face_auth',
                'original_filename': os.path.basename(input_file)
            }
            metadata_file = encrypted_file + ".meta"
            metadata_text = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            with open(metadata_file, 'w') as f:
                f.write(metadata_text)
            
            # Delete original file if requested
            if delete_original:
                try:
                    os.remove(input_file)
                    logger.info(f"Original file deleted: {input_file}")
                except Exception as e:
                    logger.warning(f"Could not delete original file {input_file}: {e}")
            
            logger.info(f"File encrypted with face authentication: {encrypted_file}")
            logger.info(f"Face key hash stored at: {key_file_path}")
            return True
        except Exception as e:
            logger.error(f"Error encrypting with face auth: {e}")
            return False
    
    def decrypt_with_face_auth(self, encrypted_file, authenticated_user, face_recognizer, face_encoding=None):
        """Decrypt file with face authentication verification (face-derived key, supports live encoding)"""
        try:
            # Check metadata
            metadata_file = encrypted_file + ".meta"
            if not os.path.exists(metadata_file):
                logger.error("Metadata file not found")
                return False
            with open(metadata_file, 'r') as f:
                metadata_content = f.read()
            # Parse metadata
            metadata = {}
            for line in metadata_content.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
            # Verify user authorization
            if 'encrypted_by' in metadata:
                authorized_user = metadata['encrypted_by']
                if authenticated_user != authorized_user:
                    logger.error(f"User {authenticated_user} not authorized to decrypt file encrypted by {authorized_user}")
                    return False
            # Use provided live encoding if available, else fallback to stored encoding
            if face_encoding is None:
                face_encoding = face_recognizer.get_user_encoding(authenticated_user)
            if face_encoding is None:
                logger.error(f"No face encoding found for user {authenticated_user}")
                return False
            key = self._derive_key_from_face_encoding(face_encoding)
            fernet = Fernet(key)
            # Decrypt the file
            original_name = metadata.get('original_filename', 'decrypted_file')
            output_file = os.path.join(os.path.dirname(encrypted_file), original_name)
            with open(encrypted_file, 'rb') as f:
                encrypted_data = f.read()
            try:
                decrypted_data = fernet.decrypt(encrypted_data)
            except Exception as e:
                logger.error(f"Decryption failed: {e}")
                return False
            with open(output_file, 'wb') as f:
                f.write(decrypted_data)
            logger.info(f"File decrypted with face authentication: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error decrypting with face auth: {e}")
            return False

if __name__ == "__main__":
    # Test the crypto manager
    crypto = CryptoManager()
    
    # Test text encryption
    test_text = "This is a test message for face-based authentication!"
    print(f"Original text: {test_text}")
    
    encrypted = crypto.encrypt_text(test_text)
    if encrypted:
        print(f"Encrypted: {encrypted}")
        
        decrypted = crypto.decrypt_text(encrypted)
        if decrypted:
            print(f"Decrypted: {decrypted}")
            print(f"Success: {test_text == decrypted}")
    
    # Test file encryption (create a test file first)
    test_file = "test.txt"
    with open(test_file, 'w') as f:
        f.write("This is a test file for encryption!")
    
    print(f"\nTesting file encryption...")
    if crypto.encrypt_file(test_file):
        print("File encrypted successfully")
        
        if crypto.decrypt_file(test_file + ".encrypted"):
            print("File decrypted successfully")
    
    # Clean up
    for file in [test_file, test_file + ".encrypted", test_file + ".encrypted.decrypted"]:
        if os.path.exists(file):
            os.remove(file)

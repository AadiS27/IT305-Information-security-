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
    def __init__(self, key_file="crypto.key"):
        """Initialize crypto manager"""
        self.key_file = key_file
        self.key = None
        self.fernet = None
        
        # Try to load existing key, create new one if doesn't exist
        if os.path.exists(key_file):
            self.load_key()
        else:
            self.generate_key()
    
    def generate_key(self):
        """Generate a new encryption key"""
        try:
            self.key = Fernet.generate_key()
            self.fernet = Fernet(self.key)
            
            # Save key to file
            with open(self.key_file, 'wb') as f:
                f.write(self.key)
            
            logger.info("New encryption key generated and saved")
            return True
            
        except Exception as e:
            logger.error(f"Error generating key: {e}")
            return False
    
    def load_key(self):
        """Load encryption key from file"""
        try:
            with open(self.key_file, 'rb') as f:
                self.key = f.read()
            
            self.fernet = Fernet(self.key)
            logger.info("Encryption key loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading key: {e}")
            return False
    
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
    
    def encrypt_file(self, input_file, output_file=None):
        """Encrypt a file"""
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
    
    def create_encrypted_folder(self, folder_path):
        """Create an encrypted folder structure"""
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
                    
                    # Encrypt file
                    if not self.encrypt_file(file_path, encrypted_file_path):
                        logger.error(f"Failed to encrypt {file_path}")
                        return False
            
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
    
    def encrypt_with_face_auth(self, input_file, user_name, face_recognizer):
        """Encrypt file with face authentication requirement"""
        try:
            # Create metadata about the encryption
            metadata = {
                'encrypted_by': user_name,
                'encryption_method': 'face_auth',
                'original_filename': os.path.basename(input_file)
            }
            
            # Encrypt the file
            encrypted_file = input_file + f".{user_name}.face_encrypted"
            if not self.encrypt_file(input_file, encrypted_file):
                return False
            
            # Store metadata
            metadata_file = encrypted_file + ".meta"
            metadata_text = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            
            with open(metadata_file, 'w') as f:
                f.write(metadata_text)
            
            logger.info(f"File encrypted with face authentication: {encrypted_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error encrypting with face auth: {e}")
            return False
    
    def decrypt_with_face_auth(self, encrypted_file, authenticated_user):
        """Decrypt file with face authentication verification"""
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
            
            # Decrypt the file
            original_name = metadata.get('original_filename', 'decrypted_file')
            output_file = os.path.join(os.path.dirname(encrypted_file), original_name)
            
            if self.decrypt_file(encrypted_file, output_file):
                logger.info(f"File decrypted with face authentication: {output_file}")
                return True
            else:
                return False
                
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

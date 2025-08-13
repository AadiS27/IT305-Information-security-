# Face-based Authentication & Cryptography System

A secure biometric authentication system that uses face recognition to protect file encryption and decryption operations. This system combines advanced machine learning models with AES encryption to provide robust security for your sensitive files.

## Features

### 🔐 Security Features
- **Face Recognition Authentication**: Multi-model approach using SVM, Random Forest, Neural Networks, and similarity matching
- **AES File Encryption**: Industry-standard encryption for files and folders
- **Biometric Access Control**: Files can be encrypted with face-authentication requirements
- **Secure Key Management**: Automatic encryption key generation and management
- **Session Management**: Secure user sessions with automatic logout

### 🤖 Machine Learning Models
- **Support Vector Machine (SVM)**: Robust classification with RBF kernel
- **Random Forest**: Ensemble learning for improved accuracy
- **Neural Network**: Multi-layer perceptron for complex pattern recognition
- **Similarity Matching**: Distance-based face recognition using face encodings
- **Ensemble Method**: Combines all models for maximum accuracy

### 📁 File Operations
- **Single File Encryption/Decryption**: Protect individual files
- **Folder Encryption/Decryption**: Encrypt entire directory structures
- **Face-Authenticated Files**: Files that require face authentication to decrypt
- **Secure File Deletion**: Overwrite files before deletion for security

### 🎥 Camera Integration
- **Real-time Face Detection**: Live camera feed with face detection
- **Multi-sample Training**: Capture multiple face samples for better accuracy
- **Automatic Face Cropping**: Intelligent face extraction and normalization
- **Quality Validation**: Ensures good quality face samples for training

## Requirements

### System Requirements
- **Python**: 3.7 or higher
- **Camera**: Working webcam or external camera
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended for large datasets)
- **Storage**: At least 1GB free space for models and data

### Dependencies
```
opencv-python>=4.5.0
face-recognition>=1.3.0
scikit-learn>=1.0.0
numpy<2.0
cryptography>=3.0.0
pillow>=8.0.0
```

## Installation

### Option 1: Automated Setup (Recommended)
```bash
# Clone or download the project
cd face_auth_crypto

# Run the setup script
python setup.py
```

### Option 2: Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir face_data models encrypted_files

# Run the application
python main.py
```

### Option 3: Virtual Environment (Recommended for development)
```bash
# Create virtual environment
python -m venv face_auth_env

# Activate virtual environment
# On Windows:
face_auth_env\Scripts\activate
# On macOS/Linux:
source face_auth_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py
```

## Quick Start Guide

### 1. First Run
```bash
python main.py
```

### 2. Register Your First User
1. Choose option **1** (Register new user)
2. Enter a username
3. Position your face in front of the camera
4. Capture 20-30 face samples for best accuracy
5. Press 'c' to capture each sample

### 3. Train the Models
1. Choose option **3** (Train recognition models)
2. Wait for training to complete (may take a few minutes)
3. Review the accuracy scores displayed

### 4. Authenticate and Use
1. Choose option **2** (Authenticate user)
2. Position your face in front of the camera
3. Press SPACE when ready
4. Once authenticated, you can encrypt/decrypt files

## Usage Examples

### File Encryption
```
1. Authenticate with your face
2. Choose option 4 (Encrypt file)
3. Enter the path to your file
4. Choose encryption type:
   - Standard encryption (password-based)
   - Face-authenticated encryption (requires face auth to decrypt)
```

### File Decryption
```
1. Authenticate with your face
2. Choose option 5 (Decrypt file)
3. Enter the path to encrypted file
4. System automatically detects encryption type and decrypts
```

### Folder Operations
```
- Option 6: Encrypt entire folder
- Option 7: Decrypt entire folder
```

## Configuration

### Recognition Threshold
- **Default**: 0.6
- **Lower values**: More lenient (may allow wrong users)
- **Higher values**: More strict (may reject correct users)
- **Adjust**: Use option 9 (Adjust settings) in the main menu

### Model Parameters
- **SVM**: RBF kernel with probability estimation
- **Random Forest**: 100 estimators
- **Neural Network**: 128-64 hidden layers with early stopping
- **Face Encoding**: 128-dimensional face vectors

## File Structure

```
face_auth_crypto/
├── main.py                 # Main application
├── setup.py               # Setup script
├── requirements.txt       # Dependencies
├── README.md              # This file
├── src/
│   ├── face_capture.py    # Camera and face capture
│   ├── face_recognizer.py # ML models and recognition
│   └── crypto_manager.py  # Encryption and decryption
├── face_data/             # User face samples (created automatically)
│   ├── user1/
│   ├── user2/
│   └── ...
├── models/                # Trained ML models (created automatically)
├── encrypted_files/       # Sample encrypted files (optional)
└── logs/                  # Application logs (created automatically)
```

## Security Considerations

### Best Practices
1. **Use good lighting** when capturing face samples
2. **Capture samples from different angles** for better recognition
3. **Keep your crypto.key file secure** - this encrypts your files
4. **Regularly retrain models** when adding new users
5. **Use face-authenticated encryption** for highly sensitive files

### Security Features
- **AES-256 encryption** for all file operations
- **Secure random key generation** using cryptography library
- **Face encoding privacy** - only mathematical representations stored
- **Session timeout** and secure logout
- **Audit logging** for all operations

### Limitations
- **Lighting sensitivity**: Poor lighting affects recognition accuracy
- **Hardware dependency**: Requires working camera
- **Single face per authentication**: Currently supports one face at a time
- **Storage**: Face samples require disk space

## Troubleshooting

### Common Issues

#### Camera Not Working
```
Error: Could not open camera
Solution: 
- Check if camera is connected and working
- Close other applications using the camera
- Try running: python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

#### Low Recognition Accuracy
```
Problem: System not recognizing your face
Solutions:
1. Capture more training samples (30-50)
2. Ensure good lighting during capture and authentication
3. Lower the recognition threshold in settings
4. Retrain models after adding more samples
```

#### Installation Errors
```
Error: Package installation failed
Solutions:
1. Update pip: python -m pip install --upgrade pip
2. Install Visual C++ Build Tools (Windows)
3. Use conda instead of pip for some packages
4. Try installing packages individually
```

#### Import Errors
```
Error: Module not found
Solutions:
1. Ensure virtual environment is activated
2. Reinstall dependencies: pip install -r requirements.txt
3. Check Python version compatibility
```

### Performance Optimization

#### For Better Accuracy
- Capture 30-50 face samples per user
- Use consistent lighting conditions
- Include samples with slight variations (glasses, no glasses, etc.)
- Retrain models regularly

#### For Faster Processing
- Reduce image resolution for faster capture
- Use SVM or Random Forest instead of Neural Network
- Limit cross-validation folds for large datasets

## Advanced Features

### Batch Operations
```python
# Example: Encrypt multiple files
for file in file_list:
    crypto_manager.encrypt_file(file)
```

### Custom Thresholds per User
```python
# Set different thresholds for different users
recognizer.set_user_threshold("user1", 0.7)
recognizer.set_user_threshold("user2", 0.5)
```

### Integration with Other Systems
The system is designed to be modular and can be integrated with:
- Web applications (Flask/Django)
- Desktop applications (Tkinter/PyQt)
- IoT devices
- Cloud storage systems

## Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Make your changes
5. Test thoroughly
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include error handling
- Write unit tests for new features

## License

This project is for educational and research purposes. Please ensure compliance with local laws and regulations regarding biometric data processing and cryptography.

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review the logs in `face_auth_crypto.log`
3. Create an issue with detailed error information
4. Include system information and steps to reproduce

## Changelog

### Version 1.0.0
- Initial release
- Multi-model face recognition
- AES file encryption
- Real-time face capture
- Ensemble recognition method
- Session management
- Comprehensive logging

---

**⚠️ Important Security Notice**: This system processes biometric data. Ensure compliance with GDPR, CCPA, and other privacy regulations in your jurisdiction. Always inform users about biometric data collection and obtain proper consent.

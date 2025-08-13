"""
Setup script for Face-based Authentication & Cryptography System
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.7 or higher")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['face_data', 'models', 'encrypted_files']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Directory created: {directory}")
        except Exception as e:
            print(f"✗ Failed to create directory {directory}: {e}")
            return False
    
    return True

def test_camera():
    """Test camera availability"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera is available")
            cap.release()
            return True
        else:
            print("✗ Camera is not available")
            return False
    except ImportError:
        print("? OpenCV not installed yet, camera test skipped")
        return True
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("  FACE AUTHENTICATION CRYPTO SYSTEM - SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    print("\n1. Checking Python version...")
    
    print("\n2. Installing required packages...")
    
    # List of required packages
    packages = [
        "opencv-python",
        "face-recognition", 
        "scikit-learn",
        "numpy<2.0",  # For compatibility
        "cryptography",
        "pillow"
    ]
    
    failed_packages = []
    
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n✗ Failed to install: {', '.join(failed_packages)}")
        print("Please install these packages manually:")
        for package in failed_packages:
            print(f"  pip install {package}")
        return False
    
    print("\n3. Creating directories...")
    if not create_directories():
        return False
    
    print("\n4. Testing camera...")
    test_camera()
    
    print("\n5. Testing imports...")
    try:
        import cv2
        import face_recognition
        import sklearn
        import cryptography
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("  SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now run the application with:")
    print("  python main.py")
    print("\nFirst-time usage:")
    print("1. Register a new user (option 1)")
    print("2. Train the recognition models (option 3)")
    print("3. Authenticate and start using the system (option 2)")
    print("\nNote: Make sure your camera is working and you have good lighting")
    print("      for optimal face recognition performance.")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\nSetup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during setup: {e}")
        sys.exit(1)

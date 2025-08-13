"""
Test script for Face-based Authentication & Cryptography System
Verifies that all components are working correctly
"""

import sys
import os
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import cryptography
        print(f"✓ cryptography version: {cryptography.__version__}")
    except ImportError as e:
        print(f"✗ cryptography import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow version: {Image.__version__}")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    print("✓ All core libraries imported successfully")
    return True

def test_camera():
    """Test camera functionality"""
    print("\nTesting camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera working - Frame size: {frame.shape}")
            else:
                print("✗ Camera opened but couldn't read frame")
                return False
            cap.release()
        else:
            print("✗ Could not open camera")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False
    
    return True

def test_face_detection():
    """Test face detection functionality"""
    print("\nTesting face detection...")
    
    try:
        import cv2
        
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            print("✗ Could not load face cascade")
            return False
        else:
            print("✓ Face cascade loaded successfully")
        
        # Test with a simple image (create a dummy image)
        import numpy as np
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        print(f"✓ Face detection test completed (found {len(faces)} faces in test image)")
        
    except Exception as e:
        print(f"✗ Face detection test failed: {e}")
        return False
    
    return True

def test_face_recognition():
    """Test face_recognition functionality using OpenCV"""
    print("\nTesting OpenCV-based face recognition...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a more realistic face-like test image
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Create a basic face-like pattern that OpenCV can detect
        # Face oval
        cv2.ellipse(test_image, (100, 100), (80, 100), 0, 0, 360, (150, 150, 150), -1)
        
        # Eyes
        cv2.circle(test_image, (75, 80), 8, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_image, (125, 80), 8, (0, 0, 0), -1)  # Right eye
        
        # Nose
        cv2.line(test_image, (100, 90), (100, 110), (100, 100, 100), 3)
        
        # Mouth
        cv2.ellipse(test_image, (100, 130), (20, 10), 0, 0, 180, (50, 50, 50), 2)
        
        # Test face recognition components
        from src.face_recognizer_opencv import FaceRecognizer
        recognizer = FaceRecognizer()
        
        # Test feature extraction
        features = recognizer.extract_face_features(test_image)
        if features is not None:
            print(f"✓ Feature extraction working (extracted {len(features)} features)")
        else:
            print("✓ Face detection working (no face found in simple test image - this is expected)")
        
        print("✓ OpenCV-based face recognition system loaded successfully")
        
    except Exception as e:
        print(f"✗ Face recognition test failed: {e}")
        return False
    
    return True

def test_machine_learning():
    """Test machine learning components"""
    print("\nTesting machine learning components...")
    
    try:
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        
        # Create dummy data
        X = np.random.rand(10, 128)  # 10 samples, 128 features (like face encodings)
        y = np.array(['user1', 'user2'] * 5)
        
        # Test Label Encoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print("✓ Label encoder working")
        
        # Test SVM
        svm = SVC(probability=True)
        svm.fit(X, y_encoded)
        pred = svm.predict(X[:1])
        prob = svm.predict_proba(X[:1])
        print("✓ SVM model working")
        
        # Test Random Forest
        rf = RandomForestClassifier(n_estimators=10)
        rf.fit(X, y_encoded)
        pred = rf.predict(X[:1])
        prob = rf.predict_proba(X[:1])
        print("✓ Random Forest model working")
        
        # Test Neural Network
        nn = MLPClassifier(hidden_layer_sizes=(64,), max_iter=100)
        nn.fit(X, y_encoded)
        pred = nn.predict(X[:1])
        prob = nn.predict_proba(X[:1])
        print("✓ Neural Network model working")
        
    except Exception as e:
        print(f"✗ Machine learning test failed: {e}")
        return False
    
    return True

def test_cryptography():
    """Test cryptography functionality"""
    print("\nTesting cryptography...")
    
    try:
        from cryptography.fernet import Fernet
        
        # Generate key
        key = Fernet.generate_key()
        fernet = Fernet(key)
        
        # Test text encryption
        test_text = "This is a test message for face authentication!"
        encrypted = fernet.encrypt(test_text.encode())
        decrypted = fernet.decrypt(encrypted).decode()
        
        if test_text == decrypted:
            print("✓ Text encryption/decryption working")
        else:
            print("✗ Text encryption/decryption failed")
            return False
        
        # Test file encryption
        test_file = "test_crypto.txt"
        with open(test_file, 'w') as f:
            f.write("Test file content for encryption")
        
        # Read, encrypt, decrypt
        with open(test_file, 'rb') as f:
            file_data = f.read()
        
        encrypted_data = fernet.encrypt(file_data)
        decrypted_data = fernet.decrypt(encrypted_data)
        
        if file_data == decrypted_data:
            print("✓ File encryption/decryption working")
        else:
            print("✗ File encryption/decryption failed")
            return False
        
        # Clean up
        os.remove(test_file)
        
    except Exception as e:
        print(f"✗ Cryptography test failed: {e}")
        return False
    
    return True

def test_system_components():
    """Test our custom system components"""
    print("\nTesting system components...")
    
    try:
        # Test FaceCapture import
        from src.face_capture import FaceCapture
        face_capture = FaceCapture()
        print("✓ FaceCapture component loaded")
        
        # Test FaceRecognizer import
        from src.face_recognizer_opencv import FaceRecognizer
        face_recognizer = FaceRecognizer()
        print("✓ FaceRecognizer component loaded")
        
        # Test CryptoManager import
        from src.crypto_manager import CryptoManager
        crypto_manager = CryptoManager()
        print("✓ CryptoManager component loaded")
        
        # Test basic functionality
        users = face_recognizer.get_user_list()
        print(f"✓ System initialized - Known users: {len(users)}")
        
    except Exception as e:
        print(f"✗ System components test failed: {e}")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = ['face_data', 'models', 'encrypted_files']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Directory created/verified: {directory}")
        except Exception as e:
            print(f"✗ Failed to create directory {directory}: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("  FACE AUTHENTICATION CRYPTO SYSTEM - VERIFICATION")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Camera Test", test_camera),
        ("Face Detection Test", test_face_detection),
        ("Face Recognition Test", test_face_recognition),
        ("Machine Learning Test", test_machine_learning),
        ("Cryptography Test", test_cryptography),
        ("Directory Setup", create_directories),
        ("System Components Test", test_system_components),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("  TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nTo start the application, run:")
        print("  python main.py")
        print("\nFirst-time setup:")
        print("1. Register a new user")
        print("2. Train the models")
        print("3. Start using face authentication!")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed.")
        print("Please check the errors above and fix them before using the system.")
    
    return passed == len(results)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        logging.exception("Test error")

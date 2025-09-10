"""
Face Recognition Module for Face-based Authentication System
Handles face encoding and recognition using OpenCV and sklearn
"""

import numpy as np
import os
import pickle
import logging
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, model_dir="models"):
        """Initialize face recognizer"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Face encodings database
        self.known_encodings = []
        self.known_names = []
        
        # ML models
        self.svm_model = None
        self.rf_model = None
        self.nn_model = None
        
        # Model accuracy metrics for presentation
        self.svm_accuracy = 0.0
        self.rf_accuracy = 0.0
        self.nn_accuracy = 0.0
        self.svm_cv_accuracy = 0.0
        self.rf_cv_accuracy = 0.0
        self.nn_cv_accuracy = 0.0
        self.svm_cv_std = 0.0
        self.rf_cv_std = 0.0
        self.nn_cv_std = 0.0
        
        # Initialize confusion matrix storage
        self.svm_confusion_matrix = None
        self.rf_confusion_matrix = None
        self.nn_confusion_matrix = None
        
        # Store test data and predictions
        self.X_test = None
        self.y_test = None
        self.svm_predictions = None
        self.rf_predictions = None
        self.nn_predictions = None
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        # Recognition threshold
        self.recognition_threshold = 0.6
        
        # OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Feature extractor for additional features
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # Try to load existing models on startup
        if self._models_exist():
            logger.info("Found existing models, loading...")
            if self.load_models():
                logger.info("Successfully loaded existing models")
            else:
                logger.warning("Failed to load existing models, will need to train new ones")
        
    def extract_face_features(self, image_path_or_array):
        """Extract face features from image using OpenCV"""
        try:
            if isinstance(image_path_or_array, str):
                # Load image from path
                image = cv2.imread(image_path_or_array)
                if image is None:
                    logger.warning(f"Could not load image: {image_path_or_array}")
                    return None
            else:
                # Use provided array
                image = image_path_or_array.copy()
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                logger.warning("No face found in image")
                return None
            
            # Use the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            
            # Extract multiple types of features
            features = self._extract_comprehensive_features(face_resized, image[y:y+h, x:x+w])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting face features: {e}")
            return None
    
    def _extract_comprehensive_features(self, face_gray, face_color):
        """Extract comprehensive features from face"""
        features = []
        
        try:
            # 1. Histogram features
            hist = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
            hist_features = hist.flatten() / np.sum(hist)  # Normalize
            features.extend(hist_features[:50])  # Take first 50 bins
            
            # 2. LBP (Local Binary Pattern) features
            lbp_features = self._calculate_lbp(face_gray)
            features.extend(lbp_features)
            
            # 3. Haar-like features (simplified)
            haar_features = self._calculate_haar_features(face_gray)
            features.extend(haar_features)
            
            # 4. Edge features
            edges = cv2.Canny(face_gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # 5. Texture features (using Gabor filters)
            gabor_features = self._calculate_gabor_features(face_gray)
            features.extend(gabor_features)
            
            # 6. Color features (if color image available)
            if len(face_color.shape) == 3:
                color_features = self._calculate_color_features(face_color)
                features.extend(color_features)
            else:
                features.extend([0] * 10)  # Placeholder for color features
            
            # 7. Geometric features
            geometric_features = self._calculate_geometric_features(face_gray)
            features.extend(geometric_features)
            
            # Ensure fixed length
            target_length = 128
            if len(features) > target_length:
                features = features[:target_length]
            elif len(features) < target_length:
                features.extend([0] * (target_length - len(features)))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error in comprehensive feature extraction: {e}")
            # Return zero vector as fallback
            return np.zeros(128, dtype=np.float32)
    
    def _calculate_lbp(self, image):
        """Calculate Local Binary Pattern features"""
        try:
            # Simple LBP implementation
            height, width = image.shape
            lbp = np.zeros((height-2, width-2), dtype=np.uint8)
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    center = image[i, j]
                    binary_string = ""
                    
                    # 8-connected neighbors
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += "1" if neighbor >= center else "0"
                    
                    lbp[i-1, j-1] = int(binary_string, 2)
            
            # Calculate histogram
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist = hist / (np.sum(hist) + 1e-7)  # Normalize
            
            return hist[:20]  # Return first 20 bins
            
        except Exception as e:
            logger.error(f"Error calculating LBP: {e}")
            return np.zeros(20)
    
    def _calculate_haar_features(self, image):
        """Calculate simplified Haar-like features"""
        try:
            h, w = image.shape
            features = []
            
            # Horizontal rectangles
            top_half = np.mean(image[:h//2, :])
            bottom_half = np.mean(image[h//2:, :])
            features.append(top_half - bottom_half)
            
            # Vertical rectangles
            left_half = np.mean(image[:, :w//2])
            right_half = np.mean(image[:, w//2:])
            features.append(left_half - right_half)
            
            # Diagonal features
            center_x, center_y = w//2, h//2
            
            # Four quadrants
            q1 = np.mean(image[:center_y, :center_x])
            q2 = np.mean(image[:center_y, center_x:])
            q3 = np.mean(image[center_y:, :center_x])
            q4 = np.mean(image[center_y:, center_x:])
            
            features.extend([q1, q2, q3, q4])
            features.append((q1 + q4) - (q2 + q3))  # Diagonal difference
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating Haar features: {e}")
            return [0] * 7
    
    def _calculate_gabor_features(self, image):
        """Calculate Gabor filter features"""
        try:
            features = []
            
            # Different orientations and frequencies
            orientations = [0, 45, 90, 135]
            frequencies = [0.1, 0.2]
            
            for angle in orientations:
                for freq in frequencies:
                    # Create Gabor kernel
                    theta = np.radians(angle)
                    kernel = cv2.getGaborKernel((21, 21), 5, theta, 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F)
                    
                    # Apply filter
                    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    
                    # Calculate response statistics
                    features.append(np.mean(filtered))
                    features.append(np.std(filtered))
            
            return features[:16]  # Limit to 16 features
            
        except Exception as e:
            logger.error(f"Error calculating Gabor features: {e}")
            return [0] * 16
    
    def _calculate_color_features(self, color_image):
        """Calculate color-based features"""
        try:
            features = []
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            
            # Calculate mean and std for each channel
            for channel in range(3):
                features.append(np.mean(color_image[:, :, channel]))
                features.append(np.std(color_image[:, :, channel]))
                features.append(np.mean(hsv[:, :, channel]))
                
            # Skin color features (simplified)
            skin_mask = self._detect_skin_pixels(color_image)
            skin_ratio = np.sum(skin_mask) / (skin_mask.shape[0] * skin_mask.shape[1])
            features.append(skin_ratio)
            
            return features[:10]
            
        except Exception as e:
            logger.error(f"Error calculating color features: {e}")
            return [0] * 10
    
    def _detect_skin_pixels(self, image):
        """Simple skin detection"""
        try:
            # Convert to YCrCb color space
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            # Define skin color thresholds
            lower_skin = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin = np.array([255, 173, 127], dtype=np.uint8)
            
            # Create mask
            skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
            
            return skin_mask > 0
            
        except Exception as e:
            logger.error(f"Error in skin detection: {e}")
            return np.zeros(image.shape[:2], dtype=bool)
    
    def _calculate_geometric_features(self, image):
        """Calculate geometric features"""
        try:
            features = []
            
            # Calculate image moments
            moments = cv2.moments(image)
            
            # Hu moments (scale, rotation, translation invariant)
            hu_moments = cv2.HuMoments(moments)
            hu_features = [-np.sign(hu) * np.log10(np.abs(hu) + 1e-10) for hu in hu_moments.flatten()]
            features.extend(hu_features)
            
            # Aspect ratio
            h, w = image.shape
            aspect_ratio = w / h
            features.append(aspect_ratio)
            
            # Fill ratio (area of face vs bounding box)
            non_zero_pixels = np.count_nonzero(image)
            total_pixels = h * w
            fill_ratio = non_zero_pixels / total_pixels
            features.append(fill_ratio)
            
            return features[:10]
            
        except Exception as e:
            logger.error(f"Error calculating geometric features: {e}")
            return [0] * 10
    
    def load_face_data(self, data_dir="face_data"):
        """Load face data from directory structure"""
        encodings = []
        names = []
        
        if not os.path.exists(data_dir):
            logger.warning(f"Face data directory {data_dir} does not exist")
            return encodings, names
        
        for user_name in os.listdir(data_dir):
            user_dir = os.path.join(data_dir, user_name)
            if not os.path.isdir(user_dir):
                continue
                
            user_encodings = []
            
            for image_file in os.listdir(user_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(user_dir, image_file)
                    encoding = self.extract_face_features(image_path)
                    
                    if encoding is not None:
                        user_encodings.append(encoding)
            
            if user_encodings:
                encodings.extend(user_encodings)
                names.extend([user_name] * len(user_encodings))
                logger.info(f"Loaded {len(user_encodings)} encodings for {user_name}")
        
        return encodings, names
    
    def train_models(self, data_dir="face_data"):
        """Train ML models with face data"""
        logger.info("Loading face data...")
        encodings, names = self.load_face_data(data_dir)
        
        if len(encodings) == 0:
            logger.error("No face data found for training")
            return False
        
        if len(set(names)) < 2:
            logger.error("Need at least 2 different users for training")
            return False
        
        # Convert to numpy arrays
        X = np.array(encodings)
        y = np.array(names)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"Training with {len(X)} samples from {len(set(names))} users")
        
        # Split data for evaluation
        if len(X) >= 4:  # Need at least 4 samples for train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y_encoded, y_encoded
            logger.warning("Using same data for training and testing due to small dataset")
        
        # Train SVM
        logger.info("Training SVM model...")
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(X_train, y_train)
        
        # Train Random Forest
        logger.info("Training Random Forest model...")
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        # Train Neural Network
        logger.info("Training Neural Network model...")
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.nn_model.fit(X_train, y_train)
        
        # Evaluate models
        models = {
            'SVM': self.svm_model,
            'Random Forest': self.rf_model,
            'Neural Network': self.nn_model
        }
        
        # Store test data for confusion matrix generation
        self.X_test = X_test
        self.y_test = y_test
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"{name} accuracy: {accuracy:.4f}")
            
            # Store predictions
            if name == 'SVM':
                self.svm_accuracy = accuracy
                self.svm_predictions = y_pred
                self.svm_confusion_matrix = confusion_matrix(y_test, y_pred)
            elif name == 'Random Forest':
                self.rf_accuracy = accuracy
                self.rf_predictions = y_pred
                self.rf_confusion_matrix = confusion_matrix(y_test, y_pred)
            elif name == 'Neural Network':
                self.nn_accuracy = accuracy
                self.nn_predictions = y_pred
                self.nn_confusion_matrix = confusion_matrix(y_test, y_pred)
            
            # Cross-validation (if enough samples)
            if len(X) >= 3:
                try:
                    cv_scores = cross_val_score(model, X, y_encoded, cv=min(3, len(X)), scoring='accuracy')
                    logger.info(f"{name} cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                    
                    # Store cross-validation metrics
                    if name == 'SVM':
                        self.svm_cv_accuracy = cv_scores.mean()
                        self.svm_cv_std = cv_scores.std()
                    elif name == 'Random Forest':
                        self.rf_cv_accuracy = cv_scores.mean()
                        self.rf_cv_std = cv_scores.std()
                    elif name == 'Neural Network':
                        self.nn_cv_accuracy = cv_scores.mean()
                        self.nn_cv_std = cv_scores.std()
                        
                except Exception as e:
                    logger.warning(f"Cross-validation failed for {name}: {e}")
        
        # Store known encodings for similarity-based recognition
        self.known_encodings = encodings
        self.known_names = names
        
        # Save models
        self.save_models()
        
        return True
    
    def get_registered_users(self):
        """Get list of registered users from face_data directory"""
        face_data_dir = "face_data"
        if not os.path.exists(face_data_dir):
            return []
        
        users = []
        for item in os.listdir(face_data_dir):
            user_dir = os.path.join(face_data_dir, item)
            if os.path.isdir(user_dir):
                # Check if directory has any image files
                image_files = [f for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    users.append(item)
        
        return users
    
    def _models_exist(self):
        """Check if all required model files exist"""
        required_files = [
            'svm_model.pkl',
            'rf_model.pkl',
            'nn_model.pkl',
            'label_encoder.pkl',
            'known_encodings.pkl'
        ]
        
        for filename in required_files:
            if not os.path.exists(os.path.join(self.model_dir, filename)):
                return False
        return True
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            # Save models
            with open(os.path.join(self.model_dir, 'svm_model.pkl'), 'wb') as f:
                pickle.dump(self.svm_model, f)
                
            with open(os.path.join(self.model_dir, 'rf_model.pkl'), 'wb') as f:
                pickle.dump(self.rf_model, f)
                
            with open(os.path.join(self.model_dir, 'nn_model.pkl'), 'wb') as f:
                pickle.dump(self.nn_model, f)
                
            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.label_encoder, f)
                
            # Save known encodings
            with open(os.path.join(self.model_dir, 'known_encodings.pkl'), 'wb') as f:
                pickle.dump((self.known_encodings, self.known_names), f)
                
            logger.info("Models saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load models
            with open(os.path.join(self.model_dir, 'svm_model.pkl'), 'rb') as f:
                self.svm_model = pickle.load(f)
                
            with open(os.path.join(self.model_dir, 'rf_model.pkl'), 'rb') as f:
                self.rf_model = pickle.load(f)
                
            with open(os.path.join(self.model_dir, 'nn_model.pkl'), 'rb') as f:
                self.nn_model = pickle.load(f)
                
            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            # Load known encodings
            with open(os.path.join(self.model_dir, 'known_encodings.pkl'), 'rb') as f:
                self.known_encodings, self.known_names = pickle.load(f)
                
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def recognize_face(self, face_image, method='ensemble'):
        """Recognize face using trained models"""
        # Extract features from face image
        features = self.extract_face_features(face_image)
        
        if features is None:
            return None, 0.0
        
        if method == 'similarity':
            return self._recognize_by_similarity(features)
        elif method == 'svm':
            return self._recognize_by_ml(features, self.svm_model)
        elif method == 'rf':
            return self._recognize_by_ml(features, self.rf_model)
        elif method == 'nn':
            return self._recognize_by_ml(features, self.nn_model)
        elif method == 'ensemble':
            return self._recognize_by_ensemble(features)
        else:
            logger.error(f"Unknown recognition method: {method}")
            return None, 0.0
    
    def _recognize_by_similarity(self, features):
        """Recognize face using similarity matching"""
        if not self.known_encodings:
            return None, 0.0
        
        # Calculate distances using cosine similarity
        similarities = []
        for known_encoding in self.known_encodings:
            # Cosine similarity
            dot_product = np.dot(features, known_encoding)
            norm_product = np.linalg.norm(features) * np.linalg.norm(known_encoding)
            similarity = dot_product / (norm_product + 1e-10)
            similarities.append(similarity)
        
        if len(similarities) == 0:
            return None, 0.0
        
        # Find best match
        best_match_index = np.argmax(similarities)
        max_similarity = similarities[best_match_index]
        
        # Convert similarity to confidence
        confidence = max_similarity
        
        if confidence >= self.recognition_threshold:
            return self.known_names[best_match_index], confidence
        else:
            return None, confidence
    
    def _recognize_by_ml(self, features, model):
        """Recognize face using ML model"""
        if model is None:
            return None, 0.0
        
        # Predict
        try:
            probabilities = model.predict_proba([features])[0]
            predicted_class = model.predict([features])[0]
            
            confidence = np.max(probabilities)
            predicted_name = self.label_encoder.inverse_transform([predicted_class])[0]
            
            if confidence >= self.recognition_threshold:
                return predicted_name, confidence
            else:
                return None, confidence
                
        except Exception as e:
            logger.error(f"Error in ML recognition: {e}")
            return None, 0.0
    
    def _recognize_by_ensemble(self, features):
        """Recognize face using ensemble of methods"""
        results = []
        
        # Get predictions from all methods
        methods = [
            ('similarity', self._recognize_by_similarity),
            ('svm', lambda feat: self._recognize_by_ml(feat, self.svm_model)),
            ('rf', lambda feat: self._recognize_by_ml(feat, self.rf_model)),
            ('nn', lambda feat: self._recognize_by_ml(feat, self.nn_model))
        ]
        
        for method_name, method_func in methods:
            try:
                name, confidence = method_func(features)
                if name is not None:
                    results.append((name, confidence, method_name))
            except Exception as e:
                logger.warning(f"Error in {method_name} method: {e}")
        
        if not results:
            return None, 0.0
        
        # Ensemble voting - majority vote with confidence weighting
        name_votes = {}
        total_confidence = 0
        
        for name, confidence, method in results:
            if name not in name_votes:
                name_votes[name] = {'count': 0, 'total_confidence': 0, 'methods': []}
            
            name_votes[name]['count'] += 1
            name_votes[name]['total_confidence'] += confidence
            name_votes[name]['methods'].append(method)
            total_confidence += confidence
        
        # Find best candidate
        best_name = None
        best_score = 0
        
        for name, data in name_votes.items():
            # Score based on vote count and average confidence
            avg_confidence = data['total_confidence'] / data['count']
            vote_ratio = data['count'] / len(results)
            score = avg_confidence * vote_ratio
            
            if score > best_score:
                best_score = score
                best_name = name
        
        # Calculate overall confidence
        overall_confidence = best_score if best_name else 0.0
        
        if overall_confidence >= self.recognition_threshold:
            return best_name, overall_confidence
        else:
            return None, overall_confidence
    
    def add_user_encoding(self, user_name, face_image):
        """Add a new user encoding"""
        features = self.extract_face_features(face_image)
        if features is not None:
            self.known_encodings.append(features)
            self.known_names.append(user_name)
            return True
        return False
    
    def get_user_list(self):
        """Get list of known users"""
        return list(set(self.known_names))
    
    def set_recognition_threshold(self, threshold):
        """Set recognition threshold"""
        self.recognition_threshold = threshold
        logger.info(f"Recognition threshold set to {threshold}")

if __name__ == "__main__":
    # Test the face recognizer
    recognizer = FaceRecognizer()
    
    print("1. Train models")
    print("2. Test recognition")
    print("3. Load existing models")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        if recognizer.train_models():
            print("Models trained successfully!")
        else:
            print("Failed to train models")
    elif choice == "2":
        recognizer.load_models()
        print("Load a test image...")
        # This would require additional test implementation
    elif choice == "3":
        if recognizer.load_models():
            print("Models loaded successfully!")
            print(f"Known users: {recognizer.get_user_list()}")
        else:
            print("Failed to load models")

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
from sklearn.metrics import accuracy_score, classification_report
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
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        # Recognition threshold
        self.recognition_threshold = 0.6
        
    def extract_face_encoding(self, image_path_or_array):
        """Extract face encoding from image"""
        try:
            if isinstance(image_path_or_array, str):
                # Load image from path
                image = face_recognition.load_image_file(image_path_or_array)
            else:
                # Use provided array (BGR to RGB conversion if needed)
                if len(image_path_or_array.shape) == 3 and image_path_or_array.shape[2] == 3:
                    image = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
                else:
                    image = image_path_or_array
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                logger.warning("No face found in image")
                return None
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                logger.warning("Could not encode face")
                return None
                
            # Return the first face encoding
            return face_encodings[0]
            
        except Exception as e:
            logger.error(f"Error extracting face encoding: {e}")
            return None
    
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
                    encoding = self.extract_face_encoding(image_path)
                    
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
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"{name} accuracy: {accuracy:.4f}")
            
            # Cross-validation (if enough samples)
            if len(X) >= 3:
                try:
                    cv_scores = cross_val_score(model, X, y_encoded, cv=min(3, len(X)), scoring='accuracy')
                    logger.info(f"{name} cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                except Exception as e:
                    logger.warning(f"Cross-validation failed for {name}: {e}")
        
        # Store known encodings for similarity-based recognition
        self.known_encodings = encodings
        self.known_names = names
        
        # Save models
        self.save_models()
        
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
        # Extract encoding from face image
        encoding = self.extract_face_encoding(face_image)
        
        if encoding is None:
            return None, 0.0
        
        if method == 'similarity':
            return self._recognize_by_similarity(encoding)
        elif method == 'svm':
            return self._recognize_by_ml(encoding, self.svm_model)
        elif method == 'rf':
            return self._recognize_by_ml(encoding, self.rf_model)
        elif method == 'nn':
            return self._recognize_by_ml(encoding, self.nn_model)
        elif method == 'ensemble':
            return self._recognize_by_ensemble(encoding)
        else:
            logger.error(f"Unknown recognition method: {method}")
            return None, 0.0
    
    def _recognize_by_similarity(self, encoding):
        """Recognize face using similarity matching"""
        if not self.known_encodings:
            return None, 0.0
        
        # Calculate distances
        distances = face_recognition.face_distance(self.known_encodings, encoding)
        
        if len(distances) == 0:
            return None, 0.0
        
        # Find best match
        best_match_index = np.argmin(distances)
        min_distance = distances[best_match_index]
        
        # Convert distance to confidence
        confidence = 1.0 - min_distance
        
        if confidence >= self.recognition_threshold:
            return self.known_names[best_match_index], confidence
        else:
            return None, confidence
    
    def _recognize_by_ml(self, encoding, model):
        """Recognize face using ML model"""
        if model is None:
            return None, 0.0
        
        # Predict
        try:
            probabilities = model.predict_proba([encoding])[0]
            predicted_class = model.predict([encoding])[0]
            
            confidence = np.max(probabilities)
            predicted_name = self.label_encoder.inverse_transform([predicted_class])[0]
            
            if confidence >= self.recognition_threshold:
                return predicted_name, confidence
            else:
                return None, confidence
                
        except Exception as e:
            logger.error(f"Error in ML recognition: {e}")
            return None, 0.0
    
    def _recognize_by_ensemble(self, encoding):
        """Recognize face using ensemble of methods"""
        results = []
        
        # Get predictions from all methods
        methods = [
            ('similarity', self._recognize_by_similarity),
            ('svm', lambda enc: self._recognize_by_ml(enc, self.svm_model)),
            ('rf', lambda enc: self._recognize_by_ml(enc, self.rf_model)),
            ('nn', lambda enc: self._recognize_by_ml(enc, self.nn_model))
        ]
        
        for method_name, method_func in methods:
            try:
                name, confidence = method_func(encoding)
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
        encoding = self.extract_face_encoding(face_image)
        if encoding is not None:
            self.known_encodings.append(encoding)
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

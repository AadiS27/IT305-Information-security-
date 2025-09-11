"""
Face Capture Module for Face-based Authentication System
Handles face detection and capture using OpenCV
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceCapture:
    def __init__(self, cascade_path=None):
        """Initialize face capture with Haar cascade"""
        if cascade_path is None:
            # Use OpenCV's pre-trained face cascade
            self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        else:
            self.cascade_path = cascade_path
            
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        # Camera settings
        self.camera = None
        self.frame_width = 640
        self.frame_height = 480
        
    def start_camera(self):
        """Start the camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
                
            logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera"""
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()
            logger.info("Camera stopped")
    
    def detect_faces(self, frame):
        """Detect faces in a frame with multiple parameter sets for robustness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection parameters
        detection_params = [
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20)},  # More sensitive
            {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},   # Default
            {'scaleFactor': 1.2, 'minNeighbors': 3, 'minSize': (40, 40)},   # Less sensitive
        ]
        
        for params in detection_params:
            faces = self.face_cascade.detectMultiScale(gray, **params)
            if len(faces) > 0:
                return faces
        
        # If no faces found, try with histogram equalization
        equalized = cv2.equalizeHist(gray)
        for params in detection_params:
            faces = self.face_cascade.detectMultiScale(equalized, **params)
            if len(faces) > 0:
                return faces
        
        return []
    
    def capture_face_samples(self, user_name, num_samples=20, save_dir="face_data"):
        """Capture multiple face samples for training"""
        if not self.start_camera():
            return False
            
        # Create user directory
        user_dir = os.path.join(save_dir, user_name)
        os.makedirs(user_dir, exist_ok=True)
        
        captured_samples = 0
        
        print(f"Capturing {num_samples} face samples for {user_name}")
        print("Position your face in front of the camera. Press 'c' to capture, 'q' to quit")
        
        while captured_samples < num_samples:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Draw rectangles around faces
            display_frame = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Face {captured_samples+1}/{num_samples}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display progress
            cv2.putText(display_frame, f"Captured: {captured_samples}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and len(faces) > 0:
                # Capture the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                # Extract and save face
                face_roi = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (160, 160))
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{user_name}_{captured_samples+1}_{timestamp}.jpg"
                filepath = os.path.join(user_dir, filename)
                
                cv2.imwrite(filepath, face_resized)
                captured_samples += 1
                
                print(f"Captured sample {captured_samples}/{num_samples}")
                time.sleep(0.5)  # Brief pause to avoid double captures
                
            elif key == ord('q'):
                break
        
        self.stop_camera()
        
        if captured_samples == num_samples:
            print(f"Successfully captured {captured_samples} samples for {user_name}")
            return True
        else:
            print(f"Captured only {captured_samples}/{num_samples} samples")
            return False
    
    def capture_single_face(self, timeout=10):
        """Capture a single face for authentication"""
        if not self.start_camera():
            return None
            
        print("Position your face in front of the camera for authentication")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Draw rectangles around faces
            display_frame = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, "Face detected - Hold still", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            remaining_time = int(timeout - (time.time() - start_time))
            cv2.putText(display_frame, f"Time remaining: {remaining_time}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE when ready", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Authentication', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and len(faces) > 0:
                # Capture the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                # Extract face
                face_roi = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (160, 160))
                
                self.stop_camera()
                return face_resized
                
            elif key == ord('q'):
                break
        
        self.stop_camera()
        return None
    
    def preview_camera(self):
        """Preview camera feed"""
        if not self.start_camera():
            return
            
        print("Camera preview - Press 'q' to quit")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face detected", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Preview', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop_camera()

if __name__ == "__main__":
    # Test the face capture
    face_capture = FaceCapture()
    
    print("1. Preview camera")
    print("2. Capture face samples")
    print("3. Test single face capture")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        face_capture.preview_camera()
    elif choice == "2":
        user_name = input("Enter user name: ")
        num_samples = int(input("Enter number of samples (default 20): ") or 20)
        face_capture.capture_face_samples(user_name, num_samples)
    elif choice == "3":
        face = face_capture.capture_single_face()
        if face is not None:
            print("Face captured successfully!")
            cv2.imshow('Captured Face', face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to capture face")

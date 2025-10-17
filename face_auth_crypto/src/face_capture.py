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
        """Start the camera with optimized settings"""
        try:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend for faster startup on Windows
            
            # Set properties for faster initialization
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lower latency
            
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            
            # Warm up camera (capture and discard a few frames)
            for _ in range(5):
                self.camera.read()
                
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
        """Detect faces in a frame - optimized for speed"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use single optimized parameter set for speed
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4,  # Balanced sensitivity
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            return faces
        
        # Fallback: Try with histogram equalization only if no faces found
        equalized = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            equalized,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def capture_face_samples(self, user_name, num_samples=20, save_dir="face_data"):
        """Capture multiple face samples for training - AUTO CAPTURE"""
        if not self.start_camera():
            return False
            
        # Create user directory
        user_dir = os.path.join(save_dir, user_name)
        os.makedirs(user_dir, exist_ok=True)
        
        captured_samples = 0
        last_capture_time = 0
        auto_capture_delay = 0.8  # Seconds between auto captures
        
        print(f"Capturing {num_samples} face samples for {user_name}")
        print("Position your face in front of the camera. Auto-capture enabled!")
        print("Press 'q' to quit early")
        
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
            cv2.putText(display_frame, "AUTO-CAPTURE MODE - Press 'q' to quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Capture', display_frame)
            
            # Auto-capture when face is detected and enough time has passed
            current_time = time.time()
            if len(faces) > 0 and (current_time - last_capture_time) >= auto_capture_delay:
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
                last_capture_time = current_time
                
                print(f"Auto-captured sample {captured_samples}/{num_samples}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        self.stop_camera()
        
        if captured_samples >= 2:  # Minimum 2 samples needed
            print(f"Successfully captured {captured_samples} samples for {user_name}")
            return captured_samples
        else:
            print(f"Captured only {captured_samples} samples - need at least 2")
            return 0
    
    def capture_single_face(self, timeout=10):
        """Capture a single face for authentication - AUTO CAPTURE after face is stable"""
        if not self.start_camera():
            return None
            
        print("Position your face in front of the camera for authentication")
        print("Auto-capture enabled - hold still when face is detected!")
        
        start_time = time.time()
        face_stable_time = None
        stability_required = 1.5  # Seconds of stable face detection before auto-capture
        last_face_position = None
        position_threshold = 30  # pixels - how much the face can move
        
        while time.time() - start_time < timeout:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Draw rectangles around faces
            display_frame = frame.copy()
            current_time = time.time()
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                x, y, w, h = largest_face
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Check if face position is stable
                if last_face_position is None:
                    last_face_position = (x, y, w, h)
                    face_stable_time = current_time
                    cv2.putText(display_frame, "Hold still...", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    lx, ly, lw, lh = last_face_position
                    # Check if face moved significantly
                    if abs(x - lx) < position_threshold and abs(y - ly) < position_threshold:
                        # Face is stable
                        time_stable = current_time - face_stable_time
                        if time_stable >= stability_required:
                            # Auto-capture!
                            face_roi = frame[y:y+h, x:x+w]
                            face_resized = cv2.resize(face_roi, (160, 160))
                            
                            self.stop_camera()
                            print("✓ Face captured successfully!")
                            return face_resized
                        else:
                            # Show countdown
                            remaining = int(stability_required - time_stable) + 1
                            cv2.putText(display_frame, f"Capturing in {remaining}...", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        # Face moved, reset timer
                        last_face_position = (x, y, w, h)
                        face_stable_time = current_time
                        cv2.putText(display_frame, "Hold still...", 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # No face detected, reset
                last_face_position = None
                face_stable_time = None
            
            remaining_time = int(timeout - (time.time() - start_time))
            cv2.putText(display_frame, f"Time remaining: {remaining_time}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "AUTO-CAPTURE - Hold still | Press 'q' to cancel", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Face Authentication', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
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

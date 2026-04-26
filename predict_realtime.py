import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from collections import deque
import joblib

# Initialize MediaPipe with new API (same config as data collection)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# For visualization - hand connections (same as data collection)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),     # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                              # Palm
]

# Configuration (same as data collection)
ROI_WIDTH = 400
ROI_HEIGHT = 400
PREDICTION_STABILITY_SIZE = 5

class RealTimePredictor:
    def __init__(self, model_path='model.pkl'):
        # Load model
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: {model_path} not found!")
            print("Please run train_model.py first.")
            exit(1)
        
        # Initialize MediaPipe with new API (same config as data collection)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        
        # For stable predictions
        self.prediction_history = deque(maxlen=PREDICTION_STABILITY_SIZE)
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        
        # Feature names for DataFrame
        self.feature_names = []
        for i in range(21):
            self.feature_names.extend([f'x{i}', f'y{i}'])
    
    def get_roi(self, frame):
        """Get the center ROI region (same as data collection)"""
        height, width = frame.shape[:2]
        x1 = (width - ROI_WIDTH) // 2
        y1 = (height - ROI_HEIGHT) // 2
        x2 = x1 + ROI_WIDTH
        y2 = y1 + ROI_HEIGHT
        return x1, y1, x2, y2
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks and connections on frame (same as data collection)"""
        if not landmarks:
            return frame
        
        h, w, _ = frame.shape
        
        # Draw landmarks as circles
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]
                
                start_x = int(start_lm.x * w)
                start_y = int(start_lm.y * h)
                end_x = int(end_lm.x * w)
                end_y = int(end_lm.y * h)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 0), 2)
        
        return frame
    
    def draw_roi(self, frame):
        """Draw ROI box on frame (same as data collection)"""
        x1, y1, x2, y2 = self.get_roi(frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame
    
    def extract_landmarks(self, frame):
        """Extract and normalize hand landmarks from full frame (same as data collection)"""
        # Convert full frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process with new MediaPipe API
        results = self.landmarker.detect(mp_image)
        
        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            
            # Extract (x, y) coordinates for all 21 landmarks
            features = []
            wrist_x = landmarks[0].x
            wrist_y = landmarks[0].y
            
            for lm in landmarks:
                # Normalize relative to wrist (same as data collection)
                norm_x = lm.x - wrist_x
                norm_y = lm.y - wrist_y
                features.extend([norm_x, norm_y])
            
            # Validate data (same as data collection)
            if any(abs(val) > 2 for val in features) or any(np.isnan(features)):
                return None, None
            
            return features, landmarks
        
        return None, None
    
    def get_stable_prediction(self, prediction):
        """Get stable prediction using majority vote"""
        self.prediction_history.append(prediction)
        
        if len(self.prediction_history) == PREDICTION_STABILITY_SIZE:
            # Get most frequent prediction
            from collections import Counter
            counter = Counter(self.prediction_history)
            most_common = counter.most_common(1)[0][0]
            
            # Only return if we have consensus (at least 3 out of 5)
            if counter[most_common] >= 3:
                return most_common
        
        return None
    
    def predict_letter(self, features):
        """Predict letter from features"""
        # Create DataFrame with proper feature names
        df = pd.DataFrame([features], columns=self.feature_names)
        
        try:
            prediction = self.model.predict(df)[0]
            return prediction
        except:
            return None
    
    def run(self):
        """Main prediction loop"""
        print("Starting ASL Real-Time Prediction")
        print("Press 'q' to quit")
        
        stable_prediction = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Draw ROI box for reference
            frame = self.draw_roi(frame)
            
            # Extract landmarks from full frame (same as data collection)
            features, landmarks = self.extract_landmarks(frame)
            
            # Draw hand landmarks and connections if detected
            if landmarks:
                frame = self.draw_landmarks(frame, landmarks)
            
            # Make prediction only if hand is detected
            current_prediction = None
            if features is not None:
                current_prediction = self.predict_letter(features)
                if current_prediction:
                    stable_prediction = self.get_stable_prediction(current_prediction)
            
            # Display results
            if landmarks:
                status = "HAND DETECTED"
                status_color = (0, 255, 0)
            else:
                status = "NO HAND DETECTED"
                status_color = (0, 0, 255)
                # Clear prediction history when no hand detected
                self.prediction_history.clear()
                stable_prediction = None
            
            # Draw UI text
            cv2.putText(frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Display stable prediction
            if stable_prediction:
                cv2.putText(frame, f"Prediction: {stable_prediction}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                
                # Draw confidence indicator (show history)
                confidence_text = f"Confidence: {self.prediction_history.count(stable_prediction)}/{len(self.prediction_history)}"
                cv2.putText(frame, confidence_text, (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Prediction: --", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 128, 128), 3)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('ASL Real-Time Prediction', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    predictor = RealTimePredictor()
    predictor.run()

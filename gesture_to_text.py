import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from collections import deque, Counter
import joblib
import time

# Initialize MediaPipe with new API (same config as existing system)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# For visualization - hand connections (same as existing system)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),     # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                              # Palm
]

# Configuration (same as existing system)
ROI_WIDTH = 400
ROI_HEIGHT = 400
PREDICTION_STABILITY_SIZE = 5
HOLD_DURATION = 3.0  # 3 seconds hold duration

class GestureToText:
    def __init__(self, model_path='model.pkl'):
        # Load model (same as existing system)
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: {model_path} not found!")
            print("Please run train_model.py first.")
            exit(1)
        
        # Initialize MediaPipe with new API (same config as existing system)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        
        # For stable predictions (same as existing system)
        self.prediction_history = deque(maxlen=PREDICTION_STABILITY_SIZE)
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        
        # Feature names for DataFrame (same as existing system)
        self.feature_names = []
        for i in range(21):
            self.feature_names.extend([f'x{i}', f'y{i}'])
        
        # Text building variables
        self.current_text = ""
        self.hold_start_time = None
        self.current_stable_letter = None
        self.last_confirmed_letter = None
        
    def get_roi(self, frame):
        """Get the center ROI region (same as existing system)"""
        height, width = frame.shape[:2]
        x1 = (width - ROI_WIDTH) // 2
        y1 = (height - ROI_HEIGHT) // 2
        x2 = x1 + ROI_WIDTH
        y2 = y1 + ROI_HEIGHT
        return x1, y1, x2, y2
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks and connections on frame (same as existing system)"""
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
        """Draw ROI box on frame (same as existing system)"""
        x1, y1, x2, y2 = self.get_roi(frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame
    
    def extract_landmarks(self, frame):
        """Extract and normalize hand landmarks from full frame (same as existing system)"""
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
                # Normalize relative to wrist (same as existing system)
                norm_x = lm.x - wrist_x
                norm_y = lm.y - wrist_y
                features.extend([norm_x, norm_y])
            
            # Validate data (same as existing system)
            if any(abs(val) > 2 for val in features) or any(np.isnan(features)):
                return None, None
            
            return features, landmarks
        
        return None, None
    
    def get_stable_prediction(self, prediction):
        """Get stable prediction using majority vote (same as existing system)"""
        self.prediction_history.append(prediction)
        
        if len(self.prediction_history) == PREDICTION_STABILITY_SIZE:
            # Get most frequent prediction
            counter = Counter(self.prediction_history)
            most_common = counter.most_common(1)[0][0]
            
            # Only return if we have consensus (at least 3 out of 5)
            if counter[most_common] >= 3:
                return most_common
        
        return None
    
    def predict_letter(self, features):
        """Predict letter from features (same as existing system)"""
        # Create DataFrame with proper feature names
        df = pd.DataFrame([features], columns=self.feature_names)
        
        try:
            prediction = self.model.predict(df)[0]
            return prediction
        except:
            return None
    
    def update_hold_timer(self, stable_letter):
        """Update hold timer and confirm letter if held for 3 seconds"""
        current_time = time.time()
        
        # If this is a new stable letter, start timer
        if self.current_stable_letter != stable_letter:
            self.current_stable_letter = stable_letter
            self.hold_start_time = current_time
            self.last_confirmed_letter = None  # Reset to allow same letter again
            return False
        
        # Check if we've held the same letter for 3 seconds
        if self.hold_start_time is not None:
            elapsed_time = current_time - self.hold_start_time
            if elapsed_time >= HOLD_DURATION:
                # Confirm the letter
                if self.last_confirmed_letter != stable_letter:
                    self.current_text += stable_letter
                    self.last_confirmed_letter = stable_letter
                    print(f"Confirmed: {stable_letter} -> Text: '{self.current_text}'")
                    return True
        
        return False
    
    def reset_hold_timer(self):
        """Reset hold timer when gesture changes or no hand detected"""
        self.current_stable_letter = None
        self.hold_start_time = None
    
    def save_text_to_file(self, filename="output.txt"):
        """Save current text to file in append mode"""
        try:
            with open(filename, 'a') as f:
                f.write(self.current_text + '\n')
            print(f"Text saved to {filename}: '{self.current_text}'")
            return True
        except Exception as e:
            print(f"Error saving to file: {e}")
            return False
    
    def draw_ui(self, frame, stable_letter, hand_detected):
        """Draw UI elements on frame"""
        height, width = frame.shape[:2]
        
        # Hand detection status
        if hand_detected:
            status = "HAND DETECTED"
            status_color = (0, 255, 0)
        else:
            status = "NO HAND DETECTED"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Current detected letter
        if stable_letter:
            cv2.putText(frame, f"Letter: {stable_letter}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
        
        # Hold timer
        if self.hold_start_time is not None and self.current_stable_letter:
            elapsed = time.time() - self.hold_start_time
            remaining = max(0, HOLD_DURATION - elapsed)
            timer_color = (0, 255, 0) if remaining == 0 else (0, 165, 255)
            cv2.putText(frame, f"Hold: {remaining:.1f}s", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, timer_color, 2)
        
        # Current text (large and clear)
        text_display = f"Text: {self.current_text}"
        text_y = height - 100
        cv2.putText(frame, text_display, (10, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Instructions
        instructions = [
            "Controls:",
            "g - add space",
            "s - save text", 
            "c - clear text",
            "q - quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (width - 250, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main gesture-to-text loop"""
        print("Starting ASL Gesture-to-Text")
        print("Hold a gesture for 3 seconds to confirm the letter")
        print("Controls: g=space, s=save, c=clear, q=quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Draw ROI box for reference
            frame = self.draw_roi(frame)
            
            # Extract landmarks from full frame (same as existing system)
            features, landmarks = self.extract_landmarks(frame)
            
            # Draw hand landmarks and connections if detected
            if landmarks:
                frame = self.draw_landmarks(frame, landmarks)
            
            # Make prediction only if hand is detected
            stable_letter = None
            if features is not None:
                current_prediction = self.predict_letter(features)
                if current_prediction:
                    stable_letter = self.get_stable_prediction(current_prediction)
            
            # Handle hold-to-confirm logic
            letter_confirmed = False
            if stable_letter:
                letter_confirmed = self.update_hold_timer(stable_letter)
            else:
                # No stable prediction, reset timer
                self.reset_hold_timer()
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                self.current_text += " "
                print(f"Added space -> Text: '{self.current_text}'")
                self.reset_hold_timer()  # Reset timer after action
            elif key == ord('s'):
                self.save_text_to_file()
                self.reset_hold_timer()  # Reset timer after action
            elif key == ord('c'):
                self.current_text = ""
                self.last_confirmed_letter = None
                print("Text cleared")
                self.reset_hold_timer()  # Reset timer after action
            
            # Draw UI
            frame = self.draw_ui(frame, stable_letter, landmarks is not None)
            
            # Show frame
            cv2.imshow('ASL Gesture-to-Text', frame)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_to_text = GestureToText()
    gesture_to_text.run()

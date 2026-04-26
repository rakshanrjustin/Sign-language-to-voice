import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from collections import deque

# Initialize MediaPipe with new API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# For visualization - hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),     # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                              # Palm
]

# Configuration
ROI_WIDTH = 400
ROI_HEIGHT = 400
SAMPLES_PER_LETTER = 30
LETTERS = [chr(ord('A') + i) for i in range(26)]

class DataCollector:
    def __init__(self):
        # Create HandLandmarker with new MediaPipe API
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)
        self.current_letter_index = 0
        self.samples_collected = 0
        self.data = []
        
    def get_roi(self, frame):
        """Get the center ROI region"""
        height, width = frame.shape[:2]
        x1 = (width - ROI_WIDTH) // 2
        y1 = (height - ROI_HEIGHT) // 2
        x2 = x1 + ROI_WIDTH
        y2 = y1 + ROI_HEIGHT
        return x1, y1, x2, y2
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks and connections on frame"""
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
        """Draw ROI box on frame"""
        x1, y1, x2, y2 = self.get_roi(frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return frame
    def extract_landmarks(self, frame):
        """Extract and normalize hand landmarks from full frame"""
        # Convert full frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process with new MediaPipe API
        results = self.landmarker.detect(mp_image)
        
        # Debug output
        if results.hand_landmarks:
            print("HAND DETECTED")
            landmarks = results.hand_landmarks[0]
            
            # Extract (x, y) coordinates for all 21 landmarks
            features = []
            wrist_x = landmarks[0].x
            wrist_y = landmarks[0].y
            
            for lm in landmarks:
                # Normalize relative to wrist
                norm_x = lm.x - wrist_x
                norm_y = lm.y - wrist_y
                features.extend([norm_x, norm_y])
            
            # Validate data
            if any(abs(val) > 2 for val in features) or any(np.isnan(features)):
                return None, None
            
            return features, landmarks
        else:
            print("NO HAND")
        
        return None, None
    
    def save_data(self):
        """Save collected data to CSV"""
        if not self.data:
            print("No data to save!")
            return
        
        df = pd.DataFrame(self.data)
        
        # Create column names
        columns = []
        for i in range(21):
            columns.extend([f'x{i}', f'y{i}'])
        columns.append('label')
        df.columns = columns
        
        df.to_csv('asl_dataset.csv', index=False)
        print(f"Saved {len(self.data)} samples to asl_dataset.csv")
    
    def run(self):
        """Main data collection loop"""
        print("Starting ASL Data Collection")
        print("Controls:")
        print("  's' - Save current sample")
        print("  'n' - Skip to next letter")
        print("  'q' - Quit and save")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Draw ROI box for reference
            frame = self.draw_roi(frame)
            
            # Extract landmarks from full frame (temporarily disabled ROI)
            features, landmarks = self.extract_landmarks(frame)
            
            # Draw hand landmarks and connections if detected
            if landmarks:
                frame = self.draw_landmarks(frame, landmarks)
            
            # Display status
            current_letter = LETTERS[self.current_letter_index]
            status = "HAND DETECTED" if landmarks else "NO HAND"
            
            # Draw UI text
            cv2.putText(frame, f"Collecting: {current_letter}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {self.samples_collected}/{SAMPLES_PER_LETTER}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, status, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if landmarks else (0, 0, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Press 's' to save, 'n' to skip, 'q' to quit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('ASL Data Collection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Move to next letter
                self.current_letter_index = (self.current_letter_index + 1) % len(LETTERS)
                self.samples_collected = 0
                print(f"Moved to letter: {LETTERS[self.current_letter_index]}")
            elif key == ord('s'):
                # Save current sample
                if features is not None:
                    features_with_label = features + [current_letter]
                    self.data.append(features_with_label)
                    self.samples_collected += 1
                    print(f"Saved sample {self.samples_collected}/{SAMPLES_PER_LETTER} for letter {current_letter}")
                    
                    # Auto-advance if collected enough samples
                    if self.samples_collected >= SAMPLES_PER_LETTER:
                        print(f"Completed collecting {current_letter}")
                        self.current_letter_index = (self.current_letter_index + 1) % len(LETTERS)
                        self.samples_collected = 0
                        print(f"Moving to letter: {LETTERS[self.current_letter_index]}")
                else:
                    print("No hand detected - cannot save sample")
        
        # Cleanup
        self.save_data()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()

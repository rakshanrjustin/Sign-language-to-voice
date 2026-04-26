# ASL Alphabet Recognition System

A complete Python project for real-time American Sign Language (A-Z) alphabet recognition using MediaPipe and machine learning.

## Features

- **Data Collection**: Guided collection system with ROI-based hand detection
- **Model Training**: KNN classifier with data validation and cleaning
- **Real-time Prediction**: Stable predictions with confidence scoring
- **No Flickering**: Uses deque-based prediction smoothing

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect Training Data
```bash
python data_collection.py
```
- Press 's' to save current sample
- Press 'n' to skip to next letter  
- Press 'q' to quit and save
- Collect exactly 30 samples per letter (A-Z)

### 2. Train Model
```bash
python train_model.py
```
- Loads and cleans the dataset
- Trains KNN classifier (n_neighbors=3)
- Saves model to `model.pkl`

### 3. Run Real-time Prediction
```bash
python predict_realtime.py
```
- Shows stable predictions with confidence
- Press 'q' to quit

## Technical Details

### Data Processing
- **ROI**: 400x400px center region
- **Normalization**: Wrist-relative coordinates
- **Features**: 21 landmarks (x, y) = 42 features
- **Validation**: Rejects NaN values and coordinates > 2

### Model
- **Algorithm**: K-Nearest Neighbors (k=3)
- **Features**: 42 normalized landmark coordinates
- **Output**: Letter prediction (A-Z)

### Prediction Stability
- Uses deque (size=5) for prediction history
- Majority vote for final prediction
- Requires 3/5 consensus for output
- Clears history when no hand detected

## Files

- `data_collection.py` - Dataset collection with guided interface
- `train_model.py` - Model training and validation
- `predict_realtime.py` - Real-time ASL recognition
- `requirements.txt` - Python dependencies
- `asl_dataset.csv` - Training data (generated)
- `model.pkl` - Trained model (generated)

## System Requirements

- Python 3.7+
- Webcam
- OpenCV, MediaPipe, scikit-learn, pandas, joblib, numpy

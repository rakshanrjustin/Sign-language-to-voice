import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        
    def load_and_clean_data(self, csv_file='asl_dataset.csv'):
        """Load and clean the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(csv_file)
        print(f"Initial dataset shape: {df.shape}")
        
        # Clean data: remove rows with any abs(value) > 2
        feature_columns = [col for col in df.columns if col != 'label']
        for col in feature_columns:
            df = df[abs(df[col]) <= 2]
        
        # Drop missing values
        df = df.dropna()
        print(f"Cleaned dataset shape: {df.shape}")
        
        return df
    
    def train_model(self, df):
        """Train the KNN model"""
        # Split features and labels
        feature_columns = [col for col in df.columns if col != 'label']
        X = df[feature_columns]
        y = df['label']
        
        print(f"Number of samples: {len(X)}")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Unique labels: {sorted(y.unique())}")
        
        # Split data for training/validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print("Training KNN model...")
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Show sample distribution
        print("\nSample distribution:")
        label_counts = y.value_counts().sort_index()
        for label, count in label_counts.items():
            print(f"  {label}: {count} samples")
        
        return train_accuracy, test_accuracy
    
    def save_model(self, filename='model.pkl'):
        """Save the trained model"""
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
    
    def run(self):
        """Main training pipeline"""
        try:
            # Load and clean data
            df = self.load_and_clean_data()
            
            if len(df) == 0:
                print("No valid data found after cleaning!")
                return
            
            # Train model
            train_acc, test_acc = self.train_model(df)
            
            # Save model
            self.save_model()
            
            print("\nTraining completed successfully!")
            print(f"Final model accuracy: {test_acc:.4f}")
            
        except FileNotFoundError:
            print("Error: asl_dataset.csv not found!")
            print("Please run data_collection.py first to collect data.")
        except Exception as e:
            print(f"Error during training: {e}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()

#!/usr/bin/env python3
"""Train LBPH face recognition model."""

import cv2
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

class FaceModelTrainer:
    """Trains LBPH face recognition model from collected samples."""
    
    def __init__(self):
        """Initialize face model trainer."""
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=config.RECOGNITION_RADIUS,
            neighbors=config.RECOGNITION_NEIGHBORS,
            grid_x=config.RECOGNITION_GRID_X,
            grid_y=config.RECOGNITION_GRID_Y
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + config.HAAR_CASCADE_PATH
        )
        
        if self.face_cascade.empty():
            raise IOError("Failed to load Haar Cascade classifier")
    
    def load_training_data(self):
        """Load face images and labels from dataset directory.
        
        Returns:
            tuple: (faces, labels, label_names) where:
                - faces: list of face images
                - labels: list of numeric labels
                - label_names: dict mapping label IDs to person names
        """
        faces = []
        labels = []
        label_names = {}
        current_label = 0
        
        if not os.path.exists(config.DATASET_DIR):
            raise IOError(f"Dataset directory not found: {config.DATASET_DIR}")
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(config.DATASET_DIR)
                      if os.path.isdir(os.path.join(config.DATASET_DIR, d))]
        
        if not person_dirs:
            raise ValueError("No training data found. Please collect face samples first.")
        
        print(f"\nLoading training data from {len(person_dirs)} persons...")
        
        for person_name in sorted(person_dirs):
            person_path = os.path.join(config.DATASET_DIR, person_name)
            image_files = [f for f in os.listdir(person_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(image_files) < config.TRAIN_MIN_SAMPLES:
                print(f"Warning: {person_name} has only {len(image_files)} samples "
                     f"(minimum recommended: {config.TRAIN_MIN_SAMPLES})")
            
            label_names[current_label] = person_name
            
            print(f"  Loading {len(image_files)} images for: {person_name} (label {current_label})")
            
            for image_file in image_files:
                image_path = os.path.join(person_path, image_file)
                
                try:
                    # Load image
                    pil_image = Image.open(image_path).convert('L')  # Convert to grayscale
                    image_array = np.array(pil_image, 'uint8')
                    
                    # Add to training data
                    faces.append(image_array)
                    labels.append(current_label)
                    
                except Exception as e:
                    print(f"    Error loading {image_file}: {str(e)}")
            
            current_label += 1
        
        print(f"\nTotal training samples loaded: {len(faces)}")
        print(f"Total unique persons: {len(label_names)}")
        
        return faces, labels, label_names
    
    def train(self, save_model=True):
        """Train the face recognition model.
        
        Args:
            save_model: Whether to save the trained model
        """
        print("="*60)
        print("Starting Face Recognition Model Training")
        print("="*60)
        
        # Load training data
        faces, labels, label_names = self.load_training_data()
        
        if len(faces) == 0:
            raise ValueError("No training data loaded!")
        
        # Train the model
        print("\nTraining LBPH face recognizer...")
        print(f"Model parameters:")
        print(f"  - Radius: {config.RECOGNITION_RADIUS}")
        print(f"  - Neighbors: {config.RECOGNITION_NEIGHBORS}")
        print(f"  - Grid X: {config.RECOGNITION_GRID_X}")
        print(f"  - Grid Y: {config.RECOGNITION_GRID_Y}")
        
        self.recognizer.train(faces, np.array(labels))
        
        print("\nTraining completed successfully!")
        
        # Save the model
        if save_model:
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            
            # Save recognizer model
            print(f"\nSaving model to: {config.MODEL_FILE}")
            self.recognizer.save(config.MODEL_FILE)
            
            # Save label names
            print(f"Saving labels to: {config.LABEL_FILE}")
            with open(config.LABEL_FILE, 'wb') as f:
                pickle.dump(label_names, f)
            
            print("\nModel and labels saved successfully!")
            print(f"\nModel details:")
            print(f"  - Trained on {len(faces)} images")
            print(f"  - {len(label_names)} unique persons:")
            for label_id, name in label_names.items():
                count = labels.count(label_id)
                print(f"    [{label_id}] {name}: {count} samples")
        
        return self.recognizer, label_names
    
    def evaluate(self, test_ratio=0.2):
        """Evaluate model performance (optional advanced feature).
        
        Args:
            test_ratio: Ratio of data to use for testing
        """
        # This is a placeholder for future implementation
        # Could implement train-test split and accuracy calculation
        print("\nNote: Model evaluation not yet implemented.")
        print("For best results, test manually with real-time recognition.")

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train LBPH face recognition model'
    )
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the trained model')
    
    args = parser.parse_args()
    
    try:
        trainer = FaceModelTrainer()
        trainer.train(save_model=not args.no_save)
        
        print("\n" + "="*60)
        print("Training process completed!")
        print("You can now run face recognition using the trained model.")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()

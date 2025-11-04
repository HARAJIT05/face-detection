#!/usr/bin/env python3
"""Face data collection script using webcam."""

import cv2
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

class FaceDataCollector:
    """Collects face samples from webcam for training."""
    
    def __init__(self, camera_index=0):
        """Initialize face data collector.
        
        Args:
            camera_index: Index of camera device to use
        """
        self.camera_index = camera_index
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + config.HAAR_CASCADE_PATH
        )
        
        if self.face_cascade.empty():
            raise IOError("Failed to load Haar Cascade classifier")
    
    def collect_samples(self, person_name, num_samples=100):
        """Collect face samples for a person.
        
        Args:
            person_name: Name of the person
            num_samples: Number of face samples to collect
        """
        # Create directory for this person
        person_dir = os.path.join(config.DATASET_DIR, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Check existing samples
        existing_files = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
        start_index = len(existing_files)
        
        print(f"\nCollecting {num_samples} face samples for: {person_name}")
        print(f"Existing samples: {start_index}")
        print(f"Target samples: {start_index + num_samples}")
        print("\nInstructions:")
        print("- Position your face in front of the camera")
        print("- Move your head slightly for different angles")
        print("- Vary expressions and lighting if possible")
        print("- Press 'q' to quit early\n")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.COLLECT_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.COLLECT_FRAME_HEIGHT)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open camera {self.camera_index}")
        
        count = 0
        saved_count = 0
        cooldown = 0  # Cooldown counter to avoid capturing too fast
        
        print("Starting capture in 3 seconds...")
        time.sleep(3)
        
        while saved_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=config.DETECTION_SCALE_FACTOR,
                minNeighbors=config.DETECTION_MIN_NEIGHBORS,
                minSize=config.DETECTION_MIN_SIZE
            )
            
            # Process detected faces
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save face image with cooldown
                if cooldown == 0 and saved_count < num_samples:
                    # Extract and resize face
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, config.COLLECT_FACE_SIZE)
                    
                    # Save image
                    filename = f"{person_name}_{start_index + saved_count + 1:04d}.jpg"
                    filepath = os.path.join(person_dir, filename)
                    cv2.imwrite(filepath, face_resized)
                    
                    saved_count += 1
                    cooldown = 5  # Wait 5 frames before next capture
                    
                    print(f"Captured: {saved_count}/{num_samples} samples", end='\r')
            
            if cooldown > 0:
                cooldown -= 1
            
            # Display progress
            progress_text = f"Samples: {saved_count}/{num_samples}"
            cv2.putText(frame, progress_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if len(faces) == 0:
                cv2.putText(frame, "No face detected!", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Collecting Face Data - Press Q to quit', frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nCollection interrupted by user")
                break
            
            count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n\nCollection complete!")
        print(f"Total samples saved: {saved_count}")
        print(f"Saved to: {person_dir}")
        
        return saved_count

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect face data using webcam')
    parser.add_argument('--name', type=str, required=True,
                       help='Name of the person')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to collect (default: 100)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    
    args = parser.parse_args()
    
    try:
        collector = FaceDataCollector(camera_index=args.camera)
        collector.collect_samples(args.name, args.samples)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()

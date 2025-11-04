#!/usr/bin/env python3
"""Complete face detection and recognition pipeline."""

import cv2
import os
import sys
import pickle
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

class FaceRecognitionPipeline:
    """Complete pipeline for face detection and recognition."""
    
    def __init__(self, camera_index=0, video_source=None):
        """Initialize face recognition pipeline.
        
        Args:
            camera_index: Index of camera device (if using webcam)
            video_source: Path to video file (if using video file)
        """
        self.camera_index = camera_index
        self.video_source = video_source
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + config.HAAR_CASCADE_PATH
        )
        
        if self.face_cascade.empty():
            raise IOError("Failed to load Haar Cascade classifier")
        
        # Initialize face recognizer
        self.recognizer = None
        self.label_names = None
        self.load_model()
    
    def load_model(self):
        """Load trained face recognition model and labels."""
        if not os.path.exists(config.MODEL_FILE):
            raise FileNotFoundError(
                f"Model file not found: {config.MODEL_FILE}\n"
                "Please train the model first using train_model.py"
            )
        
        if not os.path.exists(config.LABEL_FILE):
            raise FileNotFoundError(
                f"Label file not found: {config.LABEL_FILE}\n"
                "Please train the model first using train_model.py"
            )
        
        print("Loading trained model...")
        
        # Load recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(config.MODEL_FILE)
        
        # Load label names
        with open(config.LABEL_FILE, 'rb') as f:
            self.label_names = pickle.load(f)
        
        print(f"Model loaded successfully!")
        print(f"Recognized persons: {list(self.label_names.values())}")
    
    def detect_faces(self, frame):
        """Detect faces in a frame.
        
        Args:
            frame: Input image frame
        
        Returns:
            list: List of (x, y, w, h) tuples for detected faces
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=config.DETECTION_SCALE_FACTOR,
            minNeighbors=config.DETECTION_MIN_NEIGHBORS,
            minSize=config.DETECTION_MIN_SIZE
        )
        
        return faces, gray
    
    def recognize_face(self, face_roi):
        """Recognize a face.
        
        Args:
            face_roi: Grayscale face region of interest
        
        Returns:
            tuple: (label, confidence) or (None, None) if recognition fails
        """
        if self.recognizer is None:
            return None, None
        
        # Resize face to match training size
        face_resized = cv2.resize(face_roi, config.COLLECT_FACE_SIZE)
        
        # Predict
        label, confidence = self.recognizer.predict(face_resized)
        
        return label, confidence
    
    def draw_results(self, frame, faces, gray):
        """Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input image frame
            faces: List of detected face coordinates
            gray: Grayscale version of frame
        
        Returns:
            frame: Annotated frame
        """
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Recognize face
            label, confidence = self.recognize_face(face_roi)
            
            # Determine if recognized or unknown
            if label is not None and confidence < config.RECOGNITION_THRESHOLD:
                name = self.label_names.get(label, "Unknown")
                color = config.DISPLAY_BOX_COLOR
                text = f"{name} ({confidence:.1f})"
            else:
                name = "Unknown"
                color = config.DISPLAY_UNKNOWN_COLOR
                text = f"Unknown ({confidence:.1f})" if label is not None else "Unknown"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label background
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.rectangle(frame, (x, label_y - 20), (x + w, label_y), color, -1)
            
            # Draw label text
            cv2.putText(
                frame, text, (x + 5, label_y - 5),
                config.DISPLAY_FONT,
                config.DISPLAY_FONT_SCALE * 0.5,
                (0, 0, 0),
                1
            )
        
        return frame
    
    def run(self, save_output=False):
        """Run the face recognition pipeline.
        
        Args:
            save_output: Whether to save output video
        """
        print("\n" + "="*60)
        print("Starting Face Recognition Pipeline")
        print("="*60)
        print("\nControls:")
        print("  Press 'q' or 'Q' to quit")
        print("  Press 's' or 'S' to save screenshot")
        print("\n")
        
        # Initialize video capture
        if self.video_source:
            cap = cv2.VideoCapture(self.video_source)
            print(f"Reading from video file: {self.video_source}")
        else:
            cap = cv2.VideoCapture(self.camera_index)
            print(f"Reading from camera: {self.camera_index}")
        
        if not cap.isOpened():
            raise IOError("Cannot open video source")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or config.CAMERA_FPS
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if saving
        writer = None
        if save_output:
            output_path = os.path.join(
                config.OUTPUT_DIR,
                f"output_{int(time.time())}.avi"
            )
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(
                output_path, fourcc, fps,
                (frame_width, frame_height)
            )
            print(f"Saving output to: {output_path}")
        
        frame_count = 0
        face_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video stream")
                    break
                
                frame_count += 1
                
                # Detect and recognize faces
                faces, gray = self.detect_faces(frame)
                face_count += len(faces)
                
                # Draw results
                frame = self.draw_results(frame, faces, gray)
                
                # Display FPS
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                cv2.putText(
                    frame, f"FPS: {current_fps:.1f}", (10, 30),
                    config.DISPLAY_FONT, 0.7, (0, 255, 0), 2
                )
                
                # Display face count
                cv2.putText(
                    frame, f"Faces: {len(faces)}", (10, 60),
                    config.DISPLAY_FONT, 0.7, (0, 255, 0), 2
                )
                
                # Show frame
                cv2.imshow(config.DISPLAY_WINDOW_NAME, frame)
                
                # Write frame if saving
                if writer:
                    writer.write(frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s') or key == ord('S'):
                    screenshot_path = os.path.join(
                        config.OUTPUT_DIR,
                        f"screenshot_{int(time.time())}.jpg"
                    )
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            elapsed_time = time.time() - start_time
            
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            print("\n" + "="*60)
            print("Pipeline Statistics")
            print("="*60)
            print(f"Total frames processed: {frame_count}")
            print(f"Total faces detected: {face_count}")
            print(f"Average FPS: {frame_count / elapsed_time:.2f}")
            print(f"Total time: {elapsed_time:.2f} seconds")
            print("="*60)

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run face recognition pipeline'
    )
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--source', type=str,
                       help='Video file path (default: use webcam)')
    parser.add_argument('--save', action='store_true',
                       help='Save output video')
    
    args = parser.parse_args()
    
    try:
        pipeline = FaceRecognitionPipeline(
            camera_index=args.camera,
            video_source=args.source
        )
        pipeline.run(save_output=args.save)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()

# Face Detection and Recognition Pipeline

## Overview
This project implements a complete face detection and recognition pipeline using OpenCV and the LBPH (Local Binary Patterns Histograms) face recognition algorithm. The system can collect face samples, train a recognition model, and perform real-time face detection and recognition.

## Features
- Real-time face detection using OpenCV's Haar Cascade classifier
- Face data collection via webcam
- LBPH-based face recognition model training
- Complete pipeline for detection and recognition
- Configurable settings for model parameters and paths

## Project Structure
```
face-detection/
├── scripts/
│   ├── collect_data.py      # Webcam face sample collection
│   ├── train_model.py       # LBPH model training
│   └── face_pipeline.py     # Complete detection + recognition pipeline
├── dataset/                  # Collected face samples (organized by person)
├── models/                   # Trained face recognition models
├── output/                   # Output images and results
├── test_images/             # Test images for evaluation
├── config.py                # Configuration settings
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.7+
- Webcam (for data collection)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/HARAJIT05/face-detection.git
cd face-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect Face Data
Collect face samples for training:
```bash
python scripts/collect_data.py --name "PersonName" --samples 100
```
- `--name`: Name of the person
- `--samples`: Number of face samples to collect (default: 100)

### 2. Train the Model
Train the LBPH face recognition model:
```bash
python scripts/train_model.py
```
This will create a trained model in the `models/` directory.

### 3. Run Face Recognition
Run the complete face recognition pipeline:
```bash
python scripts/face_pipeline.py
```
Press 'q' to quit the video stream.

### 4. Using the CLI (Optional)
```bash
python main.py --mode collect --name "PersonName"
python main.py --mode train
python main.py --mode recognize
```

## Configuration
Edit `config.py` to customize:
- Dataset and model paths
- Haar Cascade classifier path
- Model parameters (neighbors, threshold, grid size)
- Camera settings

## Dependencies
- OpenCV (cv2) - Computer vision library
- NumPy - Numerical computing
- Pillow - Image processing
- imutils - Convenience functions for OpenCV

## How It Works

### Face Detection
Uses OpenCV's Haar Cascade classifier to detect faces in images/video frames.

### Face Recognition
Implements the LBPH (Local Binary Patterns Histograms) algorithm:
1. Divides face images into small regions
2. Extracts LBP features from each region
3. Creates histograms of the features
4. Compares histograms to recognize faces

### Pipeline
1. **Data Collection**: Captures face samples using webcam
2. **Training**: Trains LBPH recognizer on collected samples
3. **Recognition**: Detects faces and identifies them using the trained model

## Troubleshooting
- **No faces detected**: Ensure good lighting and face the camera directly
- **Low accuracy**: Collect more diverse samples (different angles, lighting)
- **Camera not found**: Check camera permissions and device index in config

## Future Improvements
- Support for multiple face detection models (DNN, MTCNN)
- Deep learning-based recognition (FaceNet, ArcFace)
- GUI interface
- REST API for face recognition service
- Face anti-spoofing

## License
MIT License

## Author
HARAJIT05

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

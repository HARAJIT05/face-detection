"""Configuration settings for face detection and recognition pipeline."""
import os

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
TEST_IMAGES_DIR = os.path.join(BASE_DIR, 'test_images')

# Haar Cascade Classifier Path
# OpenCV provides pre-trained Haar Cascade classifiers
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Model Files
MODEL_FILE = os.path.join(MODELS_DIR, 'face_recognizer.yml')
LABEL_FILE = os.path.join(MODELS_DIR, 'labels.pkl')

# Face Detection Parameters
DETECTION_SCALE_FACTOR = 1.3  # How much the image size is reduced at each image scale
DETECTION_MIN_NEIGHBORS = 5  # How many neighbors each candidate rectangle should have
DETECTION_MIN_SIZE = (30, 30)  # Minimum possible object size

# Face Recognition Parameters
RECOGNITION_THRESHOLD = 80  # Confidence threshold (lower is more confident)
RECOGNITION_RADIUS = 1  # LBPH radius parameter
RECOGNITION_NEIGHBORS = 8  # LBPH neighbors parameter
RECOGNITION_GRID_X = 8  # LBPH grid X parameter
RECOGNITION_GRID_Y = 8  # LBPH grid Y parameter

# Data Collection Settings
COLLECT_SAMPLES_COUNT = 100  # Default number of samples to collect per person
COLLECT_FRAME_WIDTH = 640  # Width of the video capture frame
COLLECT_FRAME_HEIGHT = 480  # Height of the video capture frame
COLLECT_FACE_SIZE = (200, 200)  # Size to resize collected face images

# Camera Settings
CAMERA_INDEX = 0  # Default camera device index (0 for primary camera)
CAMERA_FPS = 30  # Frames per second for video capture

# Display Settings
DISPLAY_WINDOW_NAME = 'Face Recognition Pipeline'
DISPLAY_FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
DISPLAY_FONT_SCALE = 0.9
DISPLAY_FONT_THICKNESS = 2
DISPLAY_BOX_COLOR = (0, 255, 0)  # Green color for bounding box (BGR)
DISPLAY_TEXT_COLOR = (255, 255, 255)  # White color for text (BGR)
DISPLAY_UNKNOWN_COLOR = (0, 0, 255)  # Red color for unknown faces (BGR)

# Training Settings
TRAIN_TEST_SPLIT = 0.2  # Percentage of data to use for testing
TRAIN_MIN_SAMPLES = 10  # Minimum samples required per person for training

# Logging Settings
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create directories if they don't exist
for directory in [DATASET_DIR, MODELS_DIR, OUTPUT_DIR, TEST_IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)

print(f"Configuration loaded successfully")
print(f"Dataset directory: {DATASET_DIR}")
print(f"Models directory: {MODELS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

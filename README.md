# Vehicle Detection, Tracking, and Classification (YOLO)

This project implements a vehicle detection, tracking, counting, and classification system for videos or webcam streams using Computer Vision with YOLO models, OpenCV, CLIP embeddings, and Python.

## Features
The system detects vehicles, assigns temporary IDs, tracks their movement across frames, classifies detected objects, and counts how many vehicles enter and exit a scene by crossing predefined virtual lines. It also estimates vehicle speed using radar-like calculations.
- Real-time vehicle detection and tracking
- Object classification using YOLO classification models
- Image embeddings using CLIP for potential re-identification
- Speed estimation (radar functionality)
- Counting of vehicles entering and exiting
- Support for webcam and local video files
- Real-time visualization with bounding boxes, IDs, and counters

## Files
- [main.py](main.py): Main script for vehicle detection, tracking, counting, classification, and speed estimation.
- [classifier.py](classifier.py): Handles object classification using YOLO classification models.
- [config.py](config.py): Configuration file with constants for video source, frame dimensions, thresholds, and vehicle keywords.
- [database.py](database.py): Handles database operations for storing and retrieving vector embeddings using SQLite.
- [embedding.py](embedding.py): Generates image embeddings using CLIP for detected objects.
- [tracker.py](tracker.py): Centroid-based object tracker for maintaining object IDs across frames.
- [utils.py](utils.py): Utility functions, including bounding box merging.
- [resetdb.py](resetdb.py): Script to reset the vector database.
- [ver_vetores.py](ver_vetores.py): Script to view and analyze stored vector embeddings.

## Requirements
- Python 3.8+
- PyTorch (CPU or CUDA build depending on your machine)
- ultralytics
- opencv-python
- numpy
- pillow (PIL)
- clip-by-openai
- scipy

Install dependencies with:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install ultralytics opencv-python numpy torch pillow git+https://github.com/openai/CLIP.git scipy
```

Note: Install the correct `torch` wheel for your CUDA version if you want GPU acceleration. See https://pytorch.org for instructions.

## Models
Place your model weights in the project root (same folder as the scripts). The project currently uses:

Classification model (used by classifier.py):
- [yolo11m-cls.pt](yolo11m-cls.pt): YOLOv11 medium classifier

Other models available in the project (not used by default, can be swapped in config):
- yolov8n.pt, yolov8s.pt, yolov8m.pt: YOLOv8 detection models (nano, small, medium)
- yolov10n.pt, yolov10s.pt, yolov10m.pt: YOLOv10 detection models
- yolo26n.pt: YOLOv2.6 nano detection
- yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt: YOLOv8 classification models
- yolo26n-cls.pt: YOLOv2.6 nano classifier

## Usage
Run the main vehicle detection and tracking script:

```bash
python main.py
```

To reset the vector database:

```bash
python resetdb.py
```

To view and analyze stored vector embeddings:

```bash
python ver_vetores.py
```

The scripts use the video source specified in `config.py` (default: "videstrada.mp4"). Modify `config.py` to change settings like frame size, thresholds, and line positions.

## Notes
- Ensure model files are in the project root.
- For GPU acceleration, install PyTorch with CUDA support.
- The system uses background subtraction for motion detection and centroid tracking for object persistence.
- Speed estimation requires calibrated distance settings in `config.py`.

## Processing Flow

### Initialization
1. Opens the SQLite database and loads all embeddings from previous runs into memory, along with the source video of each one
2. Loads the CLIP model (for embeddings) and the YOLO classification model
3. Opens the video source and validates that it is accessible
4. Creates the background subtractor and the centroid tracker

### Frame-by-frame processing
1. Applies background subtraction on the grayscale frame to detect motion
2. Filters the foreground mask with blur, threshold, and morphological operations
3. Extracts contours and generates bounding boxes, discarding those below the minimum area
4. Merges nearby bounding boxes that likely belong to the same vehicle (merge boxes)
5. The centroid tracker associates each detection to an existing ID or registers a new one

### When a vehicle crosses a trigger line (LINE1_Y or LINE2_Y)
1. Classifies the object using YOLO (vehicle or non-vehicle)
2. Crops the vehicle image and generates a CLIP embedding
3. Compares the embedding against all stored embeddings in memory
4. If a similar embedding from a **different video** is found, logs the match (informational only — no IDs are replaced and no data is deleted)
5. The embedding is **always** saved to the in-memory dictionary and to the database, regardless of whether a match was found

### Counting (LINE_Y)
- If the vehicle crosses the central line downward, it is counted as entering
- If it crosses upward, it is counted as exiting
- Each vehicle is counted only once

### Speed estimation (LINE1_Y and LINE2_Y)
- When a vehicle crosses the first radar line, the timestamp is recorded
- When it crosses the second radar line, the time difference is used to estimate speed based on the configured real-world distance (`DISTANCIA_METROS`)
- The speed is logged and displayed on screen

### Shutdown
1. Releases the video capture and closes the database connection
2. Logs a summary with the total count of vehicles entering and exiting

## License
This repository does not include a license file. Add one if you plan to share the code publicly.

---
Created automatically to document this project.

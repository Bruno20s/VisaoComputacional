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
- [testecv.py](testecv.py): Test script for computer vision functionalities.
- [ver_vetores.py](ver_vetores.py): Script to view and analyze stored vector embeddings.
- saida_ids.csv: Output CSV file containing tracked vehicle IDs.

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
pip install ultralytics opencv-python numpy torch pillow git+https://github.com/openai/CLIP.git scipy
```

Note: Install the correct `torch` wheel for your CUDA version if you want GPU acceleration. See https://pytorch.org for instructions.

## Models
Place your model weights in the project root (same folder as the scripts). The project uses various YOLO models for detection and classification:

Detection models:
- [yolov8n.pt](yolov8n.pt): YOLOv8 nano (fast, less accurate)
- [yolov8s.pt](yolov8s.pt): YOLOv8 small
- [yolov8m.pt](yolov8m.pt): YOLOv8 medium (higher accuracy)
- [yolov10n.pt](yolov10n.pt): YOLOv10 nano
- [yolov10s.pt](yolov10s.pt): YOLOv10 small
- [yolov10m.pt](yolov10m.pt): YOLOv10 medium
- [yolo26n.pt](yolo26n.pt): YOLOv2.6 nano

Classification models:
- [yolov8n-cls.pt](yolov8n-cls.pt): YOLOv8 nano classifier
- [yolov8s-cls.pt](yolov8s-cls.pt): YOLOv8 small classifier
- [yolov8m-cls.pt](yolov8m-cls.pt): YOLOv8 medium classifier
- [yolo11m-cls.pt](yolo11m-cls.pt): YOLOv11 medium classifier
- [yolo26n-cls.pt](yolo26n-cls.pt): YOLOv2.6 nano classifier

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

To run the test script:

```bash
python testecv.py
```

The scripts use the video source specified in `config.py` (default: "videstrada.mp4"). Modify `config.py` to change settings like frame size, thresholds, and line positions.

## Notes
- Ensure model files are in the project root.
- For GPU acceleration, install PyTorch with CUDA support.
- The system uses background subtraction for motion detection and centroid tracking for object persistence.
- Speed estimation requires calibrated distance settings in `config.py`.

## License
This repository does not include a license file. Add one if you plan to share the code publicly.

---
Created automatically to document this project.

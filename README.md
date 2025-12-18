# Car & Face Detection (YOLOv8)

This project implements a vehicle detection, tracking, and counting system for videos or webcam streams using Computer Vision with YOLOv8, OpenCV, and Python.

## Features
The system detects vehicles, assigns temporary IDs, tracks their movement across frames, and counts how many vehicles enter and exit a scene by crossing a predefined virtual line.
- Real-time vehicle detection
- Simple distance-based tracking
- Temporary ID assignment for vehicles

Counting of:
- vehicles that enter
- vehicles that exit

Support for:
- Webcam
- Local video files
- Real-time visualization with bounding boxes and IDs

## Files
- [car_counter.py](car_counter.py): Counts/detects cars (example script).
- [detectar_rostos.py](detectar_rostos.py): Detects faces (example script).
- [yolov8m.pt](yolov8m.pt): YOLOv8 medium model (optional, higher accuracy).
- [yolov8n.pt](yolov8n.pt): YOLOv8 nano model (optional, faster/inference-on-edge).

## Requirements
- Python 3.8+
- PyTorch (CPU or CUDA build depending on your machine)
- ultralytics
- opencv-python
- numpy

Install common dependencies with:

```bash
python -m pip install ultralytics opencv-python numpy torch
```

Note: Install the correct `torch` wheel for your CUDA version if you want GPU acceleration. See https://pytorch.org for instructions.

## Models
Place your model weights (`yolov8m.pt` or `yolov8n.pt`) in the project root (same folder as the scripts). Use `yolov8n.pt` for faster but less accurate results and `yolov8m.pt` for better accuracy.

## Usage
Basic usage examples — run from the project root:

```bash
python car_counter.py
python detectar_rostos.py
```

If the scripts accept command-line options (source video, camera index, etc.), pass them as you normally would. Eg:

```bash
python car_counter.py --source 0
```

## Notes
- This README assumes the scripts are self-contained and load model weights from the project root.
- If you see errors related to `torch` or CUDA, verify your `torch` installation matches your system GPU/drivers.

## License
This repository does not include a license file. Add one if you plan to share the code publicly.

---
Created automatically to document this small project.

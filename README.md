# Multi-Camera People Counting

Real-time image stitching and people counting using 2 cameras, SIFT, Homography, RANSAC, and YOLO.

## Overview

This project captures frames from two fixed USB cameras, stitches them into a single panorama, and performs YOLO-based person detection on the stitched output.

The system uses SIFT feature extraction, KNN matching with Lowe Ratio Test, homography estimation with RANSAC, and real-time people counting on the final stitched frame.

## Features

- Real-time image stitching from 2 cameras
- SIFT feature extraction
- KNN matching with Lowe Ratio Test
- Homography estimation using RANSAC
- Precomputed homography for faster runtime
- YOLO-based person detection
- Real-time people counting on panorama output
- FPS display during runtime

## Technologies Used

- Python
- OpenCV
- NumPy
- Ultralytics YOLO

## Project Files

- `main.py`: improved main version of the project with image stitching and YOLO people counting
- `requirements.txt`: required Python libraries

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Run the Project

Run the project with:

```bash
python main.py --cameras 0 1
```

Optional example:

```bash
python main.py --cameras 0 1 --yolo-model yolov8n.pt --person-conf 0.35 --detect-every 3 --debug
```

## Parameters

- `--cameras`: camera indices, for example `0 1`
- `--debug`: enable debug output
- `--yolo-model`: YOLO model file, default is `yolov8n.pt`
- `--person-conf`: confidence threshold for person detection
- `--detect-every`: run YOLO every N stitched frames

## How It Works

1. Capture frames from 2 cameras.
2. Detect and match SIFT features.
3. Use KNN + Lowe Ratio Test for better feature matching.
4. Estimate homography using RANSAC.
5. Stitch both camera views into one panorama.
6. Run YOLO on the stitched frame.
7. Count detected people and display the result in real time.

## Requirements

- 2 USB cameras
- Overlapping field of view between cameras
- Python 3.x
- Stable lighting for better stitching quality

## Current Limitations

- Works best when cameras are fixed
- Performance depends on hardware capability
- Stitching quality depends on overlap and scene texture

## Future Improvements

- Better blending between camera views
- More stable people tracking
- Multi-camera extension beyond 2 cameras
- Improved performance optimization

## Author

Nguyen Van Khang

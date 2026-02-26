# computer_vision

This ROS2 package provides comprehensive computer vision capabilities for autonomous search and rescue robots competing in the RoboCup Rescue Robot League. The system enables robots to detect, identify, and localize victims, hazardous materials, navigation markers, and objects in disaster environments.

## Overview

The RoboCup Rescue Robot League challenges teams to develop robots capable of operating in simulated disaster scenarios, assisting first responders in locating victims and mapping hazardous environments. This package addresses critical computer vision tasks including:

- **Object Detection**: Identifying victims, debris, and mission-critical objects using YOLO-based deep learning
- **Hazmat Detection**: Recognizing hazardous material labels and placards for safety assessment
- **QR Code Reading**: Decoding victim identification and navigation markers
- **AprilTag Detection**: Precise localization using fiducial markers
- **3D Coordinate Mapping**: Transforming 2D detections to 3D world coordinates using depth data and TF transforms
- **Data Logging**: Comprehensive CSV logging for post-mission analysis and scoring

## System Architecture

### Core Nodes

#### 1. **Detector Node** (`detector.py`)
- Subscribes to camera color and depth streams
- Runs YOLO models for object and hazmat detection
- Processes QR codes and AprilTags
- Publishes Detection messages with bounding boxes and depth
- Mode-aware: adjusts processing based on operational mode

#### 2. **Detection Manager Node** (`detection_manager_node.py`)
- Transforms detections from camera frame to target frame (default: `odom`)
- Manages operational modes (OFF, INITIALIZING, SENSOR_CRATE, MAPPING)
- Logs detections to CSV for competition scoring
- Provides services for mode control and status queries
- Publishes text messages for QR/Hazmat/AprilTag data

## Operational Modes

| Mode | Value | Description | Use Case |
|------|-------|-------------|----------|
| **OFF** | 0 | All processing disabled | System standby |
| **INITIALIZING** | 1 | Model loading, TF checks, CSV setup | Pre-mission validation |
| **SENSOR_CRATE** | 2 | QR/Hazmat/AprilTag only | Teleoperation, inspection |
| **MAPPING** | 3 | Full detection + coordinate logging | Autonomous exploration |

## Installation

### Prerequisites

```bash
# ROS2 Humble
sudo apt install ros-humble-desktop

# RealSense SDK and apriltag packages
sudo apt install ros-humble-realsense2-camera ros-humble-apriltag-ros

# Python dependencies
pip install ultralytics opencv-contrib-python pyzbar
```

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select computer_vision reseq_interfaces
source install/setup.bash
```

## Usage

### Launch System

Start the complete computer vision pipeline with RealSense camera:

```bash
# Start in SENSOR_CRATE mode (QR/Hazmat/AprilTag only)
ros2 launch computer_vision cv_launch.py mode:=2

# Start in MAPPING mode (full detection + logging)
ros2 launch computer_vision cv_launch.py mode:=3
```

### View Detection Output

```bash
# Visualization with bounding boxes
ros2 run rqt_image_view rqt_image_view /detector/model_output

# Detection messages
ros2 topic echo /object_detection/detections
```

## Topics

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/detection/mode` | `std_msgs/UInt8` | Current operational mode |
| `/object_detection/detections` | `reseq_interfaces/Detection` | All detections with bbox, depth, coords |
| `/detector/model_output` | `sensor_msgs/Image` | Visualization with drawn bounding boxes |
| `/qr_text` | `std_msgs/String` | QR code content |
| `/hazmat_text` | `std_msgs/String` | Hazmat label text |
| `/apriltag_text` | `std_msgs/String` | AprilTag ID |

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB camera stream |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Aligned depth data |
| `/tag_detections` | `apriltag_msgs/AprilTagDetectionArray` | AprilTag detections |

## Services

| Service | Type | Description |
|---------|------|-------------|
| `/detection/set_mode` | `reseq_interfaces/SetMode` | Change operational mode |
| `/detection/get_status` | `reseq_interfaces/GetStatus` | Query system status |
| `/detection/compute_coordinate` | `reseq_interfaces/ComputeCoordinate` | Compute 3D coordinates |

## CSV Logging (MAPPING Mode)

Detections in MAPPING mode are logged to `~/reseq_detections.csv` with the following fields:

- **timestamp_iso**: Detection timestamp
- **detection_id**: Unique detection counter
- **type**: Detection type (object, hazmat, qr, apriltag)
- **name**: Object class or label text
- **confidence**: Detection confidence score
- **robot**: Robot identifier (default: 'reseq')
- **detection_mode**: Detection method ('T' for trained model)
- **camera_frame**: Source camera frame
- **target_frame**: Target coordinate frame (usually 'odom')
- **x_target, y_target, z_target**: 3D coordinates in target frame
- **u_center, v_center**: 2D image center coordinates
- **depth_m**: Depth value in meters
- **bbox_xmin, bbox_ymin, bbox_w, bbox_h**: Bounding box parameters

## Trained Models

### Object Detection (`object_detection/`)
- **Model**: YOLOv11n fine-tuned on rescue objects
- **Classes**: Victims, debris, mission-specific objects
- **Weights**: `object_detection/runs/detect/train/weights/best.pt`
- **TensorRT Engine**: `object_detection/runs/detect/train/weights/best.engine` (FP16, ~9 MB)

### Hazmat Detection (`hazmat_detection/`)
- **Model**: YOLOv11n fine-tuned on hazmat placards
- **Classes**: Various NFPA 704 diamonds, DOT placards, hazard symbols
- **Weights**: `hazmat_detection/runs/detect/train/weights/best.pt`
- **TensorRT Engine**: `hazmat_detection/runs/detect/train/weights/best.engine` (FP16, ~9 MB)

### TensorRT Inference

Pre-exported TensorRT `.engine` files (FP16) are included in the repo for faster inference on Jetson GPUs (~15–20 ms vs ~40–50 ms with PyTorch). The detector node automatically loads `.engine` files when available, falling back to `.pt` weights otherwise.

> **Note:** Engine files are hardware-specific (exported on Orin Nano, JetPack 6.2). If you change the Jetson hardware or JetPack version, re-export using `scripts/export_engines.py`. See the [Jetson Orin Nano documentation](https://docs.teamisaac.it/doc/jetson-orin-nano-dev-kit-yvo2K1WGDJ) for detailed instructions.

## Training New Models

### Object Detection

```bash
cd ~/ros2_ws/src/computer_vision/object_detection

# Open Jupyter notebook for training
jupyter notebook model.ipynb
```

### Hazmat Detection

```bash
cd ~/ros2_ws/src/computer_vision/hazmat_detection

# Train model
jupyter notebook model.ipynb
```

The training datasets should be placed in the respective `dataset/` directories (ignored by git) from the links provided in the respective `model.ipynb` files.

### Exporting TensorRT Engines

After training or when deploying on a new Jetson device, export both models to TensorRT:

```bash
# On the Jetson (inside the Docker container):
python3 scripts/export_engines.py
```

This exports both hazmat and object detection models to FP16 TensorRT engines (~5–10 min per model). See the [Jetson Orin Nano documentation](https://docs.teamisaac.it/doc/jetson-orin-nano-dev-kit-yvo2K1WGDJ) for full setup and deployment instructions.

## Configuration

### Camera Intrinsics

Default intrinsics for RealSense D435 are configured in `detection_manager_node.py`:

```python
self.declare_parameter('f_x', 910.3245)
self.declare_parameter('f_y', 909.7875)
self.declare_parameter('c_x', 648.6353)
self.declare_parameter('c_y', 369.6105)
```

Adjust these parameters if using a different camera or calibration.

### Target Frame

By default, detections are transformed to the `odom` frame. Change this via parameter:

```python
self.declare_parameter('target_frame', 'odom')
```

## Competition Best Practices

1. **Initialize Before Mission**: Always run mode 1 (INITIALIZING) first to validate TF tree and CSV setup
2. **Use Appropriate Mode**: MODE 2 for teleoperation, MODE 3 for autonomous exploration scoring
3. **Monitor CSV Output**: Check `~/reseq_detections.csv` after runs for data integrity
4. **Verify Coordinates**: Ensure TF transforms are publishing correctly between camera and `odom` frames
5. **Test Detection Confidence**: Adjust confidence thresholds in `detector.py` if getting false positives


#!/usr/bin/env python3
"""Export YOLO .pt models to TensorRT .engine (FP16) for NVIDIA Jetson.

Run this script **once** inside the Docker container on the Jetson device
to pre-generate TensorRT engine files.  Subsequent launches of the detector
node will pick up the cached .engine files and skip the export step.

Usage (inside the container):
    source /ros_entrypoint.sh
    python3 /ros2_ws/src/computer_vision/scripts/export_engines.py

The export takes ~2-5 minutes per model on an Orin Nano (FP16).
"""

import os
import sys
import time

from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


def export_model(pt_path: str, name: str, half: bool = True) -> None:
    engine_path = pt_path.replace('.pt', '.engine')

    if os.path.isfile(engine_path):
        print(f'[{name}] Engine already exists: {engine_path} \u2014 skipping')
        return

    if not os.path.isfile(pt_path):
        print(f'[{name}] WARNING: weights not found at {pt_path} \u2014 skipping')
        return

    print(f'[{name}] Exporting {pt_path} \u2192 TensorRT (half={half}) \u2026')
    t0 = time.time()
    model = YOLO(pt_path)
    export_path = model.export(format='engine', half=half)
    elapsed = time.time() - t0
    print(f'[{name}] Done in {elapsed:.1f}s \u2192 {export_path}')


def main() -> None:
    try:
        share = get_package_share_directory('computer_vision')
    except Exception:
        print(
            'ERROR: computer_vision package not found. Make sure you sourced the workspace first.',
            file=sys.stderr,
        )
        sys.exit(1)

    models = [
        (f'{share}/hazmat_detection/runs/detect/train/weights/best.pt', 'hazmat'),
        (f'{share}/object_detection/runs/detect/train/weights/best.pt', 'object'),
    ]

    for pt_path, name in models:
        export_model(pt_path, name)

    print('\nAll exports complete.')


if __name__ == '__main__':
    main()

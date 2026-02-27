# The comments in this file are now in English as requested.
import os
import traceback

import cv2
import rclpy
import rclpy.logging
from ament_index_python.packages import get_package_share_directory
from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from ultralytics import YOLO

from computer_vision.detector_modules.apriltag_detector import AprilTagDetector
from computer_vision.detector_modules.concentric_c import OrientationDetection
from computer_vision.detector_modules.motion_detection import MotionDetection
from computer_vision.detector_modules.qr_reader import process_qr_codes
from reseq_interfaces.msg import Detection

# Mode constants
MODE_OFF = 0
MODE_INITIALIZING = 1
MODE_SENSOR_CRATE = 2
MODE_MAPPING = 3


class Detector(Node):
    def __init__(self):
        super().__init__('detector')
        self.bridge = CvBridge()

        # Declare the image topic parameter with a default value for the Realsense camera
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.get_logger().info(f"Listening for images on: '{image_topic}'")

        # Mode handling – QoS must match the TRANSIENT_LOCAL publisher in
        # detection_manager so the detector receives the latched mode even when
        # it subscribes after the mode was already published.
        self.current_mode = MODE_OFF
        mode_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(UInt8, '/detection/mode', self._on_mode_update, mode_qos)
        self.get_logger().info('Subscribed to /detection/mode for runtime control')

        # Camera Subscriptions
        # Use the topic from the parameter for the color image subscription
        self.color_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )

        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10,
        )
        self.depth_image = None

        self.apriltag_subscription = self.create_subscription(
            AprilTagDetectionArray, '/tag_detections', self._apriltag_callback, 10
        )

        # Publishers
        self.model_pub = self.create_publisher(Image, '/detector/model_output', 10)
        self.detection_pub = self.create_publisher(Detection, '/object_detection/detections', 10)
        self.depth_pub = self.create_publisher(Image, '/object_detection/depthsized', 10)

        # YOLO Models – prefer TensorRT .engine (FP16) for maximum inference
        # speed on NVIDIA Jetson.  If an .engine file does not yet exist next to
        # the .pt weights it is exported automatically on first launch (one‑time
        # cost of ~2-5 min on Orin Nano).
        self.img_size = (640, 640)  # Standard YOLO input size
        share_folder = get_package_share_directory('computer_vision')
        pt_path1 = f'{share_folder}/hazmat_detection/runs/detect/train/weights/best.pt'
        pt_path2 = f'{share_folder}/object_detection/runs/detect/train/weights/best.pt'

        self.model1 = self._load_model(pt_path1, 'hazmat')
        self.model2 = self._load_model(pt_path2, 'object')

        # Custom CV Modules
        self.od = OrientationDetection()
        self.md = MotionDetection()
        self.ad = AprilTagDetector(self)

        # Camera intrinsics (default, can be overridden later)
        self.f_x = 910.3245
        self.f_y = 909.7875
        self.c_x = 648.6353
        self.c_y = 369.6105

        # State
        self.camera_frame = 'camera_depth_optical_frame'
        self.detection_counter = 1

        # Frame skipping for CPU-heavy modules.
        # They detect static targets so running less frequently is fine.
        self._frame_counter = 0
        self._orient_interval = 15  # run orientation every 15 frames
        self._qr_interval = 10  # run QR every 10 frames
        self._motion_interval = 2  # run motion every 2 frames

        # Downscaled resolution for CPU-heavy modules (orientation, QR, motion).
        self._cpu_processing_size = (640, 480)

        self.get_logger().info('Detector initialized successfully')

    # ------------------------------------------------------------------
    # TensorRT model helper
    # ------------------------------------------------------------------
    def _load_model(self, pt_path: str, name: str) -> YOLO:
        """Load a YOLO model, preferring TensorRT FP16 .engine for speed.

        If the .engine file already exists beside the .pt weights it is loaded
        directly.  Otherwise the .pt model is exported to TensorRT FP16 first
        (requires CUDA/TensorRT on the device).  When neither a GPU nor
        TensorRT is available the .pt weights are used as a fallback.
        """
        engine_path = pt_path.replace('.pt', '.engine')

        if os.path.isfile(engine_path):
            self.get_logger().info(f'[{name}] Loading TensorRT engine: {engine_path}')
            return YOLO(engine_path, task='detect')

        # Attempt TensorRT export (FP16 for best Jetson performance)
        try:
            self.get_logger().info(
                f'[{name}] TensorRT engine not found \u2013 exporting from {pt_path} '
                '(this may take a few minutes on first launch)\u2026'
            )
            model = YOLO(pt_path)
            export_path = model.export(format='engine', half=True)
            self.get_logger().info(f'[{name}] TensorRT export complete: {export_path}')
            return YOLO(export_path, task='detect')
        except Exception as e:
            self.get_logger().warn(
                f'[{name}] TensorRT export failed ({e}), falling back to PyTorch weights'
            )
            return YOLO(pt_path)

    def _on_mode_update(self, msg):
        new_mode = int(msg.data)
        if new_mode != self.current_mode:
            self.get_logger().info(f'Mode changed: {self.current_mode} -> {new_mode}')
            self.current_mode = new_mode
            self._apply_mode(new_mode)

    def _apply_mode(self, mode):
        if mode == MODE_OFF:
            self.get_logger().info('Detector: OFF - shutting down all pipelines')
        elif mode == MODE_INITIALIZING:
            self.get_logger().info(
                'Detector: INITIALIZING - loading models only (no inference yet)'
            )
        elif mode == MODE_SENSOR_CRATE:
            self.get_logger().info('Detector: SENSOR_CRATE - running QR/Hazmat/Apriltag only')
        elif mode == MODE_MAPPING:
            self.get_logger().info('Detector: MAPPING - running YOLO + others')

    def image_callback(self, msg):
        mode = self.current_mode
        if mode == MODE_OFF:
            return

        color_image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Prepare the 640x640 RGB input for YOLO models
        img_rgb_for_yolo = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_for_yolo = cv2.resize(img_rgb_for_yolo, self.img_size)

        # Frame skipping: run CPU-heavy modules only every Nth frame.
        run_orient = self._frame_counter % self._orient_interval == 0
        run_qr = self._frame_counter % self._qr_interval == 0
        run_motion = mode == MODE_MAPPING and self._frame_counter % self._motion_interval == 0
        self._frame_counter += 1

        # Downscale once — used for all CPU modules and as the visualization
        # canvas.  YOLO boxes are rescaled to this resolution automatically
        # in _process_hazmat / _process_objects.
        small = cv2.resize(color_image_bgr, self._cpu_processing_size)

        if run_orient:
            small, _, _ = self.od.process_image(small)

        qr_detections = []
        if run_qr:
            small, qr_detections = process_qr_codes(small)

        # Ego-motion compensated motion detection (optical flow).
        if run_motion:
            _, motion_bboxes = self.md.process_image(small)
            for x, y, bw, bh in motion_bboxes:
                cv2.rectangle(
                    small,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 255, 0),
                    2,
                )

        visualization_frame = small

        if mode in (MODE_SENSOR_CRATE, MODE_MAPPING):
            self._process_hazmat(visualization_frame, img_rgb_for_yolo, msg)
            self._process_qr_codes(qr_detections, msg)

        if mode == MODE_MAPPING:
            self._process_objects(visualization_frame, img_rgb_for_yolo, msg)

        # Publish the frame with drawn detections.
        model_output_msg = self.bridge.cv2_to_imgmsg(visualization_frame, encoding='bgr8')
        self.model_pub.publish(model_output_msg)

    def depth_callback(self, msg):
        if self.current_mode == MODE_OFF:
            return
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _process_qr_codes(self, qr_detections, msg):
        if not qr_detections:
            return

        for det in qr_detections:
            bbox = det['bbox']
            label = det['text']

            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)

            depth_value = 0.0
            if self.depth_image is not None:
                mid_x = int((x1 + x2) / 2)
                mid_y = int((y1 + y2) / 2)
                if (
                    0 <= mid_y < self.depth_image.shape[0]
                    and 0 <= mid_x < self.depth_image.shape[1]
                ):
                    depth_value = float(self.depth_image[mid_y, mid_x]) / 1e3

            det_msg = self._create_detection_msg(
                header=msg.header,
                det_type='qr',
                label=label,
                conf=1.0,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                depth_value=depth_value,
            )
            self.detection_pub.publish(det_msg)

    def _process_hazmat(self, visualization_frame, yolo_input_image, msg):
        try:
            results = self.model1(yolo_input_image, verbose=False)

            for result in results:
                # Scale boxes from YOLO's 640x640 back to the original image size for correct drawing
                orig_h, orig_w = visualization_frame.shape[:2]
                result.boxes.orig_shape = yolo_input_image.shape[:2]
                boxes = result.boxes.xyxy.cpu().numpy()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (orig_w / self.img_size[0])
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (orig_h / self.img_size[1])

                for i, conf in enumerate(result.boxes.conf):
                    if conf <= 0.5:
                        continue

                    x1, y1, x2, y2 = boxes[i]
                    label = result.names[result.boxes.cls[i].item()]
                    depth_value = 0.0
                    # Depth calculation would also need coordinate scaling, for now we default to 0

                    det_msg = self._create_detection_msg(
                        header=msg.header,
                        det_type='hazmat',
                        label=label,
                        conf=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        depth_value=depth_value,
                    )
                    self.detection_pub.publish(det_msg)

                    cv2.rectangle(
                        visualization_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                    )
        except Exception as e:
            self.get_logger().error(f'Error in _process_hazmat: {e}')

    def _apriltag_callback(self, msg):
        detection_msgs = self.ad.process_detections(
            msg, self.depth_image, self._create_detection_msg
        )
        for det_msg in detection_msgs:
            self.detection_pub.publish(det_msg)

    def _process_objects(self, visualization_frame, yolo_input_image, msg):
        try:
            results = self.model2(yolo_input_image, verbose=False)

            for result in results:
                # Scale boxes from YOLO's 640x640 back to the original image size for correct drawing
                orig_h, orig_w = visualization_frame.shape[:2]
                result.boxes.orig_shape = yolo_input_image.shape[:2]
                boxes = result.boxes.xyxy.cpu().numpy()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (orig_w / self.img_size[0])
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (orig_h / self.img_size[1])

                for i, conf in enumerate(result.boxes.conf):
                    if conf <= 0.5:
                        continue

                    x1, y1, x2, y2 = boxes[i]
                    label = result.names[result.boxes.cls[i].item()]
                    depth_value = 0.0
                    # Depth calculation would also need coordinate scaling, for now we default to 0

                    det_msg = self._create_detection_msg(
                        header=msg.header,
                        det_type='object',
                        label=label,
                        conf=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        depth_value=depth_value,
                    )
                    self.detection_pub.publish(det_msg)

                    cv2.rectangle(
                        visualization_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                    )
        except Exception as e:
            self.get_logger().error(f'Error in _process_objects: {e}')

    def _create_detection_msg(self, header, det_type, label, conf, x1, y1, x2, y2, depth_value):
        det = Detection()
        det.header.stamp = header.stamp
        det.header.frame_id = self.camera_frame
        det.detection = self.detection_counter
        self.detection_counter += 1
        det.type = det_type
        det.name = str(label)
        det.confidence = float(conf)
        det.xmin = int(x1)
        det.ymin = int(y1)
        det.width = int(x2 - x1)
        det.height = int(y2 - y1)
        det.depth_center = depth_value
        det.camera_frame = self.camera_frame
        det.robot = 'reseq'
        det.mode = 'T'
        return det


def main(args=None):
    rclpy.init(args=args)
    detector = None
    try:
        detector = Detector()
        rclpy.spin(detector)
    except Exception as err:
        if detector:
            detector.get_logger().fatal(
                f'Error in the Detector node: {str(err)}\n{traceback.format_exc()}'
            )
        raise err
    finally:
        if detector:
            detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

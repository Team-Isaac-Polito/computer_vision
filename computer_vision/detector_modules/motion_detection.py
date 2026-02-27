import cv2
import numpy as np


class MotionDetection:
    """Detect real object motion while compensating for camera ego-motion.

    Designed for the ReseQ snake robot whose camera undergoes significant
    translation and rotation during locomotion.  The algorithm:

    1. Tracks sparse features (Shi-Tomasi corners) with Lucas-Kanade optical
       flow between consecutive frames.
    2. Estimates a **homography** (8-DOF perspective transform) via RANSAC to
       model the global camera motion — this handles the perspective changes
       that occur when the snake head tilts or rotates.
    3. Warps the previous frame into the current frame's perspective.
    4. Computes the per-pixel difference; only genuine scene changes (moving
       objects/people) remain after compensation.
    5. Applies **temporal consistency**: a motion region must appear in at
       least two consecutive processed frames to be reported, eliminating
       transient false positives from parallax or estimation noise.
    6. Masks a configurable border margin to suppress edge artifacts caused
       by warp out-of-bounds and lens-induced parallax at image corners.
    """

    def __init__(self):
        self.prev_gray = None
        self.prev_thresh = None  # for temporal consistency

        # Shi-Tomasi corner detection parameters for feature tracking.
        self.feature_params = dict(
            maxCorners=300,  # more features → better estimation
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7,
        )

        # Lucas-Kanade optical flow parameters.
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        # Motion detection thresholds.
        self.min_area = 800  # ignore contours smaller than this (px²)
        self.diff_threshold = 30  # per-pixel intensity diff threshold
        self.bx_thickness = 2  # bounding-box line thickness
        self.min_good_tracks = 10  # minimum tracked features for a reliable estimate
        self.border_pct = 0.05  # ignore this fraction of each border

        self.debug_contours = False

    # ------------------------------------------------------------------
    @staticmethod
    def _make_border_mask(h: int, w: int, margin: float) -> np.ndarray:
        """Create a mask that is 0 in a border region and 255 in the centre."""
        mask = np.zeros((h, w), dtype=np.uint8)
        my = max(int(h * margin), 1)
        mx = max(int(w * margin), 1)
        mask[my : h - my, mx : w - mx] = 255
        return mask

    # ------------------------------------------------------------------
    def process_image(self, frame):
        """Detect motion compensating for camera ego-motion.

        Returns
        -------
        frame : np.ndarray
            The input frame with motion bounding boxes drawn (for standalone use).
        bboxes : list[tuple[int, int, int, int]]
            List of ``(x, y, w, h)`` bounding boxes of detected motion regions.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gaussian blur to suppress sensor noise before any comparison.
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_thresh = np.zeros_like(gray)
            return frame, []

        # ----- 1. Track features from previous → current frame -----
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)

        if prev_pts is None or len(prev_pts) < self.min_good_tracks:
            self.prev_gray = gray.copy()
            self.prev_thresh = np.zeros_like(gray)
            return frame, []

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        good_prev = prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) < self.min_good_tracks:
            self.prev_gray = gray.copy()
            self.prev_thresh = np.zeros_like(gray)
            return frame, []

        # ----- 2. Estimate camera motion (homography) -----
        # Homography (8-DOF) handles perspective changes much better than
        # a 4-DOF affine when the snake head tilts or rotates.
        H, inlier_mask = cv2.findHomography(
            good_prev,
            good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        if H is None:
            self.prev_gray = gray.copy()
            self.prev_thresh = np.zeros_like(gray)
            return frame, []

        # If estimation quality is poor (few inliers), the residual will
        # contain large false-positive regions — skip detection this frame.
        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        inlier_ratio = n_inliers / max(len(good_prev), 1)
        if inlier_ratio < 0.5:
            self.prev_gray = gray.copy()
            self.prev_thresh = np.zeros_like(gray)
            return frame, []

        h, w = gray.shape

        # ----- 3. Warp previous frame to align with current -----
        warped_prev = cv2.warpPerspective(self.prev_gray, H, (w, h))

        # ----- 4. Frame difference on aligned frames -----
        diff = cv2.absdiff(warped_prev, gray)

        # Mask out warp-invalid pixels and border regions.
        warp_valid = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, H, (w, h))
        border_mask = self._make_border_mask(h, w, self.border_pct)
        combined_mask = cv2.bitwise_and(warp_valid, border_mask)
        diff = cv2.bitwise_and(diff, combined_mask)

        # ----- 5. Thresholding + morphological cleanup -----
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=2)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # ----- 6. Temporal consistency -----
        # Require motion to be present in *both* the current and the
        # previous processed frame.  A single-frame flash of false motion
        # (e.g. from parallax on a nearby wall) is suppressed.
        # Dilate the previous mask slightly so small shifts between frames
        # still overlap.
        prev_dilated = cv2.dilate(
            self.prev_thresh,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
        stable = cv2.bitwise_and(thresh, prev_dilated)
        self.prev_thresh = thresh.copy()

        # ----- 7. Contour detection -----
        contours, _ = cv2.findContours(stable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        if contours:
            if self.debug_contours:
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_area:
                    x, y, bw, bh = cv2.boundingRect(contour)
                    bboxes.append((x, y, bw, bh))
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + bw, y + bh),
                        (0, 255, 0),
                        self.bx_thickness,
                    )

        self.prev_gray = gray.copy()
        return frame, bboxes


if __name__ == '__main__':
    md = MotionDetection()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, bboxes = md.process_image(frame)

        cv2.imshow('Motion Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

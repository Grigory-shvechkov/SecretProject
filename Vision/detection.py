"""
detection.py -- object detectors for the fish tank vision system.

Three detectors live here:

  RedBallDetector    -- benchmark detector. Pure math (HSV color filtering),
                        fast enough for real-time on a Pi 3. Finds the
                        largest red blob and returns its center + radius.

  ColorMarkerDetector -- calibration helper. Same HSV-filtering approach as
                        RedBallDetector, but returns EVERY blob of a given
                        color instead of just the largest one. Used to find
                        colored corner markers (stickers/tape/paper) for
                        automatic tank-bounds calibration -- see
                        coordinate_mapper.py's marker calibration mode.

  FishDetector       -- AI detector. Runs a YOLO neural network. Slow on a
                        Pi 3 (~1-3 s per frame) but knows actual object
                        classes. Pass fish-trained weights via model_path
                        for real fish detection (stock yolov8n.pt has NO
                        fish class -- it is only useful to verify the
                        pipeline works).

Design rule: detect() methods return DATA ONLY (tuples/lists), never draw
or open windows. Drawing is a separate draw() method. This keeps detectors
reusable by run.py and coordinate_mapper.py alike.
"""

import cv2
import numpy as np


class RedBallDetector:
    """Detect the largest red object in a frame via HSV color filtering.

    Parameters
    ----------
    min_area : int
        Smallest contour area (in pixels) considered a real object.
        Filters out reflections and specks.
    min_radius : int
        Smallest enclosing-circle radius (in pixels) accepted.
    lower_sat, lower_val : int
        Saturation / brightness floors for the red mask. Lower these
        (e.g. to 100 / 50) if the ball gets dull or dark deeper in the
        water -- water absorbs red light with depth.
    """

    def __init__(self, min_area=300, min_radius=10, lower_sat=120, lower_val=70):
        self.min_area = min_area
        self.min_radius = min_radius

        # Red wraps around the ends of the HSV hue wheel (0-180 in OpenCV),
        # so we need two ranges: near 0 and near 180.
        self.lower_red1 = np.array([0, lower_sat, lower_val])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, lower_sat, lower_val])
        self.upper_red2 = np.array([180, 255, 255])

    def detect(self, frame, hsv=None):
        """Find the ball in one frame.

        Parameters
        ----------
        hsv : optional precomputed HSV frame (blurred + cvtColor already
            applied). Pass this in when another detector is running on
            the same frame this tick, so the blur/color-convert cost
            isn't paid twice -- see ColorMarkerDetector, used together in
            debug_view.py. Leave as None to have this method compute it.

        Returns
        -------
        result : (x, y, radius) tuple of ints, or None if nothing found.
                 x, y = pixel coordinates of the ball's center.
        mask   : black-and-white debug image (white = pixels judged red).
        """
        # Blur to suppress per-pixel sensor noise, then go to HSV so the
        # color check survives lighting changes.
        if hsv is None:
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Two red ranges OR'd into one mask.
        mask = cv2.inRange(hsv, self.lower_red1, self.upper_red1) | \
               cv2.inRange(hsv, self.lower_red2, self.upper_red2)

        # Erode kills tiny specks; dilate restores the ball to full size.
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Outline every white blob, keep the biggest, wrap it in a circle.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        result = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > self.min_area:
                (x, y), r = cv2.minEnclosingCircle(c)
                if r > self.min_radius:
                    result = (int(x), int(y), int(r))

        return result, mask

    def draw(self, frame, result):
        """Draw the detection onto a frame (for display). Returns the frame."""
        if result:
            x, y, r = result
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)          # ball outline
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)          # center dot
            cv2.putText(frame, f"ball ({x},{y}) r={r}",
                        (x - 40, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


class ColorMarkerDetector:
    """Detect every blob of one color in a frame -- used to find physical
    corner markers (colored stickers/tape/paper) for calibration, not the
    tracked object itself.

    Unlike RedBallDetector, this returns ALL blobs above min_area (largest
    first), since calibration needs to see two separate markers at once.
    Pick a marker color that won't be confused with whatever's being
    tracked -- yellow sits far from the red ball/fish marker's hue (0-10 /
    170-180) so the two never overlap. Watch out for warm aquarium
    lighting or yellow-toned gravel/decor washing background pixels into
    this range; narrow the band or check a background reference
    (color_test.py) if that happens.

    Parameters
    ----------
    lower_hsv, upper_hsv : (H, S, V) tuples
        Single HSV range for the marker color (no red-style hue wraparound
        needed for most colors). Defaults to a yellow range.
    min_area : int
        Smallest contour area (in pixels) considered a real marker.
    """

    def __init__(self, lower_hsv=(20, 60, 40), upper_hsv=(35, 255, 255), min_area=150):
        self.lower = np.array(lower_hsv)
        self.upper = np.array(upper_hsv)
        self.min_area = min_area

    def detect(self, frame, hsv=None):
        """Find every marker blob in one frame.

        Parameters
        ----------
        hsv : optional precomputed HSV frame -- see RedBallDetector.detect
            for why. Leave as None to have this method compute it.

        Returns
        -------
        centers : list of (x, y) pixel coordinates, largest blob first.
                  Empty list if nothing found.
        mask    : black-and-white debug image (white = pixels judged
                  to be this color).
        """
        if hsv is None:
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_area:
                (x, y), _ = cv2.minEnclosingCircle(c)
                blobs.append((int(x), int(y), area))

        blobs.sort(key=lambda b: b[2], reverse=True)
        return [(x, y) for x, y, _ in blobs], mask

    def draw(self, frame, centers):
        """Draw every detected marker onto a frame (for display). Returns the frame."""
        for (x, y) in centers:
            cv2.circle(frame, (x, y), 8, (255, 0, 0), 2)
            cv2.putText(frame, f"({x},{y})", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame


class FishDetector:
    """YOLO-based object detector.

    Notes
    -----
    * `ultralytics` is imported inside __init__ on purpose: it is a heavy
      dependency, and this way ball mode works even if it isn't installed.
    * Stock yolov8n.pt is trained on COCO, which contains NO fish class.
      Supply fish-trained weights (e.g. from Roboflow Universe) via
      model_path for actual fish detection.
    * Expect ~1-3 seconds per frame on a Pi 3. That is normal.
    """

    def __init__(self, model_path="yolov8n.pt", conf=0.4):
        from ultralytics import YOLO
        self.model = YOLO(model_path)   # downloads weights on first ever run
        self.conf = conf                # minimum confidence to report

    def detect(self, frame):
        """Run the network on one frame.

        Returns
        -------
        detections : list of (name, confidence, (x1, y1, x2, y2)) tuples.
                     (x1, y1) = top-left corner, (x2, y2) = bottom-right.
                     Empty list if nothing found.
        raw        : the raw ultralytics result object (needed by draw()).
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []
        for box in results[0].boxes:
            name = self.model.names[int(box.cls)]          # class id -> word
            xyxy = tuple(int(v) for v in box.xyxy[0])      # box corners
            detections.append((name, float(box.conf), xyxy))
        return detections, results[0]

    def draw(self, frame, raw):
        """Return the frame annotated by YOLO (boxes + labels)."""
        return raw.plot()

    @staticmethod
    def center_of(detection):
        """Convenience: turn one detection into (cx, cy, half_size),
        the same shape RedBallDetector returns, so the coordinate
        mapper can consume fish detections identically to ball ones.
        """
        _, _, (x1, y1, x2, y2) = detection
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = max(x2 - x1, y2 - y1) // 2
        return (cx, cy, half)
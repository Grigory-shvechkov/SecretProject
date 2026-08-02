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

  FishDetector       -- AI detector. Runs a TensorFlow Lite neural network
                        (tflite-runtime) -- fast enough for real-time on a
                        Pi 4 (tens of ms/frame) and knows actual object
                        classes, unlike the color detectors above. Pass a
                        fish-trained .tflite model via model_path for real
                        fish detection (a stock COCO-trained model has NO
                        fish class -- it is only useful to verify the
                        pipeline works). Chosen over PyTorch/ultralytics
                        YOLO because official PyTorch has no wheels at all
                        for this Pi's 32-bit ARM (armv7l) architecture.

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
    """TensorFlow Lite object detector.

    Replaced an earlier PyTorch/ultralytics YOLO version of this class:
    official PyTorch publishes NO wheels at all for 32-bit ARM (armv7l --
    this project's actual Pi architecture), at any version, on PyPI or
    piwheels, so it could never be installed here. tflite-runtime does
    publish an armv7l wheel, and TFLite is purpose-built for exactly this
    kind of embedded inference besides -- expect tens of milliseconds per
    frame on a Pi 4, not the ~1-3s/frame YOLO+torch would have cost even
    if it could run at all.

    Notes
    -----
    * `tflite_runtime` is imported inside __init__ on purpose, same
      reasoning as the old ultralytics import: it's a real dependency,
      and this way ball mode works even if it isn't installed.
    * Unlike ultralytics' YOLO(), tflite-runtime has NO auto-download --
      model_path must point at an actual .tflite file you already have.
      A stock COCO-trained SSD model (e.g. the classic "detect.tflite" +
      "labelmap.txt" pair from TensorFlow's object detection examples)
      proves the pipeline runs but has NO fish class -- same caveat the
      old stock yolov8n.pt had. Supply fish-trained weights, converted to
      .tflite, for real fish detection.
    * Assumes the model was exported WITH the standard TFLite detection
      post-processing op baked in, i.e. its 4 output tensors are (in
      order) [boxes, classes, scores, num_detections] with NMS already
      applied -- true of virtually every prebuilt or edge-export TFLite
      detection model. A custom export with a different output order
      would need detect() below adjusted to match.
    """

    def __init__(self, model_path, labels_path, conf=0.4):
        from tflite_runtime.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        _, self._input_h, self._input_w, _ = self._input_details[0]["shape"]
        self._is_float_model = self._input_details[0]["dtype"] == np.float32

        with open(labels_path) as f:
            self.labels = [line.strip() for line in f if line.strip()]

        self.conf = conf                # minimum confidence to report

    def detect(self, frame):
        """Run the network on one frame.

        Returns
        -------
        detections : list of (name, confidence, (x1, y1, x2, y2)) tuples,
                     pixel coordinates already scaled to the ORIGINAL
                     frame's size (the model itself only sees a resized
                     copy). (x1, y1) = top-left corner, (x2, y2) =
                     bottom-right. Empty list if nothing found.
        raw        : always None -- kept only so this matches the
                     (detections, raw) shape every other detector's
                     detect() returns. Unlike ultralytics' Results
                     object, TFLite's raw output tensors aren't useful to
                     carry forward once decoded into `detections`.
        """
        frame_h, frame_w = frame.shape[:2]
        resized = cv2.resize(frame, (self._input_w, self._input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb, axis=0)
        if self._is_float_model:
            # Quantized (uint8) models take raw 0-255 input as-is; float
            # models expect it centered/scaled to roughly [-1, 1].
            input_data = (np.float32(input_data) - 127.5) / 127.5

        self.interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self._output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self._output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self._output_details[2]["index"])[0]

        detections = []
        for box, cls, score in zip(boxes, classes, scores):
            if score < self.conf:
                continue
            ymin, xmin, ymax, xmax = box    # normalized 0-1, model's own convention
            xyxy = (
                int(xmin * frame_w), int(ymin * frame_h),
                int(xmax * frame_w), int(ymax * frame_h),
            )
            class_id = int(cls)
            name = self.labels[class_id] if 0 <= class_id < len(self.labels) else str(class_id)
            detections.append((name, float(score), xyxy))

        return detections, None

    def draw(self, frame, detections):
        """Draw every detection's box + label onto frame IN PLACE (same
        contract as RedBallDetector/ColorMarkerDetector's draw() -- unlike
        the old ultralytics-backed version, which had to return a FRESH
        image via raw.plot() instead). Returns the frame.

        Red, deliberately NOT the same green run.py's _draw_quad() uses
        for the calibrated tank outline -- both get drawn onto the same
        debug frame, and two different things in the same color there
        would be indistinguishable."""
        for name, score, (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} {score:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame

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
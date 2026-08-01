"""
run.py -- the whole fish tank vision pipeline, start to finish, in one
script. No separate calibration step, no other scripts to run first --
just:

    python run.py

What it does, in order, every time you run it:
    1. Opens both cameras.
    2. Auto-detects the EXPECTED_MARKERS yellow corner markers glued to
       the tank's interior corners, in each camera's view (retries for a
       bit if they're not all visible yet -- check placement/lighting if
       it times out), and works out which corner is which automatically
       (order_quad_points() in coordinate_mapper.py).
    3. Builds a rotation-correct pixel -> cm mapping from those corners
       plus TANK_SIZE_CM below -- correct even if a camera is mounted
       off-center or at a slight angle, not perfectly square-on.
    4. Loops forever: finds the fish with FishDetector (YOLO) in both
       cameras, combines the two views into one 3D (X, Y, Z) position,
       and POSTs it to the feed-fish API.

The only setup this needs: measure your tank and set TANK_SIZE_CM below,
and physically stick EXPECTED_MARKERS yellow markers (stickers/tape/
paper) at the tank's interior corners, once, permanently. Everything else
above is automatic, every time you run this file.

(Vision/debug_view.py and Vision/color_test.py still exist as optional
troubleshooting tools -- e.g. if markers aren't being found, color_test.py
helps you pick a better HSV range for your actual camera/lighting -- but
neither is a required step. This script alone is the whole pipeline.)
"""

import os
import sys
import time

VISION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Vision")
sys.path.insert(0, VISION_DIR)

import requests

from capture import Camera
from detection import FishDetector, ColorMarkerDetector
from coordinate_mapper import CoordinateMapper, order_quad_points

# ===== CONFIG =====

# Confirmed via `v4l2-ctl --list-devices` + `--list-formats-ext` on the Pi:
# each USB camera exposes two /dev/video nodes, but only one of the pair
# actually streams video -- the other is a metadata-only node with no
# formats. On this rig that's indices 0 and 2 (1 and 3 are the metadata
# nodes). Confirm which physical camera is "front" vs "side" and swap
# these if backwards.
FRONT_CAMERA_INDEX = 0
SIDE_CAMERA_INDEX = 2

# Measure your tank's interior and set this: (width_X, height_Y, depth_Z) in cm.
TANK_SIZE_CM = (60.0, 30.0, 30.0)

# Yellow corner-marker HSV range -- see Vision/color_test.py if this
# doesn't cleanly segment your actual stickers/tape under your tank's
# real lighting.
MARKER_LOWER = (20, 60, 40)
MARKER_UPPER = (35, 255, 255)
EXPECTED_MARKERS = 4
CORNER_DETECT_TIMEOUT_SECONDS = 30.0

API_URL = "https://feed-fish.onrender.com/newPos"
REQUEST_TIMEOUT_SECONDS = 5.0
SEND_INTERVAL_SECONDS = 0.5

CM_TO_INCHES = 1 / 2.54

# Stock yolov8n.pt has NO fish class (see detection.py's FishDetector
# docstring) -- fine to confirm the pipeline runs, useless for real fish
# tracking until pointed at fish-trained weights. Expect ~1-3s/frame on a
# Pi 3, vs. near-instant for the marker color filtering above.
FISH_MODEL_PATH = "yolov8n.pt"
FISH_CONF_THRESHOLD = 0.4


def _best_center(detections):
    """Pick the highest-confidence YOLO detection this tick and convert it
    to the (cx, cy, half) shape CoordinateMapper.combine() expects. None
    if nothing was detected."""
    if not detections:
        return None
    best = max(detections, key=lambda d: d[1])
    return FishDetector.center_of(best)


def _find_corners(cam, label):
    """Watch this camera's live feed until all EXPECTED_MARKERS yellow
    markers are visible at once, then return their pixel positions
    ordered (top-left, top-right, bottom-right, bottom-left).

    This is the entire calibration step -- it replaces what used to be a
    separate `python coordinate_mapper.py --camera N --markers` script
    you had to run first. It just happens automatically here, using the
    live camera this script already has open.
    """
    detector = ColorMarkerDetector(MARKER_LOWER, MARKER_UPPER)
    deadline = time.monotonic() + CORNER_DETECT_TIMEOUT_SECONDS
    last_count = -1
    while time.monotonic() < deadline:
        frame = cam.read()
        if frame is not None:
            centers, _ = detector.detect(frame)
            if len(centers) != last_count:
                print(f"{label}: found {len(centers)}/{EXPECTED_MARKERS} corner markers...")
                last_count = len(centers)
            if len(centers) >= EXPECTED_MARKERS:
                return order_quad_points(centers[:EXPECTED_MARKERS])
        time.sleep(0.1)
    raise TimeoutError(
        f"{label}: only found {max(last_count, 0)}/{EXPECTED_MARKERS} yellow corner markers "
        f"after {CORNER_DETECT_TIMEOUT_SECONDS:.0f}s. Check marker placement/lighting, or run "
        f"Vision/color_test.py to verify MARKER_LOWER/MARKER_UPPER against your camera.")


def send_position(position):
    """POST a position dict like {'x': .., 'y': .., 'z': ..} to the API."""
    try:
        response = requests.post(
            API_URL,
            json=position,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        print(f"Sent {position} -> {response.status_code} {response.json()}")
    except requests.exceptions.RequestException as error:
        print(f"Request failed for {position}: {error}")


def main():
    print("Starting vision pipeline. Press Ctrl+C to stop.")
    try:
        with Camera(FRONT_CAMERA_INDEX) as front_cam, Camera(SIDE_CAMERA_INDEX) as side_cam:
            print(f"Looking for {EXPECTED_MARKERS} yellow corner markers on each camera...")
            try:
                front_corners = _find_corners(front_cam, "FRONT")
                side_corners = _find_corners(side_cam, "SIDE")
            except TimeoutError as error:
                print(error)
                return

            mapper = CoordinateMapper(front_corners, side_corners, TANK_SIZE_CM)
            print("Calibrated. Starting fish tracking.")

            detector_front = FishDetector(model_path=FISH_MODEL_PATH, conf=FISH_CONF_THRESHOLD)
            detector_side = FishDetector(model_path=FISH_MODEL_PATH, conf=FISH_CONF_THRESHOLD)

            while True:
                front_frame = front_cam.read()
                side_frame = side_cam.read()

                if front_frame is None or side_frame is None:
                    print("Dropped frame, skipping this tick.")
                    time.sleep(SEND_INTERVAL_SECONDS)
                    continue

                front_detections, _ = detector_front.detect(front_frame)
                side_detections, _ = detector_side.detect(side_frame)

                front_result = _best_center(front_detections)
                side_result = _best_center(side_detections)

                if front_result is None or side_result is None:
                    print("Object not visible in both cameras, skipping this tick.")
                    time.sleep(SEND_INTERVAL_SECONDS)
                    continue

                position_cm, y_diff = mapper.combine(front_result, side_result)
                if not mapper.in_tank(position_cm):
                    print(f"Detected position {position_cm}cm is outside the tank, skipping.")
                    time.sleep(SEND_INTERVAL_SECONDS)
                    continue

                if y_diff > 3.0:
                    print(f"Warning: front/side cameras disagree on height by {y_diff}cm")

                x_cm, y_cm, z_cm = position_cm
                send_position({
                    "x": round(x_cm * CM_TO_INCHES, 2),
                    "y": round(y_cm * CM_TO_INCHES, 2),
                    "z": round(z_cm * CM_TO_INCHES, 2),
                })

                time.sleep(SEND_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()

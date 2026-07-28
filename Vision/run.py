"""
run.py -- main loop for the fish tank vision pipeline.

    capture (2 USB cameras)
        -> detect (RedBallDetector, one per camera)
        -> map (CoordinateMapper: two 2D detections -> one 3D position, in cm)
        -> convert cm -> inches (API/Roblox side works in inches, see old/simulation.py)
        -> POST the position to the feed-fish API

Requires a calibration.json in this folder first -- generate one per camera with:
    python coordinate_mapper.py --camera <index>

Run from inside the Vision/ folder:
    python run.py
"""

import time

import requests

from capture import Camera
from detection import RedBallDetector
from coordinate_mapper import CoordinateMapper

# ===== CONFIG =====

# Confirmed via `v4l2-ctl --list-devices` + `--list-formats-ext` on the Pi:
# each USB camera exposes two /dev/video nodes, but only one of the pair
# actually streams video -- the other is a metadata-only node with no
# formats. On this rig that's indices 0 and 2 (1 and 3 are the metadata
# nodes). Confirm which physical camera is "front" vs "side" and swap
# these if backwards. Prefer replacing these with the stable
# /dev/v4l/by-path/... names once known, since raw indices can shift
# across reboots/replugs.
FRONT_CAMERA_INDEX = 0
SIDE_CAMERA_INDEX = 2

CALIBRATION_PATH = "calibration.json"

API_URL = "https://feed-fish.onrender.com/newPos"
REQUEST_TIMEOUT_SECONDS = 5.0
SEND_INTERVAL_SECONDS = 0.5

CM_TO_INCHES = 1 / 2.54


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
    mapper = CoordinateMapper.load_or_none(CALIBRATION_PATH)
    if mapper is None:
        print(f"No calibration found at {CALIBRATION_PATH}.")
        print("Run 'python coordinate_mapper.py --camera <index>' for each camera first.")
        return

    detector_front = RedBallDetector()
    detector_side = RedBallDetector()

    print("Starting vision pipeline. Press Ctrl+C to stop.")
    try:
        with Camera(FRONT_CAMERA_INDEX) as front_cam, Camera(SIDE_CAMERA_INDEX) as side_cam:
            while True:
                front_frame = front_cam.read()
                side_frame = side_cam.read()

                if front_frame is None or side_frame is None:
                    print("Dropped frame, skipping this tick.")
                    time.sleep(SEND_INTERVAL_SECONDS)
                    continue

                front_result, _ = detector_front.detect(front_frame)
                side_result, _ = detector_side.detect(side_frame)

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

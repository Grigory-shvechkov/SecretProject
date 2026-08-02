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
    4. Loops forever: finds the fish with FishDetector (TensorFlow Lite)
       in both cameras, combines the two views into one 3D (X, Y, Z)
       position, and POSTs it to the feed-fish API.

The only setup this needs: measure your tank and set TANK_SIZE_CM below,
and physically stick EXPECTED_MARKERS yellow markers (stickers/tape/
paper) at the tank's interior corners, once, permanently. Everything else
above is automatic, every time you run this file.

(Vision/debug_view.py and Vision/color_test.py still exist as optional
troubleshooting tools -- e.g. if markers aren't being found, color_test.py
helps you pick a better HSV range for your actual camera/lighting -- but
neither is a required step. This script alone is the whole pipeline.)

Pass --debug to also open a live window (both cameras side by side, one
window total) showing whatever's currently detected drawn on top of the
feed -- corner markers while calibrating, the calibrated tank outline +
fish detection boxes once tracking -- everything this script is doing,
made visible, instead of just the console prints. Needs an actual
display (local monitor, or VNC/X11-forwarded SSH) -- a plain headless
SSH session can't show it. Press 'q' in the window (or Ctrl+C in the
terminal, same as always) to stop.

By default the console stays quiet during normal operation -- only
startup/calibration messages, request failures, and the stop message
print. Pass -v/--verbose to also print every tick's outcome (position
sent, frame dropped, object not visible, etc.).

Pass --no-detect to skip loading/running FishDetector entirely (camera
feed + calibrated tank outline only, with --debug -- nothing is ever
tracked/sent). Useful for telling apart "TFLite inference is slow" from
"the debug window itself is slow":

    python run.py --debug --no-detect
"""

import argparse
import os
import sys
import time

VISION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Vision")
sys.path.insert(0, VISION_DIR)

import cv2
import numpy as np
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
# Governs ONLY how often a position is actually POSTed to the API -- NOT
# the main loop's pace. It used to be a blanket time.sleep() at the end
# of every loop iteration, which meant the --debug window (and detection
# itself) could never update faster than the API's own throttle. Now the
# loop runs flat-out (camera read + detection + debug display, every
# tick, as fast as the hardware allows) and this just gates send_position
# calls via a last-sent timestamp -- see main().
SEND_INTERVAL_SECONDS = 0.5

# Loop-pacing sleep for the ONE case where there's genuinely nothing to
# do yet (both cameras returned no frame) -- avoids pegging a CPU core
# busy-polling cam.read() in a tight spin. Deliberately much shorter than
# SEND_INTERVAL_SECONDS, which is an API-traffic decision, not a "how
# idle can we afford to be" one.
IDLE_POLL_SECONDS = 0.05

CM_TO_INCHES = 1 / 2.54

# --debug window sizing -- each camera's feed is resized to this before
# the two are stacked side by side, purely so the combined window fits a
# normal screen regardless of the cameras' native capture resolution.
DEBUG_WINDOW_NAME = "Fish Tank Vision (run.py --debug)"
DEBUG_DISPLAY_W = 480
DEBUG_DISPLAY_H = 360

# FishDetector needs an actual .tflite model + labelmap.txt -- unlike the
# old ultralytics YOLO() call, tflite-runtime has no auto-download. Put
# both files in Vision/models/ (create the folder). A stock COCO-trained
# model (e.g. the classic "detect.tflite" + "labelmap.txt" pair from
# TensorFlow's object detection examples) proves the pipeline runs but
# has NO fish class -- same caveat the old stock yolov8n.pt had. Point
# these at fish-trained weights, converted to .tflite, for real fish
# detection. See detection.py's FishDetector docstring for the output-
# tensor-format assumption this relies on.
FISH_MODEL_PATH = os.path.join(VISION_DIR, "models", "detect.tflite")
FISH_LABELS_PATH = os.path.join(VISION_DIR, "models", "labelmap.txt")
FISH_CONF_THRESHOLD = 0.4


def _best_center(detections):
    """Pick the highest-confidence detection this tick and convert it to
    the (cx, cy, half) shape CoordinateMapper.combine() expects. None if
    nothing was detected."""
    if not detections:
        return None
    best = max(detections, key=lambda d: d[1])
    return FishDetector.center_of(best)


def _draw_quad(frame, corners):
    """Draw a 4-point quad (e.g. the calibrated tank corners) onto frame
    IN PLACE. Just cv2.polylines over already-known points -- no fitting
    needed, unlike debug_view.py's cv2.minAreaRect (this script only ever
    has the exact 4 ordered corners _find_corners returned, not a raw
    blob list to fit a box to)."""
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)


def _labeled_view(label, frame):
    """Resize one camera's frame to debug size and caption it. Pure image
    prep -- does NOT touch any window, so it's safe to call once per
    camera even though only one shared window ever gets shown per tick."""
    view = cv2.resize(frame, (DEBUG_DISPLAY_W, DEBUG_DISPLAY_H))
    cv2.putText(view, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return view


def _show_debug_frame(image):
    """Show one image in the SINGLE shared debug window and check for the
    quit key. Deliberately just one persistent window for the entire run
    (calibration AND tracking), with exactly one cv2.imshow()/waitKey()
    call per tick -- an earlier version of this opened a separate native
    window per camera and called waitKey() twice per tick, which rendered
    incorrectly (stale/garbled content bleeding through, like a leftover
    copy of the terminal) over a remote X11/VNC display. One window,
    called once per tick, is the same pattern Vision/debug_view.py
    already uses without that problem.
    Returns True if 'q' was pressed (caller should stop), else False."""
    cv2.imshow(DEBUG_WINDOW_NAME, image)
    return (cv2.waitKey(1) & 0xFF) == ord('q')


def _find_corners(cam, label, debug=False):
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
            if debug:
                view = detector.draw(frame.copy(), centers)
                view = _labeled_view(f"{label} (calibrating)", view)
                if _show_debug_frame(view):
                    raise KeyboardInterrupt
            if len(centers) >= EXPECTED_MARKERS:
                return order_quad_points(centers[:EXPECTED_MARKERS])
        time.sleep(0.1)
    raise TimeoutError(
        f"{label}: only found {max(last_count, 0)}/{EXPECTED_MARKERS} yellow corner markers "
        f"after {CORNER_DETECT_TIMEOUT_SECONDS:.0f}s. Check marker placement/lighting, or run "
        f"Vision/color_test.py to verify MARKER_LOWER/MARKER_UPPER against your camera.")


def send_position(position, verbose=False):
    """POST a position dict like {'x': .., 'y': .., 'z': ..} to the API.

    The success line is only printed when verbose -- it fires every
    SEND_INTERVAL_SECONDS while a fish is visible, so left unconditional
    it dominates the console. Failures print regardless of verbose: a
    silent-by-default run shouldn't also hide the one message that means
    something is actually broken.
    """
    try:
        response = requests.post(
            API_URL,
            json=position,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        if verbose:
            print(f"Sent {position} -> {response.status_code} {response.json()}")
    except requests.exceptions.RequestException as error:
        print(f"Request failed for {position}: {error}")


def main(debug=False, verbose=False, no_detect=False):
    print("Starting vision pipeline. Press Ctrl+C to stop." +
          (" (or 'q' in the debug window)" if debug else ""))
    if no_detect:
        print("--no-detect: fish detection is OFF -- camera feed and tank outline only, "
              "nothing will ever be tracked/sent while this is on.")
    try:
        with Camera(FRONT_CAMERA_INDEX) as front_cam, Camera(SIDE_CAMERA_INDEX) as side_cam:
            print(f"Looking for {EXPECTED_MARKERS} yellow corner markers on each camera...")
            try:
                front_corners = _find_corners(front_cam, "FRONT", debug=debug)
                side_corners = _find_corners(side_cam, "SIDE", debug=debug)
            except TimeoutError as error:
                print(error)
                return

            mapper = CoordinateMapper(front_corners, side_corners, TANK_SIZE_CM)
            print("Calibrated. Starting fish tracking.")

            # Skip even LOADING the model when no_detect -- not just
            # skipping the per-frame detect() calls -- so this mode has
            # zero TFLite cost at all, for isolating whether inference or
            # the debug window itself is the actual bottleneck.
            if no_detect:
                detector_front = detector_side = None
            else:
                detector_front = FishDetector(model_path=FISH_MODEL_PATH, labels_path=FISH_LABELS_PATH,
                                               conf=FISH_CONF_THRESHOLD)
                detector_side = FishDetector(model_path=FISH_MODEL_PATH, labels_path=FISH_LABELS_PATH,
                                              conf=FISH_CONF_THRESHOLD)

            last_sent = 0.0

            while True:
                front_frame = front_cam.read()
                side_frame = side_cam.read()

                if front_frame is None or side_frame is None:
                    if verbose:
                        print("Dropped frame, skipping this tick.")
                    time.sleep(IDLE_POLL_SECONDS)
                    continue

                if no_detect:
                    front_detections, side_detections = [], []
                else:
                    front_detections, _ = detector_front.detect(front_frame)
                    side_detections, _ = detector_side.detect(side_frame)

                if debug:
                    front_view = front_frame.copy()
                    side_view = side_frame.copy()
                    if not no_detect:
                        front_view = detector_front.draw(front_view, front_detections)
                        side_view = detector_side.draw(side_view, side_detections)
                    _draw_quad(front_view, front_corners)
                    _draw_quad(side_view, side_corners)
                    front_view = _labeled_view("FRONT", front_view)
                    side_view = _labeled_view("SIDE", side_view)
                    if _show_debug_frame(np.hstack([front_view, side_view])):
                        raise KeyboardInterrupt

                front_result = _best_center(front_detections)
                side_result = _best_center(side_detections)

                if front_result is None or side_result is None:
                    if verbose:
                        print("Object not visible in both cameras, skipping this tick.")
                    continue

                position_cm, y_diff = mapper.combine(front_result, side_result)
                if not mapper.in_tank(position_cm):
                    if verbose:
                        print(f"Detected position {position_cm}cm is outside the tank, skipping.")
                    continue

                if y_diff > 3.0 and verbose:
                    print(f"Warning: front/side cameras disagree on height by {y_diff}cm")

                # Rate-limit the actual POST to SEND_INTERVAL_SECONDS,
                # independent of how fast the loop above is running --
                # the loop itself no longer sleeps at all here, so
                # detection/debug-display cadence is unaffected by this.
                now = time.monotonic()
                if now - last_sent >= SEND_INTERVAL_SECONDS:
                    x_cm, y_cm, z_cm = position_cm
                    send_position({
                        "x": round(x_cm * CM_TO_INCHES, 2),
                        "y": round(y_cm * CM_TO_INCHES, 2),
                        "z": round(z_cm * CM_TO_INCHES, 2),
                    }, verbose=verbose)
                    last_sent = now
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish tank vision pipeline.")
    parser.add_argument("--debug", action="store_true",
                         help="also open a live window showing whatever's currently detected "
                              "(corner markers while calibrating; the calibrated tank outline + "
                              "fish detection boxes once tracking). Needs a real display (local "
                              "monitor or VNC/X11-forwarded SSH).")
    parser.add_argument("-v", "--verbose", action="store_true",
                         help="print per-tick status (every position sent, every skipped frame) "
                              "instead of running quietly. Without this, only startup/calibration "
                              "messages, request failures, and Ctrl+C/quit print anything.")
    parser.add_argument("--no-detect", action="store_true",
                         help="skip loading/running FishDetector entirely -- camera feed and "
                              "calibrated tank outline still show (with --debug), but nothing is "
                              "ever detected/tracked/sent. For isolating whether TFLite inference "
                              "or the debug window itself is a performance problem.")
    args = parser.parse_args()
    main(debug=args.debug, verbose=args.verbose, no_detect=args.no_detect)

"""
debug_view.py -- live debug viewer for the fish tank vision pipeline.

Shows both cameras side by side, live, with two things drawn automatically
every frame -- no manual calibration step:

  1. The "focus rectangle" -- the tank's visible bounds, computed from
     colored corner markers (stickers/tape/paper) you place at the tank's
     corners and leave there permanently. With EXPECTED_MARKERS (4) points
     found, this is a properly ROTATED rectangle (cv2.minAreaRect) so a
     camera that isn't perfectly level/square-on to the tank still gets
     an accurate box, not a plain axis-aligned one. The box only updates
     when the FULL expected set is found in one check -- a partial set
     (one sticker occluded/dropped out) keeps the last good box instead
     of shrinking to fit whatever's currently visible.
  2. The tracked object (red ball / fish), circled, if currently visible.

This is a development tool, not part of the production loop (run.py) --
it needs an actual display (local monitor, or VNC/X11-forwarded SSH),
which a plain headless SSH session won't have.

Controls:
    Q - quit
    M - toggle marker-mask view (shows the raw color-threshold mask used
        to find the corner markers -- literally "where the program is
        looking" for stickers -- useful for tuning MARKER_LOWER/UPPER)

Run from inside the Vision/ folder:
    python debug_view.py
"""

import cv2
import numpy as np

from capture import Camera
from detection import RedBallDetector, ColorMarkerDetector

FRONT_CAMERA_INDEX = 0
SIDE_CAMERA_INDEX = 2

# HSV range for the corner-marker stickers. Default assumes hot pink /
# magenta -- rare in typical aquarium environments (unlike blue, which
# clashes with common blue tank backgrounds/gravel). Widened + lowered
# sat/val floors vs. the "ideal" magenta swatch, since real camera
# capture (lighting, webcam color reproduction, material glossiness)
# reads noticeably less saturated/bright than a digital color picker.
# NOTE: the upper bound now edges into RedBallDetector's red range
# (170-180) -- if the tracked object starts falsely registering as a
# marker, narrow this back down using the mask view ('m') to find where
# your actual sticker's hue sits, then tighten around it.
MARKER_LOWER = (135, 60, 40)
MARKER_UPPER = (175, 255, 255)

# How many corner stickers you actually placed (2 diagonal, or 4, one per
# corner). ColorMarkerDetector returns every blob above min_area sorted
# largest-first -- capping to this count keeps a stray false-positive
# blob (a reflection, background clutter) from ever joining the min/max
# box calculation, since it'd have to out-rank a real sticker in area to
# get in.
EXPECTED_MARKERS = 4

DISPLAY_W = 640
DISPLAY_H = 480

# The corner stickers are physically static once placed -- re-locating
# them from scratch every single frame is wasted CPU. Only re-run
# ColorMarkerDetector every Nth frame; the tracked object still gets
# checked every frame since it actually moves.
MARKER_RECHECK_EVERY_N_FRAMES = 5


def _rect_from_markers(centers):
    """Turn marker centers into rectangle corners.

    With 3+ points: the minimum-area ROTATED rectangle (cv2.minAreaRect)
    that fits them -- this is what actually accounts for a camera that
    isn't perfectly level/square-on to the tank glass. A plain axis-
    aligned min/max box can't represent a tilted view at all.

    With exactly 2 points: falls back to an axis-aligned box, since two
    opposite corners alone don't constrain a rotation angle.

    Returns a list of 4 (x, y) int points, or None if fewer than 2.
    """
    if len(centers) < 2:
        return None

    if len(centers) == 2:
        (x1, y1), (x2, y2) = centers
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return [(left, top), (right, top), (right, bottom), (left, bottom)]

    points = np.array(centers, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    return [(int(x), int(y)) for x, y in box]


def _annotate(frame, object_detector, marker_detector, last_box, run_marker_detection):
    """Detect on the RAW frame first, then draw. Order matters here:
    RedBallDetector.draw() paints a blue center-dot on the object, which
    is the same color family as the default marker color -- detecting
    markers on an already-annotated frame risks picking up that dot as a
    phantom corner marker. Detecting both up front avoids that entirely.

    hsv is computed once here and handed to both detectors instead of
    each doing its own blur + color-convert -- that redundant pass was
    happening 4x per loop (2 detectors x 2 cameras).

    Parameters
    ----------
    last_box : the last computed marker rectangle (4 points), reused
        whenever this check doesn't find the full expected marker set --
        see below for why that matters.
    run_marker_detection : whether to actually re-run marker detection
        this frame, or just keep drawing the last known rectangle.

    Returns (annotated_bgr, marker_mask_bgr, box_to_reuse_next_call, hsv_for_sampling).
    """
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    object_result, _ = object_detector.detect(frame, hsv=hsv)
    frame = object_detector.draw(frame, object_result)

    if run_marker_detection:
        marker_centers, marker_mask = marker_detector.detect(frame, hsv=hsv)
        # Cap to the expected count (largest-first) so a stray
        # false-positive blob elsewhere in frame can never join the box
        # calculation -- it would have to out-rank a real sticker in
        # area to get in.
        marker_centers = marker_centers[:EXPECTED_MARKERS]

        # Only trust a FULL set. Accepting a partial set (e.g. 2 out of
        # 4 because one sticker briefly dropped below threshold) would
        # replace a good box with one sized to whatever's currently
        # visible -- that's what was causing the size to cycle between
        # too-big/too-small/half-size on every recheck.
        box = last_box
        if len(marker_centers) >= EXPECTED_MARKERS:
            box = _rect_from_markers(marker_centers)
        frame = marker_detector.draw(frame, marker_centers)
    else:
        marker_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        box = last_box

    if box is not None:
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    else:
        cv2.putText(frame, f"waiting for all {EXPECTED_MARKERS} corner markers",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    marker_mask_bgr = cv2.cvtColor(cv2.resize(marker_mask, (DISPLAY_W, DISPLAY_H)), cv2.COLOR_GRAY2BGR)
    # Resized to match the displayed frame, so a click's (x, y) in the
    # window maps directly onto this array with no extra scaling math.
    hsv_display = cv2.resize(hsv, (DISPLAY_W, DISPLAY_H))
    return frame, marker_mask_bgr, box, hsv_display


WINDOW_NAME = "Fish Tank Vision Debug"


def _make_mouse_callback(sampled_hsv):
    """Returns an OpenCV mouse callback that prints the HSV value of
    whatever pixel you click. Use this to read the color your camera
    ACTUALLY captures for a marker or for background clutter -- printer
    ink, tank lighting, and webcam color reproduction all shift a color
    from what a swatch/screen suggests, so this beats guessing at ranges.

    sampled_hsv is a dict {"front": hsv_array_or_None, "side": ...} kept
    up to date by the main loop -- the callback just reads whatever's
    currently in it when a click happens.
    """
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if x < DISPLAY_W:
            label, hsv_frame, px, py = "FRONT", sampled_hsv.get("front"), x, y
        else:
            label, hsv_frame, px, py = "SIDE", sampled_hsv.get("side"), x - DISPLAY_W, y

        if hsv_frame is None:
            return

        h, s, v = hsv_frame[py, px]
        print(f"{label} click at ({px},{py}) -> HSV ({h}, {s}, {v})")

    return on_mouse


def main():
    object_detector_front = RedBallDetector()
    object_detector_side = RedBallDetector()
    marker_detector_front = ColorMarkerDetector(MARKER_LOWER, MARKER_UPPER)
    marker_detector_side = ColorMarkerDetector(MARKER_LOWER, MARKER_UPPER)

    show_mask = False
    frame_count = 0
    front_box = None
    side_box = None

    sampled_hsv = {"front": None, "side": None}
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _make_mouse_callback(sampled_hsv))

    print("Debug viewer running. Q to quit, M to toggle marker-mask view, click to print a pixel's HSV.")
    with Camera(FRONT_CAMERA_INDEX) as front_cam, Camera(SIDE_CAMERA_INDEX) as side_cam:
        while True:
            front_frame = front_cam.read()
            side_frame = side_cam.read()

            if front_frame is None or side_frame is None:
                continue

            run_markers = (frame_count % MARKER_RECHECK_EVERY_N_FRAMES == 0)

            front_view, front_mask, front_box, front_hsv = _annotate(
                front_frame, object_detector_front, marker_detector_front, front_box, run_markers)
            side_view, side_mask, side_box, side_hsv = _annotate(
                side_frame, object_detector_side, marker_detector_side, side_box, run_markers)

            sampled_hsv["front"] = front_hsv
            sampled_hsv["side"] = side_hsv

            frame_count += 1

            cv2.putText(front_view, "FRONT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(side_view, "SIDE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            display = np.hstack([front_mask, side_mask]) if show_mask else np.hstack([front_view, side_view])

            cv2.putText(display, "Q: quit   M: toggle marker-mask   click: sample HSV",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('m'):
                show_mask = not show_mask

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

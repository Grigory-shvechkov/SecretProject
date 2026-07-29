"""
debug_view.py -- live debug viewer for the fish tank vision pipeline.

Shows both cameras side by side, live, with two things drawn automatically
every frame -- no manual calibration step:

  1. The "focus rectangle" -- the tank's visible bounds, computed each
     frame from colored corner markers (stickers/tape/paper) you place at
     the tank's corners and leave there permanently. Works with 2
     (diagonal) or all 4 markers -- the box is just the min/max of
     whichever ones are currently visible.
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

DISPLAY_W = 640
DISPLAY_H = 480

# The corner stickers are physically static once placed -- re-locating
# them from scratch every single frame is wasted CPU. Only re-run
# ColorMarkerDetector every Nth frame; the tracked object still gets
# checked every frame since it actually moves.
MARKER_RECHECK_EVERY_N_FRAMES = 5


def _annotate(frame, object_detector, marker_detector, last_bounds, run_marker_detection):
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
    last_bounds : the last computed marker bounds, reused when
        run_marker_detection is False (skips ColorMarkerDetector work
        entirely on throttled frames).
    run_marker_detection : whether to actually re-run marker detection
        this frame, or just keep drawing the last known rectangle.

    Returns (annotated_bgr, marker_mask_bgr, bounds_to_reuse_next_call).
    """
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    object_result, _ = object_detector.detect(frame, hsv=hsv)
    frame = object_detector.draw(frame, object_result)

    if run_marker_detection:
        marker_centers, marker_mask = marker_detector.detect(frame, hsv=hsv)
        # Fall back to the last known good box on a transient miss --
        # only overwrite it when this check actually finds 2+ markers.
        # Nulling it out here would flash the box off for one bad frame
        # even though nothing physically moved.
        bounds = last_bounds
        if len(marker_centers) >= 2:
            xs = [x for x, _ in marker_centers]
            ys = [y for _, y in marker_centers]
            bounds = (min(xs), max(xs), min(ys), max(ys))
        frame = marker_detector.draw(frame, marker_centers)
    else:
        marker_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        bounds = last_bounds

    if bounds is not None:
        left, right, top, bottom = bounds
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "0/2+ corner markers visible",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    marker_mask_bgr = cv2.cvtColor(cv2.resize(marker_mask, (DISPLAY_W, DISPLAY_H)), cv2.COLOR_GRAY2BGR)
    return frame, marker_mask_bgr, bounds


def main():
    object_detector_front = RedBallDetector()
    object_detector_side = RedBallDetector()
    marker_detector_front = ColorMarkerDetector(MARKER_LOWER, MARKER_UPPER)
    marker_detector_side = ColorMarkerDetector(MARKER_LOWER, MARKER_UPPER)

    show_mask = False
    frame_count = 0
    front_bounds = None
    side_bounds = None

    print("Debug viewer running. Q to quit, M to toggle marker-mask view.")
    with Camera(FRONT_CAMERA_INDEX) as front_cam, Camera(SIDE_CAMERA_INDEX) as side_cam:
        while True:
            front_frame = front_cam.read()
            side_frame = side_cam.read()

            if front_frame is None or side_frame is None:
                continue

            run_markers = (frame_count % MARKER_RECHECK_EVERY_N_FRAMES == 0)

            front_view, front_mask, front_bounds = _annotate(
                front_frame, object_detector_front, marker_detector_front, front_bounds, run_markers)
            side_view, side_mask, side_bounds = _annotate(
                side_frame, object_detector_side, marker_detector_side, side_bounds, run_markers)

            frame_count += 1

            cv2.putText(front_view, "FRONT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(side_view, "SIDE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            display = np.hstack([front_mask, side_mask]) if show_mask else np.hstack([front_view, side_view])

            cv2.putText(display, "Q: quit   M: toggle marker-mask",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("Fish Tank Vision Debug", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('m'):
                show_mask = not show_mask

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

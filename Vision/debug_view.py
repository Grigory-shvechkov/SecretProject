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
# clashes with common blue tank backgrounds/gravel) and stays clear of
# RedBallDetector's red range (170-180). Change if you used a different
# sticker color.
MARKER_LOWER = (150, 120, 70)
MARKER_UPPER = (165, 255, 255)

DISPLAY_W = 640
DISPLAY_H = 480


def _annotate(frame, object_detector, marker_detector):
    """Detect on the RAW frame first, then draw. Order matters here:
    RedBallDetector.draw() paints a blue center-dot on the object, which
    is the same color family as the default marker color -- detecting
    markers on an already-annotated frame risks picking up that dot as a
    phantom corner marker. Detecting both up front avoids that entirely.

    Returns (annotated_bgr, marker_mask_bgr), both resized for display.
    """
    object_result, _ = object_detector.detect(frame)
    marker_centers, marker_mask = marker_detector.detect(frame)

    frame = object_detector.draw(frame, object_result)
    frame = marker_detector.draw(frame, marker_centers)

    if len(marker_centers) >= 2:
        xs = [x for x, _ in marker_centers]
        ys = [y for _, y in marker_centers]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"{len(marker_centers)}/2+ corner markers visible",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    marker_mask_bgr = cv2.cvtColor(cv2.resize(marker_mask, (DISPLAY_W, DISPLAY_H)), cv2.COLOR_GRAY2BGR)
    return frame, marker_mask_bgr


def main():
    object_detector_front = RedBallDetector()
    object_detector_side = RedBallDetector()
    marker_detector_front = ColorMarkerDetector(MARKER_LOWER, MARKER_UPPER)
    marker_detector_side = ColorMarkerDetector(MARKER_LOWER, MARKER_UPPER)

    show_mask = False

    print("Debug viewer running. Q to quit, M to toggle marker-mask view.")
    with Camera(FRONT_CAMERA_INDEX) as front_cam, Camera(SIDE_CAMERA_INDEX) as side_cam:
        while True:
            front_frame = front_cam.read()
            side_frame = side_cam.read()

            if front_frame is None or side_frame is None:
                continue

            front_view, front_mask = _annotate(front_frame, object_detector_front, marker_detector_front)
            side_view, side_mask = _annotate(side_frame, object_detector_side, marker_detector_side)

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

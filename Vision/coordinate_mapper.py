"""
coordinate_mapper.py -- merge two 2D camera detections into one 3D tank
position.

Camera layout this file assumes:

    FRONT camera      -> images the tank's X (left-right) x Y (up-down) face
    LEFT-SIDE camera  -> images the tank's Z (front-back depth) x Y (up-down) face

Y is seen by BOTH cameras. We average the two Y readings and also report
how much they disagree -- a large disagreement means bad calibration or a
false detection in one view.

Coordinate convention for the output:
    X: 0 at the tank's left wall,  increasing to the right   (cm)
    Y: 0 at the water/tank TOP,    increasing downward        (cm)
       (matches image coordinates, where y grows downward;
        flip it in your own code if you prefer "up = positive")
    Z: 0 at the tank's front wall, increasing toward the back (cm)

Calibration
-----------
Pixel coordinates mean nothing physically until we know where the tank's
4 interior corners sit in each camera's image:

    corners = [(x, y), (x, y), (x, y), (x, y)]
              # top-left, top-right, bottom-right, bottom-left,
              # AS SEEN BY THAT CAMERA

All 4 corners -- not just 2 diagonal ones -- are required, and pixels are
mapped to cm with a full perspective homography (cv2.findHomography /
cv2.perspectiveTransform), not a simple per-axis min/max box. This is
what keeps positions correct even when a camera is mounted off-center or
at a slight angle to the tank glass: 2 diagonal corners can only ever
describe an AXIS-ALIGNED box in pixel space, which silently distorts
every position the moment the camera isn't perfectly square-on. 4
corners plus a homography instead captures however that rectangle
actually projects into the image -- rotated, skewed, whatever -- and
undoes exactly that projection, per pixel, on the way to cm.

Get corners via the interactive helpers at the bottom of this file:

    python coordinate_mapper.py --camera 0               # click the tank's 4 corners
    python coordinate_mapper.py --camera 0 --markers      # auto-detect 4 colored
                                                           # corner markers instead

The --markers mode looks for four physical markers (stickers/tape/paper)
placed at the tank interior's 4 corners, in a color distinct from
whatever's being tracked (default assumes yellow). No clicking needed --
point the camera, confirm all 4 markers are highlighted, press 's'.

Limitations (fine for tracking, know them anyway):
* Assumes the 4 corners are coplanar and the tank glass is flat (true for
  virtually all rectangular tanks) -- the homography is only wrong if
  that assumption breaks.
* Assumes each camera's lens has negligible distortion; a strongly
  wide-angle/fisheye lens would need distortion correction before this
  homography is accurate near the edges.
* order_quad_points() (see below) assumes a moderate tilt -- a camera
  mounted close to 45 degrees off-axis from the tank could confuse which
  corner is "top-left" vs "top-right". Not a realistic mounting for this
  project.
"""

import json
import os

import cv2
import numpy as np


class CoordinateMapper:
    """Combine front-camera and side-camera detections into (X, Y, Z) cm,
    using a full perspective homography per camera so an off-center/
    tilted camera still produces correct real-world positions."""

    def __init__(self, front_corners, side_corners, tank_size_cm):
        """
        Parameters
        ----------
        front_corners, side_corners : 4x (x, y) pixel points
            Tank interior corners AS SEEN BY THAT CAMERA, in order
            top-left, top-right, bottom-right, bottom-left. Get these
            from the interactive calibration helpers at the bottom of
            this file (order_quad_points() sorts raw marker/click points
            into this order for you).
        tank_size_cm : (width_X, height_Y, depth_Z)
            Real interior dimensions of the tank in centimeters.
        """
        self.front_corners = self._validate_quad(front_corners, "front_corners")
        self.side_corners = self._validate_quad(side_corners, "side_corners")
        self.size = tuple(tank_size_cm)
        W, H, D = self.size

        # FRONT camera images the X (width) x Y (height) face; SIDE
        # camera images the Z (depth) x Y (height) face. Each homography
        # maps that camera's 4 corner pixels onto a real-world rectangle
        # of the matching size in cm, with (0,0) at the top-left corner --
        # matching the output coordinate convention documented above.
        self._front_H = self._homography(self.front_corners, W, H)
        self._side_H = self._homography(self.side_corners, D, H)

    @staticmethod
    def _validate_quad(corners, name):
        pts = [tuple(float(v) for v in p) for p in corners]
        if len(pts) != 4:
            raise ValueError(f"{name} must have exactly 4 (x, y) points, got {len(pts)}")
        return pts

    @staticmethod
    def _homography(corners, real_w, real_h):
        src = np.array(corners, dtype=np.float32)
        dst = np.array([[0, 0], [real_w, 0], [real_w, real_h], [0, real_h]], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        if H is None:
            raise ValueError(
                "could not compute a homography from the given corners -- "
                "are they collinear, duplicated, or not actually a quadrilateral?")
        return H

    @staticmethod
    def _apply(H, px, py):
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def combine(self, front_det, side_det):
        """Merge one detection from each camera into a 3D position.

        Parameters
        ----------
        front_det : (x, y, size) from the FRONT camera.
        side_det  : (x, y, size) from the SIDE camera.
            Both in the shape FishDetector.center_of() (or
            RedBallDetector.detect()) returns.

        Returns
        -------
        pos    : (X, Y, Z) in cm, rounded to 1 decimal.
        y_diff : float, cm of disagreement between the two cameras'
                 height readings. Big values (> a few cm) mean the two
                 cameras are probably not looking at the same object,
                 or calibration is off.
        """
        fx, fy, _ = front_det
        sx, sy, _ = side_det

        X, Y_front = self._apply(self._front_H, fx, fy)
        Z, Y_side = self._apply(self._side_H, sx, sy)

        Y = (Y_front + Y_side) / 2.0
        y_diff = abs(Y_front - Y_side)

        return (round(X, 1), round(Y, 1), round(Z, 1)), round(y_diff, 1)

    def in_tank(self, pos, margin_cm=2.0):
        """True if a 3D position lies inside the tank (with a little
        tolerance). Use to reject detections of things OUTSIDE the tank
        (e.g. something the same color on the desk behind it). This check
        operates entirely in real-world cm, AFTER the homography has
        already corrected for camera rotation/tilt, so it needs no
        rotation-awareness of its own."""
        X, Y, Z = pos
        W, H, D = self.size
        m = margin_cm
        return (-m <= X <= W + m) and (-m <= Y <= H + m) and (-m <= Z <= D + m)

    # ------------------------------------------------------------------
    # saving / loading calibration
    # ------------------------------------------------------------------

    def save(self, path="calibration.json"):
        """Write calibration to a JSON file."""
        with open(path, "w") as f:
            json.dump({
                "front_corners": self.front_corners,
                "side_corners": self.side_corners,
                "tank_size_cm": self.size,
            }, f, indent=2)

    @classmethod
    def load(cls, path="calibration.json"):
        """Build a mapper from a saved JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls(d["front_corners"], d["side_corners"], d["tank_size_cm"])

    @classmethod
    def load_or_none(cls, path="calibration.json"):
        """Like load(), but returns None instead of crashing if the
        file doesn't exist -- lets run.py print a friendly hint."""
        return cls.load(path) if os.path.exists(path) else None


# ----------------------------------------------------------------------
# Point ordering
#
# Marker blobs (or manual clicks, if taken out of order) don't
# necessarily arrive in [top-left, top-right, bottom-right, bottom-left]
# order, but the homography needs them in exactly that order. Standard
# "order_points" trick: top-left has the smallest x+y, bottom-right the
# largest x+y; top-right has the smallest y-x, bottom-left the largest
# y-x. See the module docstring for the tilt assumption this relies on.
# ----------------------------------------------------------------------
def order_quad_points(points):
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]  # y - x
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return [tuple(float(v) for v in p) for p in (top_left, top_right, bottom_right, bottom_left)]


# ----------------------------------------------------------------------
# Interactive calibration helpers
#
#   python coordinate_mapper.py --camera 0
#
# Shows the live camera feed. Click the tank's 4 interior corners, IN
# ORDER: top-left, top-right, bottom-right, bottom-left (as you see them
# on screen). Clicking all 4 -- rather than just 2 diagonal corners -- is
# what lets the mapper correct for a camera that isn't perfectly
# square-on to the tank glass.
# Press 's' to print the corners, 'r' to redo, 'q' to quit.
# Run once per camera, paste the printed corners plus your tank's
# measured size into run.py (or save them with CoordinateMapper.save()).
# ----------------------------------------------------------------------

EXPECTED_MARKERS = 4


def _calibrate_camera_markers(camera_index, lower_hsv, upper_hsv):
    """Auto calibration using four colored corner markers instead of manual
    clicks. Place one marker (sticker/tape/paper) at each of the tank
    interior's 4 corners -- a color distinct from whatever's being
    tracked.

    Live preview highlights every blob of that color found. Once at
    least 4 are visible, the computed quadrilateral (ordered via
    order_quad_points) is drawn; press 's' to print the corners.
    """
    from capture import Camera
    from detection import ColorMarkerDetector

    detector = ColorMarkerDetector(lower_hsv, upper_hsv)
    win = f"Calibrate camera {camera_index} (markers) - place {EXPECTED_MARKERS} corner markers"

    with Camera(camera_index) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                break

            centers, mask = detector.detect(frame)
            frame = detector.draw(frame, centers)

            corners = None
            if len(centers) >= EXPECTED_MARKERS:
                corners = order_quad_points(centers[:EXPECTED_MARKERS])
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, "press 's' to print corners",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"found {len(centers)}/{EXPECTED_MARKERS} markers -- adjust lighting/position",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            cv2.imshow(win, frame)
            cv2.imshow(win + " - mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and corners is not None:
                print("\ncorners for this camera (top-left, top-right, bottom-right, bottom-left):")
                print(f"    {corners}\n")

    cv2.destroyAllWindows()


def _calibrate_camera(camera_index):
    from capture import Camera

    corner_labels = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            print(f"corner {len(clicks)} ({corner_labels[len(clicks) - 1]}): pixel ({x}, {y})")

    win = f"Calibrate camera {camera_index} - click tank corners"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    with Camera(camera_index) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                break

            for i, (x, y) in enumerate(clicks):
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, corner_labels[i], (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if len(clicks) == 4:
                pts = np.array(clicks, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, "press 's' to print corners, 'r' to redo",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame,
                            f"click tank {corner_labels[len(clicks)]} corner ({len(clicks)}/4)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                clicks.clear()
                print("cleared, click again")
            if key == ord('s') and len(clicks) == 4:
                print("\ncorners for this camera (top-left, top-right, bottom-right, bottom-left):")
                print(f"    {clicks}\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Tank corner calibration helper")
    p.add_argument("--camera", type=int, default=0,
                   help="camera index to calibrate (run once per camera)")
    p.add_argument("--markers", action="store_true",
                   help="auto-detect 4 colored corner markers instead of clicking")
    p.add_argument("--lower", type=int, nargs=3, default=[20, 60, 40],
                   metavar=("H", "S", "V"),
                   help="lower HSV bound for marker color (default: yellow, widened)")
    p.add_argument("--upper", type=int, nargs=3, default=[35, 255, 255],
                   metavar=("H", "S", "V"),
                   help="upper HSV bound for marker color (default: yellow, widened)")
    args = p.parse_args()

    if args.markers:
        _calibrate_camera_markers(args.camera, tuple(args.lower), tuple(args.upper))
    else:
        _calibrate_camera(args.camera)

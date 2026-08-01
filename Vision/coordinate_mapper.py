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

There's no separate calibration script/step for this anymore -- run.py
finds the 4 yellow corner markers itself, live, every time it starts (see
run.py's _find_corners()), and order_quad_points() below sorts whatever
it finds into the (top-left, top-right, bottom-right, bottom-left) order
CoordinateMapper needs.

All 4 corners -- not just 2 diagonal ones -- are used, and pixels are
mapped to cm with a full perspective homography (cv2.findHomography /
cv2.perspectiveTransform), not a simple per-axis min/max box. This is
what keeps positions correct even when a camera is mounted off-center or
at a slight angle to the tank glass: 2 diagonal corners can only ever
describe an AXIS-ALIGNED box in pixel space, which silently distorts
every position the moment the camera isn't perfectly square-on. 4
corners plus a homography instead captures however that rectangle
actually projects into the image -- rotated, skewed, whatever -- and
undoes exactly that projection, per pixel, on the way to cm.

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
            top-left, top-right, bottom-right, bottom-left. run.py
            supplies these itself (see _find_corners()) -- order_quad_points()
            below sorts raw marker points into this order for you.
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
            Both in the shape FishDetector.center_of() returns.

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
    # saving / loading calibration (optional -- run.py doesn't use these,
    # it re-detects corners fresh every run, but they're handy if you
    # want to cache a calibration yourself)
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
        file doesn't exist."""
        return cls.load(path) if os.path.exists(path) else None


# ----------------------------------------------------------------------
# Point ordering
#
# Marker blobs don't necessarily arrive in [top-left, top-right,
# bottom-right, bottom-left] order, but the homography needs them in
# exactly that order. Standard "order_points" trick: top-left has the
# smallest x+y, bottom-right the largest x+y; top-right has the smallest
# y-x, bottom-left the largest y-x. See the module docstring for the tilt
# assumption this relies on.
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

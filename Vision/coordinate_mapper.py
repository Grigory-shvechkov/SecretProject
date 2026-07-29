"""
coordinate_mapper.py -- merge two 2D camera detections into one 3D tank
position.

Camera layout this file assumes:

    FRONT camera  -> image x = tank X (left-right)
                     image y = tank Y (up-down)

    LEFT-SIDE camera -> image x = tank Z (front-back depth)
                        image y = tank Y (up-down)

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
Pixel coordinates mean nothing physically until we know where the tank
sits in each camera's image. Calibration = telling the mapper the pixel
bounds of the tank interior in each view:

    bounds = (px_left, px_right, px_top, px_bottom)

Get these by running the ball benchmark, holding the ball at the tank's
corners, and reading the printed pixel coordinates. Or use the
interactive helpers at the bottom of this file:

    python coordinate_mapper.py --camera 0               # click the tank corners
    python coordinate_mapper.py --camera 0 --markers      # auto-detect 2 colored
                                                           # corner markers instead

The --markers mode looks for two physical markers (stickers/tape/paper)
placed at diagonally-opposite interior corners of the tank, in a color
distinct from whatever's being tracked (default assumes hot pink/magenta,
since red is taken by RedBallDetector and blue commonly clashes with tank
backgrounds/gravel). No clicking needed -- point the camera,
confirm both markers are highlighted, press 's'.

Limitations (fine for tracking, know them anyway):
* Linear mapping ignores water refraction and lens distortion, so
  positions near walls/edges are approximate.
* Assumes each camera is square-on to its glass face and level.
  Straighten the cameras physically rather than compensating in code.
"""

import json
import os


class CoordinateMapper:
    """Combine front-camera and side-camera detections into (X, Y, Z) cm."""

    def __init__(self, front_bounds, side_bounds, tank_size_cm):
        """
        Parameters
        ----------
        front_bounds : (px_left, px_right, px_top, px_bottom)
            Pixel edges of the tank interior in the FRONT camera image.
        side_bounds : (px_left, px_right, px_top, px_bottom)
            Pixel edges of the tank interior in the LEFT-SIDE camera image.
            NOTE: px_left should be the tank's FRONT wall as seen from the
            side, so that Z grows from front to back. If your side camera
            sees the front wall on the image's right, swap left/right here.
        tank_size_cm : (width_X, height_Y, depth_Z)
            Real interior dimensions of the tank in centimeters.
        """
        self.front = tuple(front_bounds)
        self.side = tuple(side_bounds)
        self.size = tuple(tank_size_cm)

        # Fail fast on bad calibration instead of dividing by zero later.
        if self.front[0] == self.front[1] or self.front[2] == self.front[3]:
            raise ValueError("front_bounds have zero width or height")
        if self.side[0] == self.side[1] or self.side[2] == self.side[3]:
            raise ValueError("side_bounds have zero width or height")

    # ------------------------------------------------------------------
    # core math
    # ------------------------------------------------------------------

    @staticmethod
    def _map(px, lo, hi, real_size):
        """Linearly map a pixel coordinate to 0..real_size cm.

        px outside [lo, hi] still maps (extrapolates) -- callers can use
        the in_tank() check to reject those.
        """
        return (px - lo) / (hi - lo) * real_size

    def combine(self, front_det, side_det):
        """Merge one detection from each camera into a 3D position.

        Parameters
        ----------
        front_det : (x, y, radius_or_size) from the FRONT camera.
        side_det  : (x, y, radius_or_size) from the SIDE camera.
            Both in the shape RedBallDetector.detect() returns
            (FishDetector.center_of() produces the same shape).

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
        W, H, D = self.size

        X = self._map(fx, self.front[0], self.front[1], W)
        Y_front = self._map(fy, self.front[2], self.front[3], H)
        Y_side = self._map(sy, self.side[2], self.side[3], H)
        Z = self._map(sx, self.side[0], self.side[1], D)

        Y = (Y_front + Y_side) / 2.0
        y_diff = abs(Y_front - Y_side)

        return (round(X, 1), round(Y, 1), round(Z, 1)), round(y_diff, 1)

    def in_tank(self, pos, margin_cm=2.0):
        """True if a 3D position lies inside the tank (with a little
        tolerance). Use to reject detections of red things OUTSIDE the
        tank (e.g. something red on the desk behind it)."""
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
                "front_bounds": self.front,
                "side_bounds": self.side,
                "tank_size_cm": self.size,
            }, f, indent=2)

    @classmethod
    def load(cls, path="calibration.json"):
        """Build a mapper from a saved JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls(d["front_bounds"], d["side_bounds"], d["tank_size_cm"])

    @classmethod
    def load_or_none(cls, path="calibration.json"):
        """Like load(), but returns None instead of crashing if the
        file doesn't exist -- lets run.py print a friendly hint."""
        return cls.load(path) if os.path.exists(path) else None


# ----------------------------------------------------------------------
# Interactive calibration helper
#
#   python coordinate_mapper.py --camera 0
#
# Shows the live camera feed. Click the tank's interior corners:
#   1st click: top-left corner of the tank interior
#   2nd click: bottom-right corner of the tank interior
# Press 's' to print the bounds tuple, 'r' to redo, 'q' to quit.
# Run once per camera, paste the two tuples plus your tank's measured
# size into run.py (or save them with CoordinateMapper.save()).
# ----------------------------------------------------------------------

def bounds_from_markers(centers):
    """Turn 2+ marker-center pixel coordinates into a (px_left, px_right,
    px_top, px_bottom) bounds tuple -- the same shape front_bounds/
    side_bounds expect. Only the two extreme corners matter, so this
    works regardless of which marker is "top-left" vs "bottom-right" --
    it just takes the min/max across whatever markers were found.
    """
    xs = [x for x, _ in centers]
    ys = [y for _, y in centers]
    return (min(xs), max(xs), min(ys), max(ys))


def _calibrate_camera_markers(camera_index, lower_hsv, upper_hsv):
    """Auto calibration using two colored corner markers instead of manual
    clicks. Place a marker (sticker/tape/paper) at the tank interior's
    top-left corner and another at the bottom-right corner -- a color
    distinct from whatever RedBallDetector is tracking.

    Live preview highlights every blob of that color found. Once at
    least 2 are visible, the computed bounding box is drawn; press 's'
    to print the bounds tuple.
    """
    import cv2
    from capture import Camera
    from detection import ColorMarkerDetector

    detector = ColorMarkerDetector(lower_hsv, upper_hsv)
    win = f"Calibrate camera {camera_index} (markers) - place 2 corner markers"

    with Camera(camera_index) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                break

            centers, mask = detector.detect(frame)
            frame = detector.draw(frame, centers)

            if len(centers) >= 2:
                bounds = bounds_from_markers(centers[:2])
                left, right, top, bottom = bounds
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "press 's' to print bounds",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"found {len(centers)}/2 markers -- adjust lighting/position",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            cv2.imshow(win, frame)
            cv2.imshow(win + " - mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and len(centers) >= 2:
                print("\nbounds tuple for this camera:")
                print(f"    {bounds_from_markers(centers[:2])}\n")

    cv2.destroyAllWindows()


def _calibrate_camera(camera_index):
    import cv2
    from capture import Camera

    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))
            print(f"corner {len(clicks)}: pixel ({x}, {y})")

    win = f"Calibrate camera {camera_index} - click tank corners"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    with Camera(camera_index) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                break

            # draw clicked corners and, once we have both, the tank box
            for (x, y) in clicks:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            if len(clicks) == 2:
                cv2.rectangle(frame, clicks[0], clicks[1], (0, 255, 0), 2)
                cv2.putText(frame, "press 's' to print bounds, 'r' to redo",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame,
                            "click tank TOP-LEFT then BOTTOM-RIGHT corner",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                clicks.clear()
                print("cleared, click again")
            if key == ord('s') and len(clicks) == 2:
                (x1, y1), (x2, y2) = clicks
                left, right = sorted((x1, x2))
                top, bottom = sorted((y1, y2))
                print("\nbounds tuple for this camera:")
                print(f"    ({left}, {right}, {top}, {bottom})\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Tank corner calibration helper")
    p.add_argument("--camera", type=int, default=0,
                   help="camera index to calibrate (run once per camera)")
    p.add_argument("--markers", action="store_true",
                   help="auto-detect 2 colored corner markers instead of clicking")
    p.add_argument("--lower", type=int, nargs=3, default=[150, 120, 70],
                   metavar=("H", "S", "V"),
                   help="lower HSV bound for marker color (default: hot pink/magenta)")
    p.add_argument("--upper", type=int, nargs=3, default=[165, 255, 255],
                   metavar=("H", "S", "V"),
                   help="upper HSV bound for marker color (default: hot pink/magenta)")
    args = p.parse_args()

    if args.markers:
        _calibrate_camera_markers(args.camera, tuple(args.lower), tuple(args.upper))
    else:
        _calibrate_camera(args.camera)
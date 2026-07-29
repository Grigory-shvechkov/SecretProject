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
    R - reset the accumulated multi-click HSV average (both cameras) --
        clicking samples several pixels into a running mean instead of
        just the last one (see _make_mouse_callback), so use R when you
        want to start a fresh average for a different marker/location
        instead of restarting the whole program.
    L - lock in the currently displayed box as the trusted reference for
        outlier rejection (both cameras) -- press this once you've
        visually confirmed the box's alignment looks right. It hard-
        resets the rolling box average to exactly the box on screen right
        now, discarding any prior average, but does NOT freeze it: future
        accepted candidates still update it normally afterward, so slow
        legitimate drift (a nudged camera) keeps being tracked.

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

# Box outlier rejection (feature 2) -- see _evaluate_box_candidate.
BOX_EMA_ALPHA = 0.2
# Effective smoothing window ~= 1/BOX_EMA_ALPHA = 5 ACCEPTED checks.
# Marker rechecks only happen every MARKER_RECHECK_EVERY_N_FRAMES=5
# frames, so 5 accepted checks = ~25 raw frames (roughly 1-2s at a
# typical Pi USB-cam framerate) -- fast enough to absorb a deliberate
# slow camera nudge within a couple of seconds, slow enough that any
# single accepted-but-borderline box only nudges the average by 20%.

BOX_AREA_REL_TOLERANCE = 0.35
# Reject a candidate if its area differs from the rolling average by
# more than 35%. Ordinary per-recheck jitter (webcam noise, a
# sticker's detected centroid moving a few pixels, minor lighting
# flicker) moves the fitted rectangle's area by low single-digit
# percent. A false-positive blob replacing one real corner marker is
# a point from somewhere ELSE in the frame (background clutter, a
# reflection), not a jittered version of the true corner -- since
# cv2.minAreaRect's enclosing rectangle must stretch to cover that
# outlier point along with the 3 good ones, this typically inflates
# the fitted area by 50-100%+. 0.35 sits comfortably above normal
# jitter and comfortably below a real swap.

BOX_CENTROID_TOLERANCE_FRAC = 0.35
# Reject a candidate if its centroid shifts, from the rolling
# average's centroid, by more than 0.35 * (rolling average's
# perimeter / 4). perimeter/4 is used as a scale-invariant "average
# side length" so the same fraction works regardless of camera
# resolution/zoom. A centroid is the mean of 4 points, so one bad
# corner offset by delta shifts the centroid by delta/4 -- and
# because the false blob is generally far from the true corner (not
# merely nearby), that delta is typically a large fraction of the
# box's own size, well above 0.35x a side length. This is a backup
# signal to the area check above (catches a bad point that happens
# to have a similar area footprint but is displaced sideways).


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
    return [(int(round(x)), int(round(y))) for x, y in box]


def _box_metrics(points):
    """Rotation-invariant summary of a 4-point box: centroid, area,
    perimeter. Deliberately never touches cv2.minAreaRect's angle
    output -- that value is known to jump discontinuously between
    ranges across OpenCV versions even for a physically stationary,
    unrotated box, so it is unusable for frame-to-frame comparison.
    Centroid/area/perimeter are all invariant to which corner
    cv2.boxPoints happened to list first or what angle convention
    was used to construct an equivalent rectangle, so they sidestep
    the wraparound bug entirely rather than patching around it."""
    pts = np.array(points, dtype=np.float32)
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    area = float(cv2.contourArea(pts))
    perim = float(cv2.arcLength(pts, True))
    return cx, cy, area, perim


def _evaluate_box_candidate(stats, candidate_points):
    """stats is None (no history yet) or a dict {"cx","cy","area","perim"}.
    Returns (accept: bool, new_stats: dict). On reject, new_stats is
    returned UNCHANGED -- a rejected candidate never pollutes the
    rolling average, only an accepted one does."""
    cx, cy, area, perim = _box_metrics(candidate_points)

    if stats is None:
        # Bootstrap: the very first-ever candidate for this side has
        # nothing to compare against. Always accept, and seed the
        # average directly from this box's own metrics (not
        # EMA-blended -- there is no prior mean to blend with).
        return True, {"cx": cx, "cy": cy, "area": area, "perim": perim}

    area_dev = abs(area - stats["area"]) / max(stats["area"], 1.0)
    side_len = max(stats["perim"], 1.0) / 4.0
    centroid_shift = ((cx - stats["cx"]) ** 2 + (cy - stats["cy"]) ** 2) ** 0.5

    if area_dev > BOX_AREA_REL_TOLERANCE or centroid_shift > BOX_CENTROID_TOLERANCE_FRAC * side_len:
        return False, stats

    a = BOX_EMA_ALPHA
    new_stats = {
        "cx": (1 - a) * stats["cx"] + a * cx,
        "cy": (1 - a) * stats["cy"] + a * cy,
        "area": (1 - a) * stats["area"] + a * area,
        "perim": (1 - a) * stats["perim"] + a * perim,
    }
    return True, new_stats


def _lock_box(points):
    """Hard-reset seed for lock-in: returns a fresh stats dict built
    directly from the given box's own metrics, discarding any prior
    EMA state entirely (identical math to the bootstrap branch of
    _evaluate_box_candidate)."""
    cx, cy, area, perim = _box_metrics(points)
    return {"cx": cx, "cy": cy, "area": area, "perim": perim}


def _annotate(frame, object_detector, marker_detector, last_box, run_marker_detection, box_stats, label=""):
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
    box_stats : rolling-average stats dict for THIS side (see
        _evaluate_box_candidate), or None if no box has been accepted
        yet. Even when a full marker set is found, the candidate box
        built from it still has to pass this outlier check before it's
        accepted/drawn -- guards against a false-positive blob (a
        reflection, background clutter) that happened to out-rank a
        real sticker in area and got treated as one of the "full set" of
        EXPECTED_MARKERS, which would otherwise still make it into
        cv2.minAreaRect via _rect_from_markers and produce a badly wrong
        (but still "complete") box.
    label : "FRONT"/"SIDE", used only for the console message printed
        when a candidate is rejected.

    Returns (annotated_bgr, marker_mask_bgr, box_to_reuse_next_call,
    box_stats_to_reuse_next_call, hsv_for_sampling).
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
            candidate = _rect_from_markers(marker_centers)
            # A "full set" can still be WRONG: a false-positive blob can
            # out-rank a real sticker in area and sit among the top-N,
            # producing a complete-but-badly-wrong rectangle. Gate the
            # candidate against the rolling average before trusting it.
            accepted, box_stats = _evaluate_box_candidate(box_stats, candidate)
            if accepted:
                box = candidate
            else:
                print(f"{label} box candidate rejected as outlier "
                      f"(area/centroid deviate from rolling average); "
                      f"reusing last known-good box.")
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
    return frame, marker_mask_bgr, box, box_stats, hsv_display


WINDOW_NAME = "Fish Tank Vision Debug"


def _new_hsv_stats():
    """Fresh, empty running-average accumulator for one camera side."""
    return {"n": 0, "sin_sum": 0.0, "cos_sum": 0.0, "s_sum": 0.0, "v_sum": 0.0}


def _update_hsv_stats(stats, h, s, v):
    """H uses a circular mean (accumulate as a unit vector at angle
    h*2 degrees, recover via atan2) since OpenCV hue is circular over
    [0,180) representing the full 0-360 degree wheel, and this
    project's own MARKER_UPPER=175 sits only 5 degrees from that
    seam -- a plain arithmetic mean would silently break the moment
    clicked samples straddle it (e.g. 178 and 2 averaging to ~90,
    which is nowhere near either). S and V are ordinary linear
    channels and use a plain running-sum mean."""
    stats["n"] += 1
    angle = np.deg2rad(float(h) * 2.0)
    stats["sin_sum"] += np.sin(angle)
    stats["cos_sum"] += np.cos(angle)
    stats["s_sum"] += float(s)
    stats["v_sum"] += float(v)


def _hsv_mean(stats):
    """Returns (mean_h, mean_s, mean_v, n), or None if no samples yet."""
    if stats["n"] == 0:
        return None
    mean_h = (np.degrees(np.arctan2(stats["sin_sum"], stats["cos_sum"])) / 2.0) % 180.0
    # atan2 can land a hair below 0 for samples straddling the exact
    # hue wrap seam (e.g. h=178 and h=2, whose unit vectors nearly
    # cancel in sin), which the % 180.0 above turns into ~179.99999...
    # instead of ~0.0 -- the same hue point in OpenCV's 0-179 space,
    # just on the wrong side of the seam due to floating-point sign.
    # Snap that artifact back to 0.0 rather than displaying "180.0".
    if mean_h > 180.0 - 1e-6:
        mean_h = 0.0
    mean_s = stats["s_sum"] / stats["n"]
    mean_v = stats["v_sum"] / stats["n"]
    return mean_h, mean_s, mean_v, stats["n"]


def _make_mouse_callback(sampled_hsv, hsv_stats):
    """Returns an OpenCV mouse callback that prints the HSV value of
    whatever pixel you click. Use this to read the color your camera
    ACTUALLY captures for a marker or for background clutter -- printer
    ink, tank lighting, and webcam color reproduction all shift a color
    from what a swatch/screen suggests, so this beats guessing at ranges.

    Each click also folds into a running mean HSV for that side (see
    _update_hsv_stats/_hsv_mean) instead of only reporting that one raw
    pixel -- a single click can land on a shadowed/glare-y patch of a
    marker, so clicking several different markers/spots and reading the
    running average gives a far more representative color to threshold
    against. Press 'r' (see main()) to reset the accumulator and start a
    fresh average for a different color/location.

    sampled_hsv is a dict {"front": hsv_array_or_None, "side": ...} kept
    up to date by the main loop -- the callback just reads whatever's
    currently in it when a click happens.

    hsv_stats is a dict {"front": stats_dict, "side": stats_dict} (see
    _new_hsv_stats) that this callback updates and reads in place, and
    that main()'s 'r' key handler resets.
    """
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if x < DISPLAY_W:
            label, hsv_frame, px, py, side_key = "FRONT", sampled_hsv.get("front"), x, y, "front"
        else:
            label, hsv_frame, px, py, side_key = "SIDE", sampled_hsv.get("side"), x - DISPLAY_W, y, "side"

        if hsv_frame is None:
            return

        # Defensive bounds check: currently always in-range because
        # cv2.namedWindow(WINDOW_NAME) uses the default WINDOW_AUTOSIZE
        # flag (the window can't be manually resized, so click coords
        # always fall within the displayed frame) -- but indexing
        # unchecked would be fragile if that ever changes.
        frame_h, frame_w = hsv_frame.shape[:2]
        if not (0 <= px < frame_w and 0 <= py < frame_h):
            return

        h, s, v = hsv_frame[py, px]
        _update_hsv_stats(hsv_stats[side_key], h, s, v)
        mean_h, mean_s, mean_v, n = _hsv_mean(hsv_stats[side_key])
        print(f"{label} click #{n} at ({px},{py}) -> HSV ({h}, {s}, {v}) | "
              f"running mean HSV ({mean_h:.1f}, {mean_s:.1f}, {mean_v:.1f}) over {n} sample(s)")

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
    # Rolling-average box stats per side, for outlier rejection (feature
    # 2). None until the first box is ever accepted (or until 'l' locks
    # one in) -- see _evaluate_box_candidate for why None always accepts.
    front_box_stats = None
    side_box_stats = None

    sampled_hsv = {"front": None, "side": None}
    hsv_stats = {"front": _new_hsv_stats(), "side": _new_hsv_stats()}
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _make_mouse_callback(sampled_hsv, hsv_stats))

    print("Debug viewer running. Q to quit, M to toggle marker-mask view, "
          "R to reset HSV click average, L to lock in the current box, "
          "click to sample HSV (accumulates a running average).")
    with Camera(FRONT_CAMERA_INDEX) as front_cam, Camera(SIDE_CAMERA_INDEX) as side_cam:
        while True:
            front_frame = front_cam.read()
            side_frame = side_cam.read()

            if front_frame is None or side_frame is None:
                continue

            run_markers = (frame_count % MARKER_RECHECK_EVERY_N_FRAMES == 0)

            front_view, front_mask, front_box, front_box_stats, front_hsv = _annotate(
                front_frame, object_detector_front, marker_detector_front, front_box,
                run_markers, front_box_stats, label="FRONT")
            side_view, side_mask, side_box, side_box_stats, side_hsv = _annotate(
                side_frame, object_detector_side, marker_detector_side, side_box,
                run_markers, side_box_stats, label="SIDE")

            sampled_hsv["front"] = front_hsv
            sampled_hsv["side"] = side_hsv

            frame_count += 1

            cv2.putText(front_view, "FRONT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(side_view, "SIDE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            display = np.hstack([front_mask, side_mask]) if show_mask else np.hstack([front_view, side_view])

            cv2.putText(display, "Q: quit  M: mask  R: reset HSV samples  L: lock-in box  click: sample HSV (avg)",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('m'):
                show_mask = not show_mask

            if key == ord('r'):
                hsv_stats["front"] = _new_hsv_stats()
                hsv_stats["side"] = _new_hsv_stats()
                print("HSV click accumulator reset (front & side).")

            if key == ord('l'):
                # Hard-reset the rolling box average to exactly what's
                # currently on screen -- the user has visually confirmed
                # it looks right, so it should immediately become the
                # trusted reference, overriding whatever the automatic
                # EMA had been tracking (which might be contaminated by
                # a run of borderline-but-accepted candidates). This is
                # NOT a permanent freeze: _evaluate_box_candidate runs
                # exactly as normal afterward, so the system keeps
                # absorbing genuinely-good future candidates and can
                # still track slow legitimate drift (e.g. a later camera
                # nudge) without needing 'l' pressed again.
                if front_box is not None:
                    front_box_stats = _lock_box(front_box)
                    print(f"FRONT box locked in as trusted reference (area={front_box_stats['area']:.0f}).")
                else:
                    print("FRONT: no box currently displayed, nothing to lock in.")
                if side_box is not None:
                    side_box_stats = _lock_box(side_box)
                    print(f"SIDE box locked in as trusted reference (area={side_box_stats['area']:.0f}).")
                else:
                    print("SIDE: no box currently displayed, nothing to lock in.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

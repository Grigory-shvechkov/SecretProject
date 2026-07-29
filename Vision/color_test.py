"""
color_test.py -- empirical HSV-range picker for the corner-marker color.

WHY THIS EXISTS: guessing marker HSV ranges from a digital swatch doesn't
work. "Hot pink" or "magenta" on a screen is not what a webcam sees once
that color is printed on paper/tape and lit by real tank lighting -- ink
reproduces darker and duller than its digital source, so a range picked by
eye needs widening just to see the marker at all... and a widened range
starts also matching background clutter (reflections, gravel, tank trim),
especially near the bottom of frame. That failure mode is invisible until
you actually test the derived range against a live scene.

So instead of typing in numbers, this tool measures them: hold up a
candidate color sample (a scrap of tape/paper/sticker) to a REAL camera
under the REAL tank lighting, drag a rectangle over it in the live feed,
and the tool derives an HSV range from what the sensor actually captured.
It then immediately tests that exact derived range against the live scene
using ColorMarkerDetector -- the SAME class debug_view.py, coordinate_
mapper.py and run.py use in production -- so "looks clean here" is
guaranteed to mean "will behave identically once pasted into those files."
Multiple candidates can be sampled into numbered slots in one session and
ranked by a numeric score, so the choice of marker color stops being a
guess and becomes a measurement.

This is a standalone diagnostic tool, run manually, NOT part of the
production loop (run.py). Like debug_view.py, it needs an actual display
(local monitor, or VNC/X11-forwarded SSH) -- a plain headless SSH session
won't show the windows it opens.

Interaction, start to finish
-----------------------------
1. Press a number key 1-9 to "arm" a slot. The HUD shows which slot is
   armed and its label (if any).
2. Hold the physical candidate up to the camera. Left-click-drag a
   rectangle over its color in the live feed and release -- this samples
   the armed slot from the pixels under that rectangle (mean + stddev in
   HSV) and immediately derives + tests an HSV range for it. Repeat for
   as many candidates as you want to compare, arming a different slot
   each time.
3. (Recommended) Point the camera at the empty tank -- nothing held up --
   and press 'b' to capture a background reference. Every slot's score
   then also reflects how much of the REAL background it would falsely
   light up, which is the specific failure mode ("catches clutter near
   the bottom of frame") that motivated this tool.
4. Press 'e' any time to re-test every sampled slot's derived range
   against the CURRENT live frame (and the background reference, if
   captured) -- useful after moving a candidate to a different spot in
   the tank, or after adjusting the range width with '[' / ']'.
5. Press 'p' (or just 'q' to quit) to print the ranking: every candidate,
   sorted by score, with its derived (H, S, V) lower/upper tuple printed
   in the exact copy-pasteable form MARKER_LOWER/MARKER_UPPER use in
   debug_view.py, coordinate_mapper.py, and ColorMarkerDetector's
   defaults.

Keybindings
-----------
    1-9   arm that slot number for the next drag-sample
    drag  (left mouse button, on the live feed) sample the armed slot's
          color from the rectangle you draw, then auto-derive + auto-test
          its range against the current frame
    b     capture (or replace) the background reference frame
    e     re-evaluate ALL sampled slots against the current live frame
          and the background reference, without re-sampling their color
    l     label the armed slot (typed at the console)
    c     clear the armed slot's sample (start that slot over)
    r     reset EVERYTHING -- all slots and the background reference
          (console confirmation required, since this is destructive)
    [ ]   decrease / increase the range-width multiplier k (see below)
    m     toggle a second window showing the armed slot's derived-range
          mask against the live feed (kept as a SEPARATE window, unlike
          debug_view.py's in-place mask toggle -- sampling must always
          happen by dragging on the real color image, so the main window
          can never be swapped out for a mask view the way debug_view
          swaps its single display)
    p     print the full ranking + copy-pasteable HSV tuples to console
    q     print the final ranking, then quit

Run from inside the Vision/ folder:
    python color_test.py                  # camera 0
    python color_test.py --camera 2       # a different USB camera index
"""

import argparse
import math

import cv2
import numpy as np

from capture import Camera
from detection import ColorMarkerDetector

# ----------------------------------------------------------------------
# Tunables. Keep these in sync with production defaults (ColorMarkerDetector
# in detection.py) so a range that scores well here behaves IDENTICALLY
# once pasted into debug_view.py / coordinate_mapper.py -- testing against
# a different min_area than production would use is testing the wrong
# thing.
# ----------------------------------------------------------------------
DEFAULT_MIN_AREA = 150          # matches ColorMarkerDetector's own default

# Range-width multiplier: derived range = mean +/- k * stddev per channel.
# 2.5 std devs covers ~98.8% of a roughly-normal distribution -- wide
# enough to survive normal frame-to-frame sensor noise and minor lighting
# flicker, without ballooning so wide it starts overlapping a neighboring
# color's hue. Adjustable live with '[' / ']' since the "right" width is
# a judgment call that depends on how noisy the room lighting actually is.
DEFAULT_K = 2.5
K_MIN, K_MAX, K_STEP = 1.0, 4.0, 0.25

# Minimum half-width applied AFTER the k*stddev calculation, per channel.
# Reasoning: a very small/very evenly-lit sample rectangle can produce a
# near-zero stddev, which would derive a razor-thin range that only
# matches the exact pixels sampled -- and fails the instant the candidate
# tilts slightly or a lighting flicker shifts brightness. These floors are
# sized off the ranges already proven to work elsewhere in this project
# (MARKER_LOWER/UPPER in debug_view.py spans a 40-wide hue band and
# near-full-width sat/val bands), so a derived range is never allowed to
# be narrower than real capture noise in this project has already shown
# it needs to be.
H_MIN_HALF = 4
S_MIN_HALF = 20
V_MIN_HALF = 20

# Sanity-check slack (pixels) for "is the biggest detected blob actually
# where I dragged the sample rectangle, or did the derived range just
# find something else in the scene and call IT the biggest blob instead."
POSITION_SLACK_PX = 20

# HUD text.
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Color Candidate Tester"
MASK_WINDOW_NAME = "Mask Preview"


# ----------------------------------------------------------------------
# Circular hue statistics.
#
# OpenCV hue is a WHEEL (0-180, where 180 wraps back to 0), not a line.
# A plain mean/stddev of hue values is wrong near that wraparound: e.g.
# samples of {178, 2} are two hues apart on the wheel, but a linear mean
# gives 90 (a completely different color) instead of ~0/180. Saturation
# and value have no such wraparound (they really are 0-255 lines), so
# only hue needs this treatment.
# ----------------------------------------------------------------------
def _circular_hue_mean_std(hue_values):
    """mean and stddev of a set of OpenCV hues (0-180), correctly handling
    the wraparound at 0/180. Returns (mean_hue, circ_std), both in the
    same 0-180 hue units.

    Standard circular-statistics formulas: treat each hue as an angle
    (doubled, since OpenCV's 0-180 represents a full 0-360 degree wheel),
    average as unit vectors, then convert the resultant vector's angle
    and length back into hue units. Resultant length R is 1.0 for
    perfectly consistent hue and shrinks toward 0.0 as hues scatter
    around the wheel -- circ_std = sqrt(-2 * ln(R)) is the standard
    angular stddev derived from that length.
    """
    angles = hue_values.astype(np.float64) * (2.0 * np.pi / 180.0)
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    mean_angle = math.atan2(sin_mean, cos_mean)
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    mean_hue = mean_angle * (180.0 / (2.0 * np.pi))

    resultant = math.sqrt(sin_mean ** 2 + cos_mean ** 2)
    resultant = max(resultant, 1e-9)  # guard log(0) for a fully-scattered sample
    circ_std = math.sqrt(-2.0 * math.log(resultant)) * (180.0 / (2.0 * np.pi))
    return mean_hue, circ_std


def _derive_range(mean_hsv, std_hsv, k):
    """mean +/- k*stddev per channel, floored to the MIN_HALF constants
    and clipped to valid OpenCV HSV bounds (H: 0-180, S/V: 0-255).

    Returns (lower, upper, hue_wraps) where lower/upper are int (H, S, V)
    tuples ready to paste into MARKER_LOWER/MARKER_UPPER, and hue_wraps
    is True if the true derived hue interval would need to cross the
    0/180 boundary to represent this color (see the warning this
    triggers in _evaluate_slot / the console report -- ColorMarkerDetector
    only supports ONE contiguous range, unlike RedBallDetector's two-range
    wraparound workaround, so a candidate that needs wraparound cannot be
    made to work with ColorMarkerDetector at all; this tool must surface
    that instead of silently clipping into a broken range and calling it
    fine).
    """
    h_mean, s_mean, v_mean = mean_hsv
    h_std, s_std, v_std = std_hsv

    h_half = max(k * h_std, H_MIN_HALF)
    s_half = max(k * s_std, S_MIN_HALF)
    v_half = max(k * v_std, V_MIN_HALF)

    h_lo_raw, h_hi_raw = h_mean - h_half, h_mean + h_half
    hue_wraps = h_lo_raw < 0 or h_hi_raw > 180

    lower = (
        int(round(np.clip(h_lo_raw, 0, 180))),
        int(round(np.clip(s_mean - s_half, 0, 255))),
        int(round(np.clip(v_mean - v_half, 0, 255))),
    )
    upper = (
        int(round(np.clip(h_hi_raw, 0, 180))),
        int(round(np.clip(s_mean + s_half, 0, 255))),
        int(round(np.clip(v_mean + v_half, 0, 255))),
    )
    return lower, upper, hue_wraps


# ----------------------------------------------------------------------
# Candidate slot
# ----------------------------------------------------------------------
class Sample:
    """Everything known about one candidate color, sampled into one of
    the numbered slots. Score fields start as None ("not yet evaluated")
    and are filled in by _evaluate_slot -- kept separate from the raw
    sampled stats so changing k or re-testing never has to re-sample.
    """

    def __init__(self, label, mean_hsv, std_hsv, n_pixels, rect):
        self.label = label
        self.mean_hsv = mean_hsv      # (h, s, v) floats, as actually captured
        self.std_hsv = std_hsv        # (h, s, v) floats
        self.n_pixels = n_pixels
        self.rect = rect              # (x1, y1, x2, y2) the sample was dragged from
        self.lower = None
        self.upper = None
        self.hue_wraps = None

        # score fields -- None means "press 'e' to (re-)evaluate"
        self.consistency_score = None
        self.segmentation_score = None
        self.background_score = None
        self.final_score = None
        self.num_live_blobs = None
        self.clutter_ratio = None
        self.position_ok = None
        self.num_bg_blobs = None
        self.bg_area_px = None

    def rect_center_and_size(self):
        x1, y1, x2, y2 = self.rect
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), (x2 - x1), (y2 - y1)

    def rederive(self, k):
        """Recompute lower/upper from the (unchanged) sampled stats and
        invalidate stale scores, since they were computed against the OLD
        range."""
        self.lower, self.upper, self.hue_wraps = _derive_range(
            self.mean_hsv, self.std_hsv, k)
        self.consistency_score = None
        self.segmentation_score = None
        self.background_score = None
        self.final_score = None
        self.num_live_blobs = None
        self.clutter_ratio = None
        self.position_ok = None
        self.num_bg_blobs = None
        self.bg_area_px = None


def _sample_rectangle(hsv_frame, rect):
    """Mean + stddev, in HSV, of every pixel inside rect=(x1,y1,x2,y2).
    Hue uses circular stats (see _circular_hue_mean_std); sat/val use
    plain stats since they don't wrap.
    """
    x1, y1, x2, y2 = rect
    patch = hsv_frame[y1:y2, x1:x2]
    h_vals = patch[:, :, 0].reshape(-1)
    s_vals = patch[:, :, 1].reshape(-1).astype(np.float64)
    v_vals = patch[:, :, 2].reshape(-1).astype(np.float64)

    h_mean, h_std = _circular_hue_mean_std(h_vals)
    mean_hsv = (h_mean, float(np.mean(s_vals)), float(np.mean(v_vals)))
    std_hsv = (h_std, float(np.std(s_vals)), float(np.std(v_vals)))
    return mean_hsv, std_hsv, h_vals.size


# ----------------------------------------------------------------------
# Scoring
#
# Three sub-scores (0-100, higher = better), combined into one final
# score. Each is documented at the point it's computed below with WHY
# it's weighted the way it is; the short version:
#
#   consistency   (30%, or 40% if no background captured) -- how tightly
#                 the sampled pixels clustered. This is the leading
#                 indicator: a color that varies a lot under CURRENT
#                 lighting is unlikely to hold up later even if it
#                 happens to segment cleanly in this one snapshot.
#   segmentation  (40%, or 60% if no background captured) -- does the
#                 derived range, run through the REAL ColorMarkerDetector
#                 pipeline, find exactly one clean blob roughly where the
#                 sample was taken, in the CURRENT live scene? This is
#                 the most direct test there is: it is not a proxy, it is
#                 the actual production code path being exercised live.
#   background    (30%, when a reference frame was captured) -- how much
#                 of an EMPTY-TANK frame this range would incorrectly
#                 light up. This is the single most valuable signal for
#                 the specific problem that motivated this tool (a wide
#                 range catching background clutter), but it's optional
#                 (requires an extra capture step), so it can't be
#                 mandatory or dominant the way segmentation is.
# ----------------------------------------------------------------------
def _blob_areas(mask, min_area):
    """Areas (largest first) of every contour in mask that meets
    min_area. NOTE: this only recovers geometry (area) from a mask that
    ColorMarkerDetector.detect() already produced -- the blur, HSV
    conversion, inRange thresholds and erode/dilate that actually
    classify pixels all happened inside detect(), unchanged. We are
    reading the SAME mask the real detector returns, not reclassifying
    pixels a second time with different logic.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    areas = [a for a in areas if a >= min_area]
    areas.sort(reverse=True)
    return areas


def _evaluate_slot(sample, live_frame, live_hsv, background_frame, background_hsv,
                    min_area):
    """(Re-)score one slot against the current live frame (and background
    reference, if captured). Mutates sample's score fields in place.
    """
    detector = ColorMarkerDetector(sample.lower, sample.upper, min_area=min_area)

    # --- consistency: from the SAMPLED stddev, not the live scene ---
    # Hue is weighted ~3x heavier than sat/val: hue is what actually
    # distinguishes this color from every OTHER color in the scene, so
    # variance there is the dangerous kind. Sat/val naturally swing more
    # from glare, shadow and distance-to-light -- production ranges
    # already documented in this project (e.g. MARKER_LOWER/UPPER's
    # 60-255 / 40-255 sat/val floors in debug_view.py) treat that as
    # normal, so this score shouldn't punish it as harshly.
    h_std, s_std, v_std = sample.std_hsv
    consistency = 100.0 - (2.0 * h_std + 0.7 * s_std + 0.7 * v_std)
    consistency = max(0.0, min(100.0, consistency))
    sample.consistency_score = round(consistency, 1)

    # --- segmentation: run the REAL detector on the CURRENT live frame ---
    centers, live_mask = detector.detect(live_frame, hsv=live_hsv)
    areas = _blob_areas(live_mask, detector.min_area)
    num_blobs = len(areas)
    sample.num_live_blobs = num_blobs

    if num_blobs == 0:
        # The range this candidate produced can't even find itself in
        # the exact frame it was just tested against -- an outright fail.
        segmentation = 0.0
        sample.clutter_ratio = None
        sample.position_ok = False
    else:
        main_area, extra_area = areas[0], sum(areas[1:])
        clutter_ratio = (extra_area / main_area) if main_area > 0 else 1.0
        clutter_ratio_capped = min(clutter_ratio, 1.0)
        sample.clutter_ratio = round(clutter_ratio, 3)

        # Sanity check: is the BIGGEST blob actually near where the
        # sample rectangle was dragged? If the derived range instead
        # found some other object in the scene as its largest blob, a
        # clean-looking number here would be actively misleading.
        (rx, ry), rw, rh = sample.rect_center_and_size()
        main_x, main_y = centers[0]
        dist = math.hypot(main_x - rx, main_y - ry)
        diag = math.hypot(rw, rh)
        position_ok = dist <= (diag / 2.0 + POSITION_SLACK_PX)
        sample.position_ok = position_ok

        # Every EXTRA qualifying blob beyond the first is penalized
        # heavily (40 pts each) -- that's literally the false-positive
        # failure mode this tool exists to catch. clutter_ratio adds a
        # second, finer-grained penalty for how much stray area those
        # extra blobs amount to relative to the real one, since "one
        # small extra fleck" and "one extra blob the same size as the
        # real marker" shouldn't score the same.
        segmentation = 100.0 - 40.0 * (num_blobs - 1) - 100.0 * clutter_ratio_capped
        segmentation = max(0.0, min(100.0, segmentation))
        if not position_ok:
            segmentation = min(segmentation, 30.0)

    sample.segmentation_score = round(segmentation, 1)

    # --- background: how much of an EMPTY-TANK frame lights up ---
    background = None
    if background_hsv is not None:
        _, bg_mask = detector.detect(background_frame, hsv=background_hsv)
        bg_areas = _blob_areas(bg_mask, detector.min_area)
        num_bg_blobs = len(bg_areas)
        bg_area_px = int(cv2.countNonZero(bg_mask))
        sample.num_bg_blobs = num_bg_blobs
        sample.bg_area_px = bg_area_px

        # Area-based penalty is scaled by min_area, not raw frame size,
        # because what matters isn't "what fraction of the whole image"
        # but "how many marker-sized false blobs' worth of area is this"
        # -- that's the unit a false detection actually gets measured in
        # downstream (EXPECTED_MARKERS logic in debug_view.py). On top of
        # that, every qualifying false blob is a DIRECT hit (a literal
        # phantom marker in production), so it's penalized again, harder,
        # per blob.
        background = 100.0 / (1.0 + bg_area_px / (5.0 * detector.min_area))
        background -= 20.0 * num_bg_blobs
        background = max(0.0, min(100.0, background))
        sample.background_score = round(background, 1)
    else:
        sample.background_score = None

    # --- weighted final ---
    if background is None:
        final = 0.4 * consistency + 0.6 * segmentation
    else:
        final = 0.3 * consistency + 0.4 * segmentation + 0.3 * background
    sample.final_score = round(final, 1)


# ----------------------------------------------------------------------
# Console report
# ----------------------------------------------------------------------
def _print_report(slots):
    filled = [(n, s) for n, s in sorted(slots.items()) if s is not None]
    if not filled:
        print("\nNo candidates sampled yet -- arm a slot (1-9) and drag over a color.\n")
        return

    def sort_key(item):
        _, s = item
        # unscored slots sort last; None can't compare to float directly
        return (s.final_score is None, -(s.final_score or 0))

    ranked = sorted(filled, key=sort_key)

    print("\n" + "=" * 72)
    print("CANDIDATE RANKING")
    print("=" * 72)
    def _fmt(v):
        # Score fields are None right after a sample is taken/relabeled
        # or after 'k' changes (rederive() invalidates them on purpose --
        # see Sample.rederive) until 'e' is pressed again. Never format
        # None with a numeric spec; show "n/a" instead.
        return f"{v:5.1f}" if v is not None else "  n/a"

    for rank, (n, s) in enumerate(ranked, start=1):
        wrap_flag = " [WRAPS 0/180 -- see note below]" if s.hue_wraps else ""
        print(f"#{rank}  slot {n} \"{s.label}\"{wrap_flag}")
        print(f"      final={_fmt(s.final_score)}  consistency={_fmt(s.consistency_score)}  "
              f"segmentation={_fmt(s.segmentation_score)}  background={_fmt(s.background_score)}")
        print(f"      sampled HSV mean=({s.mean_hsv[0]:.1f}, {s.mean_hsv[1]:.1f}, "
              f"{s.mean_hsv[2]:.1f})  std=({s.std_hsv[0]:.1f}, {s.std_hsv[1]:.1f}, "
              f"{s.std_hsv[2]:.1f})  n={s.n_pixels}px")
        if s.lower is not None:
            print(f"      MARKER_LOWER = {s.lower}")
            print(f"      MARKER_UPPER = {s.upper}")
        if s.final_score is None:
            print("      (not yet evaluated -- press 'e')")
        if s.num_live_blobs == 0:
            print("      WARNING: this range found NOTHING in the live frame it was just tested against.")
        elif s.position_ok is False:
            print("      WARNING: the biggest blob found was NOT near where you sampled -- "
                  "this range is probably matching something else in the scene.")
        print()

    best_n, best = ranked[0]
    print("-" * 72)
    if best.final_score is None:
        print("Best candidate not yet determined -- press 'e' to evaluate sampled slots.")
    else:
        print(f"WINNER: slot {best_n} \"{best.label}\" (score {best.final_score})")
        print("Paste directly into MARKER_LOWER / MARKER_UPPER "
              "(debug_view.py, coordinate_mapper.py, ColorMarkerDetector):")
        print(f"    MARKER_LOWER = {best.lower}")
        print(f"    MARKER_UPPER = {best.upper}")
        if best.hue_wraps:
            print("    NOTE: this candidate's hue straddles the OpenCV 0/180 "
                  "wraparound (like red does). ColorMarkerDetector only supports "
                  "ONE contiguous range -- see RedBallDetector for the two-range "
                  "workaround red needs. Prefer a different candidate whose hue "
                  "sits away from the 0/180 edge, or this range will not behave "
                  "the way it just tested here once real conditions shift.")
    print("=" * 72 + "\n")


# ----------------------------------------------------------------------
# HUD drawing
# ----------------------------------------------------------------------
def _draw_hud(frame, state):
    active = state["active_slot"]
    if active is None:
        active_str = "ACTIVE SLOT: none (press 1-9)"
    else:
        s = state["slots"][active]
        if s is not None:
            label = s.label
        elif state["labels"].get(active):
            label = f"{state['labels'][active]} (drag to sample)"
        else:
            label = "(empty -- drag to sample)"
        active_str = f"ACTIVE SLOT: {active}  {label}"

    bg_str = "captured" if state["background_hsv"] is not None else "NOT captured (press 'b')"
    cv2.putText(frame, f"{active_str}   k={state['k']:.2f}   background: {bg_str}",
                (10, 22), FONT, 0.55, (255, 255, 255), 2)
    cv2.putText(frame,
                "1-9 arm | drag: sample | e: evaluate | b: background | "
                "l: label | c: clear | [ ]: k | m: mask | p: report | q: quit",
                (10, frame.shape[0] - 10), FONT, 0.42, (180, 180, 180), 1)

    # live-updating ranked mini-table, top-right
    filled = [(n, s) for n, s in sorted(state["slots"].items()) if s is not None]
    ranked = sorted(filled, key=lambda item: (item[1].final_score is None,
                                               -(item[1].final_score or 0)))
    y = 50
    for n, s in ranked:
        score_str = f"{s.final_score:5.1f}" if s.final_score is not None else " n/a"
        color = (0, 255, 0) if n == active else (255, 255, 255)
        cv2.putText(frame, f"#{n} {s.label[:16]:16s} score={score_str}",
                    (10, y), FONT, 0.5, color, 1)
        y += 20

    # active drag rectangle preview
    if state["dragging"] and state["drag_start"] is not None and state["drag_current"] is not None:
        cv2.rectangle(frame, state["drag_start"], state["drag_current"], (0, 255, 255), 2)

    # last-sampled rectangle for the active slot, so you can see what was measured
    if active is not None and state["slots"][active] is not None:
        x1, y1, x2, y2 = state["slots"][active].rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

    return frame


def _sync_mask_window(state, frame, hsv):
    """Show or hide the mask-preview window for the currently-armed slot,
    reflecting the 'm' toggle. Kept as its OWN window rather than an
    in-place swap of the main feed (see module docstring) so the main
    window is always available to drag a new sample rectangle on.
    """
    active = state["active_slot"]
    sample = state["slots"].get(active) if active is not None else None

    if state["show_mask"] and sample is not None:
        detector = ColorMarkerDetector(sample.lower, sample.upper, min_area=state["min_area"])
        _, mask = detector.detect(frame, hsv=hsv)
        cv2.imshow(MASK_WINDOW_NAME, mask)
    else:
        try:
            if cv2.getWindowProperty(MASK_WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(MASK_WINDOW_NAME)
        except cv2.error:
            pass


def _make_mouse_callback(state):
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["drag_start"] = (x, y)
            state["drag_current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["drag_current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            x1, y1 = state["drag_start"]
            x2, y2 = x, y
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            if state["active_slot"] is None:
                print("No active slot armed -- press a number key 1-9 first.")
                return
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                print("Drag a larger rectangle (at least 5x5 px) over the sample.")
                return
            if state["hsv"] is None:
                return

            mean_hsv, std_hsv, n_px = _sample_rectangle(state["hsv"], (x1, y1, x2, y2))
            n = state["active_slot"]
            existing = state["slots"][n]
            label = (state["labels"].get(n)
                     or (existing.label if existing is not None else None)
                     or f"candidate {n}")
            if existing is not None:
                print(f"Slot {n} re-sampled (previous sample discarded).")

            sample = Sample(label, mean_hsv, std_hsv, n_px, (x1, y1, x2, y2))
            sample.rederive(state["k"])
            state["slots"][n] = sample

            print(f"Slot {n} \"{label}\" sampled: mean HSV=({mean_hsv[0]:.1f}, "
                  f"{mean_hsv[1]:.1f}, {mean_hsv[2]:.1f})  std=({std_hsv[0]:.1f}, "
                  f"{std_hsv[1]:.1f}, {std_hsv[2]:.1f})  n={n_px}px")

            _evaluate_slot(sample, state["frame"], state["hsv"],
                            state["background_frame"], state["background_hsv"],
                            state["min_area"])
            print(f"  -> derived MARKER_LOWER={sample.lower}  MARKER_UPPER={sample.upper}  "
                  f"score={sample.final_score}")

    return on_mouse


def main():
    parser = argparse.ArgumentParser(
        description="Empirically pick a corner-marker HSV range by sampling "
                    "physical color candidates from a live camera.")
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index to test candidates against (default: 0)")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA,
                        help="min blob area, must match production's "
                             "ColorMarkerDetector min_area to test the real thing "
                             f"(default: {DEFAULT_MIN_AREA}, ColorMarkerDetector's own default)")
    parser.add_argument("--k", type=float, default=DEFAULT_K,
                        help=f"initial range-width multiplier, mean +/- k*stddev "
                             f"(default: {DEFAULT_K})")
    args = parser.parse_args()

    state: dict = {
        "slots": {n: None for n in range(1, 10)},
        # Labels live separately from Sample objects so a slot can be
        # named ("hot pink tape") BEFORE it's ever sampled -- useful
        # while you're still getting the physical sample into position.
        "labels": {n: None for n in range(1, 10)},
        "active_slot": None,
        "background_hsv": None,
        "background_frame": None,
        "frame": None,
        "hsv": None,
        "dragging": False,
        "drag_start": None,
        "drag_current": None,
        "k": args.k,
        "min_area": args.min_area,
        "show_mask": False,
    }

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _make_mouse_callback(state))

    print("Color candidate tester running.")
    print("Press 1-9 to arm a slot, then drag a rectangle over a physical color "
          "sample in the live feed. Press 'p' any time for the ranking, 'q' to quit.\n")

    with Camera(args.camera) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            # Same blur+HSV pass ColorMarkerDetector.detect() would do
            # internally -- computed once here and handed to both the
            # rectangle-sampler and every detect() call this tick so it
            # isn't paid redundantly, matching the pattern debug_view.py
            # uses for its two detectors.
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            state["frame"] = frame
            state["hsv"] = hsv

            display = frame.copy()
            display = _draw_hud(display, state)
            cv2.imshow(WINDOW_NAME, display)

            _sync_mask_window(state, frame, hsv)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                _print_report(state["slots"])
                break

            elif key in [ord(str(d)) for d in range(1, 10)]:
                n = int(chr(key))
                state["active_slot"] = n
                existing = state["slots"][n]
                label = existing.label if existing is not None else "(empty)"
                print(f"Slot {n} armed -- {label}")

            elif key == ord('b'):
                replaced = state["background_hsv"] is not None
                state["background_hsv"] = hsv.copy()
                state["background_frame"] = frame.copy()
                print("Background reference " + ("replaced." if replaced else "captured."))

            elif key == ord('e'):
                if state["background_hsv"] is None:
                    print("Re-evaluating all sampled slots against the current live "
                          "frame (no background reference captured -- background "
                          "score will be n/a; press 'b' first to include it).")
                else:
                    print("Re-evaluating all sampled slots against the current live "
                          "frame and background reference.")
                for n, s in state["slots"].items():
                    if s is not None:
                        _evaluate_slot(s, frame, hsv, state["background_frame"],
                                        state["background_hsv"], state["min_area"])
                _print_report(state["slots"])

            elif key == ord('l'):
                n = state["active_slot"]
                if n is None:
                    print("Arm a slot (1-9) first.")
                else:
                    new_label = input(f"New label for slot {n}: ").strip()
                    if new_label:
                        state["labels"][n] = new_label
                        if state["slots"][n] is not None:
                            state["slots"][n].label = new_label
                        print(f"Slot {n} labeled \"{new_label}\".")

            elif key == ord('c'):
                n = state["active_slot"]
                if n is None:
                    print("Arm a slot (1-9) first.")
                else:
                    state["slots"][n] = None
                    print(f"Slot {n} cleared.")

            elif key == ord('r'):
                confirm = input("Type 'yes' to reset ALL slots and the background "
                                "reference: ").strip().lower()
                if confirm == "yes":
                    state["slots"] = {n: None for n in range(1, 10)}
                    state["labels"] = {n: None for n in range(1, 10)}
                    state["background_hsv"] = None
                    state["background_frame"] = None
                    print("Everything reset.")
                else:
                    print("Reset cancelled.")

            elif key == ord('['):
                state["k"] = max(K_MIN, round(state["k"] - K_STEP, 2))
                for s in state["slots"].values():
                    if s is not None:
                        s.rederive(state["k"])
                print(f"k = {state['k']:.2f} -- ranges re-derived, press 'e' to re-evaluate.")

            elif key == ord(']'):
                state["k"] = min(K_MAX, round(state["k"] + K_STEP, 2))
                for s in state["slots"].values():
                    if s is not None:
                        s.rederive(state["k"])
                print(f"k = {state['k']:.2f} -- ranges re-derived, press 'e' to re-evaluate.")

            elif key == ord('m'):
                state["show_mask"] = not state["show_mask"]

            elif key == ord('p'):
                _print_report(state["slots"])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

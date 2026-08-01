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
It then tests that exact derived range against the live scene using
ColorMarkerDetector -- the SAME class debug_view.py, coordinate_mapper.py
and run.py use in production -- so a good score here is guaranteed to mean
"will behave identically once pasted into those files."

IMPORTANT -- read this before trusting a score: the score printed
IMMEDIATELY after you drag a sample is a sanity check only. It is near-
guaranteed to look clean, because the range was just built from the mean
and stddev of those exact pixels -- testing it against that same instant
is close to circular. The TRUSTWORTHY segmentation score appears
automatically about one camera frame later (the very next tick), and again
every time you press 'e' after repositioning the candidate or holding it
steady -- that score is measuring something real: whether the derived
range still finds the candidate cleanly in a frame it was NOT derived
from. Treat the first number as "did sampling work at all", and the
number that appears a moment later as the one that actually matters.

THE "ARMED SLOT" CONCEPT: at any moment, exactly one numbered slot (1-9)
is "armed" -- number keys select it. Armed serves two roles at once: (1)
it is the destination for your NEXT drag-sample, and (2) it is treated as
"whatever physical candidate is currently in front of the camera", which
matters for 'e' (see below) and for the live Mask Preview window ('m') --
both of those only make sense for a candidate that's actually still in
view, not for some other slot you sampled five minutes ago and set down.

BACKGROUND REFERENCE / WORST-CASE DESIGN: press 'b' with the camera
pointed at the empty tank (nothing held up) to capture one background
reference frame -- press it again, under a different real lighting
moment or tank state, to add another; backgrounds accumulate, they are
never replaced. Every sampled slot's background score is then computed
against the SINGLE WORST reference frame captured so far (highest false-
positive area), not an average -- this is deliberate: it means capturing
more backgrounds can only ever tighten a candidate's apparent safety
margin, never inflate it by diluting one bad frame with several easy
ones. A candidate is only as safe as its worst tested condition.

'e' re-scores background for EVERY sampled slot (background scoring has
no dependency on what's currently in front of the camera, so this is
always safe/cheap to do for all of them at once) but schedules a live
segmentation re-test for the ARMED slot ONLY -- re-testing some other
slot's derived range against a frame that candidate isn't even in would
be meaningless.

SWITCHING CAMERAS: this project has two USB cameras. To validate a
candidate on the OTHER one, re-run this whole script with a different
`--camera <index>` rather than trying to switch cameras mid-session --
there is no in-session camera-switch key. Both the background references
and every slot's sampled HSV mean/stddev are specific to one camera's
optics and color reproduction; switching live without forcing a full
re-sample and re-background of everything would let stale numbers from
the old camera silently masquerade as still valid for the new one. Run
once per camera and compare the two printed reports.

This is a standalone diagnostic tool, run manually, NOT part of the
production loop (run.py). Like debug_view.py, it needs an actual display
(local monitor, or VNC/X11-forwarded SSH) -- a plain headless SSH session
won't show the windows it opens.

Interaction, start to finish
-----------------------------
1. Press a number key 1-9 to arm a slot. The HUD shows which slot is
   armed and its label (if any). Re-arming an already-sampled slot does
   NOT clear it -- arming just marks "this is what's in front of the
   camera right now" for the next drag and the next 'e' press.
2. Hold the physical candidate up to the camera. Left-click-drag a
   rectangle over its color in the live feed. The feed FREEZES for the
   duration of the drag (so the rectangle you see is always the exact
   pixels being measured, never ones that drifted mid-drag) and shows a
   "FROZEN" banner. Release the mouse to sample: this measures the mean
   + stddev HSV of every pixel in the rectangle and derives a range from
   it. A sanity-check score appears right away; the real score lands
   automatically about a tick later (see above).
3. Repeat step 2 for as many candidates as you want to compare, arming a
   different slot each time.
4. (Strongly recommended) Point the camera at the empty tank and press
   'b' -- ideally more than once, under different real conditions -- to
   capture background reference frame(s). Every slot's score then also
   reflects how much of the REAL background it would falsely light up.
5. Press 'e' any time (with a candidate armed) to re-test that candidate
   live -- useful after moving it to a different spot in the tank, or
   after adjusting the range width with '[' / ']'.
6. Press 'p' (or 'q' to quit) to print the full ranking: every candidate,
   sorted by score, with its derived (H, S, V) lower/upper tuple in the
   exact copy-pasteable form MARKER_LOWER/MARKER_UPPER use in
   debug_view.py, coordinate_mapper.py, and ColorMarkerDetector's
   defaults (detection.py).

Keybindings
-----------
    1-9   arm that slot number (destination for the next drag-sample,
          and "what's currently in front of the camera" for 'e' / 'm')
    drag  (left mouse button, on the video portion of the window) sample
          the armed slot's color from the rectangle you draw -- freezes
          the feed for the duration
    b     capture (append) one background-reference frame
    n     clear ALL background references (every slot loses its
          background score until 'b' is pressed again)
    e     re-score background for every sampled slot against the current
          background list, AND schedule a live segmentation re-test for
          the ARMED slot only (lands automatically on the next frame);
          then prints the full report
    l     label the armed slot (typed at the console)
    c     clear the armed slot's sample (the slot number stays reserved)
    [ ]   decrease / increase the range-width multiplier k; re-derives
          (and invalidates the stale scores of) every sampled slot
    m     toggle a SEPARATE "Mask Preview" window showing the armed
          slot's derived-range mask against the current live frame --
          kept as its own window (not an in-place swap, unlike
          debug_view.py's mask toggle) because the main window must
          always stay drag-able for sampling
    p     print the full ranked report to the console on demand
    q     print the final ranked report, then quit

Run from inside the Vision/ folder:
    python color_test.py                  # camera 0
    python color_test.py --camera 2       # a different USB camera index
    python color_test.py --min-area 200 --k 3.0
"""

import argparse
import math

import cv2
import numpy as np

from capture import Camera
from detection import ColorMarkerDetector

# ----------------------------------------------------------------------
# Tunables. Keep --min-area in sync with production's ColorMarkerDetector
# min_area -- a range that scores well here has to be tested against the
# same min_area production will actually run it with, or the segmentation/
# background scores are testing the wrong thing.
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
# Sized off the ranges already proven to work elsewhere in this project
# (MARKER_LOWER/UPPER in debug_view.py spans a 15-wide hue band and
# near-full-width sat/val bands) -- a derived range is never allowed to be
# narrower than real capture noise in this project has already shown it
# needs to be, which also guards against a very evenly-lit sample patch
# producing a near-zero stddev and a razor-thin, unreliable range.
H_MIN_HALF = 4
S_MIN_HALF = 20
V_MIN_HALF = 20

# Sanity-check slack (pixels) for "is the biggest detected blob actually
# where I dragged the sample rectangle, or did the derived range just
# find something else in the scene and call IT the biggest blob instead."
POSITION_SLACK_PX = 20

# Main-blob oversize penalty: how many times bigger than the SAMPLED
# rectangle the biggest live blob is allowed to be before segmentation
# score starts getting discounted. The sampled rectangle is the best proxy
# available for "how big the real candidate patch actually is", so a main
# blob many times that size -- even though it's a single contiguous region
# positioned right where the sample was taken -- is a sign the derived
# range has fused the real candidate together with adjacent background
# clutter into one blob, exactly the "widened range matches background
# clutter" failure this tool exists to catch, and which extra_count/
# position checks alone cannot see (a fused blob has no "extra" blob and
# is still centered on the sample point). OK_MULT allows generous headroom
# for the candidate simply being held closer/at an angle/bigger than the
# drag rectangle without penalty; beyond BAD_MULT the blob is treated as
# fully fused and floored at SIZE_PENALTY_FLOOR (not zero -- a big-but-real
# single blob is still worth something, just heavily discounted rather
# than treated as an outright fail like num_live_blobs==0 is).
SIZE_OK_MULT = 4.0
SIZE_BAD_MULT = 15.0
SIZE_PENALTY_FLOOR = 0.35

# Minimum sample-rectangle size in pixels (400px floor). Big enough that a
# lucky near-zero stddev can't come from a handful of pixels (the "tiny/
# lucky sample" failure mode this tool exists to avoid), small enough to
# still sample a small sticker at a typical holding distance.
MIN_RECT_W, MIN_RECT_H = 20, 20

# A sampled patch of this many pixels or more is treated as fully
# statistically trustworthy. Below it, confidence_factor = sqrt(n/2500)
# scales the FINAL blended score down (not any individual sub-score) --
# a mean/stddev estimate's own noise shrinks as 1/sqrt(n), so a tiny
# sample's apparently-great numbers deserve less trust.
CONFIDENCE_FULL_PIXELS = 2500

# Background false-area is normalized in units of "BG_AREA_NORM_MULT *
# min_area" -- i.e. how many marker-sized false blobs' worth of area lit
# up -- the same unit EXPECTED_MARKERS-style reasoning elsewhere in this
# project (debug_view.py) already uses, rather than a raw-pixel-count
# threshold that would mean something different at every resolution.
BG_AREA_NORM_MULT = 5

THUMBNAIL_SIZE = (40, 40)
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
# only hue needs this treatment. Identical math to debug_view.py's
# _update_hsv_stats/_hsv_mean, just also returning a stddev.
# ----------------------------------------------------------------------
def _circular_hue_mean_std(hue_values):
    """mean and stddev of a set of OpenCV hues (0-180), correctly handling
    the wraparound at 0/180. Returns (mean_hue, circ_std), both in the
    same 0-180 hue units.

    Each hue is treated as an angle (doubled, since OpenCV's 0-180
    represents a full 0-360 degree wheel), averaged as a unit vector, then
    converted back into hue units. Resultant length R is 1.0 for
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


def _sample_rectangle(hsv_frame, rect):
    """Mean + stddev, in HSV, of every pixel inside rect=(x1,y1,x2,y2).
    Hue uses circular stats (see _circular_hue_mean_std); sat/val use
    plain stats since they don't wrap. Returns (mean_hsv, std_hsv, n_px).
    """
    x1, y1, x2, y2 = rect
    patch = hsv_frame[y1:y2, x1:x2]
    h_vals = patch[:, :, 0].reshape(-1)
    s_vals = patch[:, :, 1].reshape(-1).astype(np.float64)
    v_vals = patch[:, :, 2].reshape(-1).astype(np.float64)

    h_mean, h_std = _circular_hue_mean_std(h_vals)
    mean_hsv = (h_mean, float(np.mean(s_vals)), float(np.mean(v_vals)))
    std_hsv = (h_std, float(np.std(s_vals)), float(np.std(v_vals)))
    return mean_hsv, std_hsv, int(h_vals.size)


def _derive_range(mean_hsv, std_hsv, k):
    """mean +/- k*stddev per channel, floored to the *_MIN_HALF constants
    and clipped to valid OpenCV HSV bounds (H: 0-180, S/V: 0-255).

    Returns (lower, upper, hue_wraps) where lower/upper are int (H, S, V)
    tuples ready to paste into MARKER_LOWER/MARKER_UPPER, and hue_wraps is
    True if the true (unclipped) derived hue interval would need to cross
    the 0/180 boundary to represent this color. ColorMarkerDetector only
    supports ONE contiguous hue range (unlike RedBallDetector's two-range
    wraparound workaround for red), so a candidate that needs wraparound
    cannot be made to work with ColorMarkerDetector at all -- this is
    surfaced explicitly (see the report) rather than silently clipped into
    a range that looks fine here but is quietly broken.
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


def _blob_areas(mask, min_area):
    """Areas (largest first) of every contour in mask that meets
    min_area. This only recovers geometry from a mask ColorMarkerDetector
    .detect() already produced -- the blur, HSV conversion, inRange
    threshold and erode/dilate that actually classify pixels all happened
    inside detect(), unchanged. Zero reclassification, one redundant cheap
    contour pass over the SAME mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    areas = [a for a in areas if a >= min_area]
    areas.sort(reverse=True)
    return areas


# ----------------------------------------------------------------------
# Candidate slot
# ----------------------------------------------------------------------
class Candidate:
    """Everything known about one candidate color, sampled into one of the
    numbered slots. Score fields start as None ("not yet evaluated") and
    are filled in by the _evaluate_* functions below -- kept separate from
    the raw sampled stats so changing k or re-testing never has to
    re-sample.
    """

    def __init__(self, slot, label, mean_hsv, std_hsv, n_pixels, rect, thumbnail):
        self.slot = slot
        self.label = label
        self.mean_hsv = mean_hsv        # (h, s, v) floats, as actually captured
        self.std_hsv = std_hsv          # (h, s, v) floats
        self.n_pixels = n_pixels
        self.rect = rect                # (x1, y1, x2, y2) source rectangle
        self.thumbnail = thumbnail      # 40x40 BGR crop, for the scoreboard panel
        self.lower = self.upper = None
        self.hue_wraps = None

        # scores -- all None until computed; report prints "n/a"
        self.consistency_score = None
        self.live_tested = False
        self.num_live_blobs = None
        self.main_area = None
        self.extra_area = None
        self.extra_count = None
        self.position_ok = None
        self.oversized = None
        self.segmentation_score = None
        self.bg_tested_count = 0        # how many background frames existed at last background eval
        self.bg_worst_area = None
        self.bg_worst_blobs = None
        self.background_score = None
        self.confidence_factor = None
        self.final_score = None

    def rect_center_and_diag(self):
        x1, y1, x2, y2 = self.rect
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), math.hypot(x2 - x1, y2 - y1)

    def rederive(self, k):
        """Recompute lower/upper from the (unchanged) sampled stats and
        invalidate everything downstream of the range -- NOT consistency
        (consistency only depends on the sampled stats, never the range)."""
        self.lower, self.upper, self.hue_wraps = _derive_range(self.mean_hsv, self.std_hsv, k)
        self.live_tested = False
        self.num_live_blobs = self.main_area = self.extra_area = self.extra_count = None
        self.position_ok = self.oversized = self.segmentation_score = None
        self.bg_tested_count = 0
        self.bg_worst_area = self.bg_worst_blobs = self.background_score = None
        self.confidence_factor = self.final_score = None


# ----------------------------------------------------------------------
# Scoring
#
# Three 0-100 sub-scores, combined into one final score via a confidence
# multiplier applied to the blend (not baked into each sub-score, so
# consistency/segmentation/background stay independently readable in the
# report):
#
#   consistency  -- how tightly the sampled pixels clustered. Leading
#                   indicator: a color that varies a lot under CURRENT
#                   lighting is unlikely to hold up later even if it
#                   happens to segment cleanly in one snapshot.
#   segmentation -- does the derived range, run through the REAL
#                   ColorMarkerDetector pipeline, find exactly one clean,
#                   reasonably-sized blob roughly where the sample was
#                   taken, in the CURRENT live scene? The one true
#                   end-to-end test: not a proxy, the actual production
#                   code path, exercised live. Penalized three separate
#                   ways so a bad range can't hide behind just one of
#                   them: stray SEPARATE blobs elsewhere in frame (extra_
#                   count), a main blob that's positioned wrong (position_
#                   ok), and a main blob that's positioned right and
#                   perfectly contiguous but implausibly large for the
#                   sampled patch (the fused-with-clutter case -- see
#                   SIZE_OK_MULT/SIZE_BAD_MULT above).
#   background   -- how much of an EMPTY-TANK reference frame this range
#                   would incorrectly light up, worst-case across every
#                   reference frame captured. Complements the size penalty
#                   above with ground truth from the real, sample-free
#                   background rather than an inferred proxy -- still the
#                   single most valuable signal for the "clutter near the
#                   tank bottom" complaint that motivated this tool, but
#                   optional (requires an extra capture step), so never
#                   mandatory.
# ----------------------------------------------------------------------
def _evaluate_consistency(cand):
    """Consistency depends only on the sampled stddev -- never the live
    scene -- so it's valid the instant a candidate is sampled and never
    needs re-computing when k changes or a new frame arrives.

    Hue is weighted ~2.8x heavier than sat/val: hue is what actually
    distinguishes this color from every OTHER color in the scene, so
    variance there is the dangerous kind. Sat/val naturally swing from
    glare, shadow and distance-to-light without indicating an unreliable
    COLOR -- this project's own MARKER_LOWER/UPPER already carries wide
    60-255 / 40-255 sat/val floors for exactly that reason.
    """
    h, s, v = cand.std_hsv
    consistency = 100.0 - (2.0 * h + 0.7 * s + 0.7 * v)
    cand.consistency_score = round(max(0.0, min(100.0, consistency)), 1)


def _evaluate_live_segmentation(cand, frame, hsv, min_area):
    """Run the REAL ColorMarkerDetector on the given frame/hsv and score
    how cleanly the derived range segments it. Deliberately called on a
    frame that is NOT the sampling frame whenever possible (see the
    deferred-eval scheduling around the mouse callback and the main loop)
    -- a range built from mean+/-k*stddev of a patch's own pixels will
    trivially re-detect that exact patch in that exact instant, so scoring
    it against the sampling frame itself would be close to circular.

    A single, well-positioned, contiguous main blob is NOT automatically a
    clean result: a range wide enough to fuse the real candidate together
    with adjacent background clutter produces exactly that shape (one
    blob, zero extras, centered on the sample point) while still being a
    genuinely bad range. area_fraction/extra_count/position_ok can't see
    this, since none of them look at absolute size -- see the SIZE_*_MULT
    penalty below, which compares the main blob's area against the
    sampled rectangle's area as a proxy for "how big the real candidate
    patch actually is".
    """
    detector = ColorMarkerDetector(cand.lower, cand.upper, min_area=min_area)
    centers, mask = detector.detect(frame, hsv=hsv)
    areas = _blob_areas(mask, min_area)
    cand.num_live_blobs = len(areas)
    cand.live_tested = True

    if not areas:
        # The range this candidate produced can't even find itself in the
        # frame it was just tested against -- an outright fail.
        cand.segmentation_score = 0.0
        cand.position_ok = False
        cand.main_area = cand.extra_area = cand.extra_count = None
        return

    main_area, extras = areas[0], areas[1:]
    extra_area, extra_count = sum(extras), len(extras)
    cand.main_area, cand.extra_area, cand.extra_count = main_area, extra_area, extra_count

    # Ratio-based, bounded form: "what fraction of everything this range
    # lit up is actually the one real blob", further discounted by how
    # many SEPARATE stray blobs there are -- scattered clutter is worse
    # than one contiguous stray patch of the same area, since each stray
    # blob is a candidate to wrongly rank into a top-N set downstream
    # (e.g. debug_view.py's EXPECTED_MARKERS cap).
    area_fraction = main_area / (main_area + extra_area) if (main_area + extra_area) > 0 else 0.0
    segmentation = 100.0 * area_fraction * (1.0 / (1.0 + 0.1 * extra_count))

    # Oversize check: a range fused with adjacent background clutter into
    # ONE contiguous blob produces area_fraction==1.0 and extra_count==0 --
    # indistinguishable, by the math above, from a genuinely clean result.
    # Compare the main blob's area against the sampled rectangle's area
    # (the best available proxy for "how big the real candidate patch
    # actually is") and discount smoothly once the blob is implausibly
    # bigger than that.
    (rx, ry), diag = cand.rect_center_and_diag()
    x1, y1, x2, y2 = cand.rect
    rect_area = max(1.0, (x2 - x1) * (y2 - y1))
    size_ratio = main_area / rect_area
    if size_ratio <= SIZE_OK_MULT:
        size_factor = 1.0
    elif size_ratio >= SIZE_BAD_MULT:
        size_factor = SIZE_PENALTY_FLOOR
    else:
        frac = (size_ratio - SIZE_OK_MULT) / (SIZE_BAD_MULT - SIZE_OK_MULT)
        size_factor = 1.0 - frac * (1.0 - SIZE_PENALTY_FLOOR)
    cand.oversized = size_ratio > SIZE_OK_MULT
    segmentation *= size_factor

    # Position sanity: is the BIGGEST blob actually near where the sample
    # was dragged? If the range's "main" blob is really some other object
    # in the scene, a clean-looking ratio score above would be actively
    # misleading -- cap it hard rather than trust the ratio math at face
    # value.
    dist = math.hypot(centers[0][0] - rx, centers[0][1] - ry)
    cand.position_ok = dist <= (diag / 2.0 + POSITION_SLACK_PX)
    if not cand.position_ok:
        segmentation = min(segmentation, 30.0)

    cand.segmentation_score = round(max(0.0, min(100.0, segmentation)), 1)


def _evaluate_background(cand, backgrounds, min_area):
    """Score how much of the WORST captured background frame this range
    would falsely light up. WORST-CASE, not average and not a single
    test: take the ONE background frame with the highest total false
    area, and score off THAT frame's (area, count) pair together (never
    mixing the max area from one frame with the max count from another).
    This rewards testing several real background conditions (different
    lighting moments / gravel visible or not / decorations in or out) and
    can only ever tighten, never inflate, a candidate's apparent safety
    margin as more backgrounds are added -- a candidate is only as safe
    as its worst tested condition.
    """
    detector = ColorMarkerDetector(cand.lower, cand.upper, min_area=min_area)
    per_bg = []  # (total_false_area, num_blobs) per background frame
    for bgr, hsv in backgrounds:
        _, mask = detector.detect(bgr, hsv=hsv)
        areas = _blob_areas(mask, min_area)
        per_bg.append((sum(areas), len(areas)))

    worst_area, worst_blobs = max(per_bg, key=lambda t: t[0])
    cand.bg_worst_area, cand.bg_worst_blobs = worst_area, worst_blobs
    cand.bg_tested_count = len(backgrounds)

    background = 100.0 / (1.0 + worst_area / (BG_AREA_NORM_MULT * min_area)) - 20.0 * worst_blobs
    cand.background_score = round(max(0.0, min(100.0, background)), 1)


def _maybe_finalize(cand):
    """Recompute final_score from whatever sub-scores currently exist.
    Consistency always exists once sampled. Segmentation/background may
    still be None ("not yet evaluated") -- weights renormalize over
    whatever IS available so an untested axis never silently inflates the
    score, and confidence_factor is applied to the blended result (not
    baked into each sub-score) so consistency/segmentation/background
    stay independently interpretable in the report.

    Nominal weights: segmentation is always weighted highest when present
    -- it is the one true end-to-end test, the real production code path,
    exercised live; background is the single most valuable signal for the
    specific "clutter near tank bottom" complaint that motivated this
    tool, but stays non-mandatory since capturing it is an extra step.
    """
    if cand.segmentation_score is None:
        # Only consistency exists yet (freshly sampled, deferred eval
        # hasn't landed this tick) -- report as provisional, not a real
        # ranking signal.
        cand.final_score = None
        return

    if cand.background_score is None:
        nominal = {"consistency": 0.4, "segmentation": 0.6}
    else:
        nominal = {"consistency": 0.3, "segmentation": 0.4, "background": 0.3}

    values = {"consistency": cand.consistency_score, "segmentation": cand.segmentation_score,
              "background": cand.background_score}
    total_w = sum(w for name, w in nominal.items() if values[name] is not None)
    raw = sum(w * values[name] for name, w in nominal.items() if values[name] is not None) / total_w

    cand.confidence_factor = round(min(1.0, math.sqrt(cand.n_pixels / CONFIDENCE_FULL_PIXELS)), 2)
    cand.final_score = round(raw * cand.confidence_factor, 1)


# ----------------------------------------------------------------------
# Console report
# ----------------------------------------------------------------------
def _fmt(v):
    return f"{v:.1f}" if v is not None else "n/a"


def _sorted_slots(state):
    filled = [(n, c) for n, c in sorted(state["slots"].items()) if c is not None]
    return sorted(filled, key=lambda item: (item[1].final_score is None, -(item[1].final_score or 0.0)))


def _print_report(state):
    print("\n" + "=" * 72)
    if not state["backgrounds"]:
        print("NO BACKGROUND REFERENCE CAPTURED -- these scores do not account for")
        print("false-positive risk against your real tank background. Press 'b'")
        print("(ideally several times, under different real conditions) before")
        print("trusting a ranking.")
        print("-" * 72)

    ranked = _sorted_slots(state)
    if not ranked:
        print("No candidates sampled yet -- arm a slot (1-9) and drag over a color.")
        print("=" * 72 + "\n")
        return

    print("CANDIDATE RANKING")
    print("=" * 72)
    for rank, (n, c) in enumerate(ranked, start=1):
        wrap_flag = " [HUE WRAPS 0/180]" if c.hue_wraps else ""
        conf_str = f"{c.confidence_factor * 100:.0f}%" if c.confidence_factor is not None else "n/a"
        print(f"#{rank}  slot {n} \"{c.label}\"{wrap_flag}")
        print(f"      final={_fmt(c.final_score)}   confidence={conf_str}")
        print(f"      consistency={_fmt(c.consistency_score)}  segmentation={_fmt(c.segmentation_score)}  "
              f"background={_fmt(c.background_score)}")
        print(f"      sampled HSV mean=({c.mean_hsv[0]:.1f}, {c.mean_hsv[1]:.1f}, {c.mean_hsv[2]:.1f})  "
              f"std=({c.std_hsv[0]:.1f}, {c.std_hsv[1]:.1f}, {c.std_hsv[2]:.1f})  n={c.n_pixels}px")
        if c.lower is not None:
            print(f"      MARKER_LOWER = {c.lower}")
            print(f"      MARKER_UPPER = {c.upper}")
        if c.hue_wraps:
            print("      WARNING: this candidate's hue straddles the OpenCV 0/180 wraparound "
                  "(like red does). ColorMarkerDetector has no dual-range workaround the way "
                  "RedBallDetector does, so this range cannot be made to work as a single "
                  "ColorMarkerDetector range -- prefer a different candidate.")
        if c.final_score is None:
            print("      (provisional -- press 'e' with this slot armed, or wait a tick after sampling)")
        if c.live_tested and c.num_live_blobs == 0:
            print("      WARNING: this range found NOTHING in the live frame it was tested against.")
        if c.live_tested and c.position_ok is False:
            print("      WARNING: the biggest blob found was NOT near where you sampled -- "
                  "this range is probably matching something else.")
        if c.live_tested and c.oversized:
            print("      WARNING: the main blob is much larger than the sampled rectangle -- "
                  "this range likely fuses the real candidate with background clutter into "
                  "one blob. Narrow the range ('[') or capture a background reference ('b') "
                  "to confirm.")
        if c.bg_tested_count > 0:
            print(f"      background: worst of {c.bg_tested_count} reference frame(s) -> "
                  f"false area={c.bg_worst_area:.0f}px  blobs={c.bg_worst_blobs}")
        else:
            print("      background: not tested (press 'b' to capture a reference frame)")
        print()

    best_n, best = ranked[0]
    print("-" * 72)
    if best.final_score is None:
        print("No fully-scored winner yet -- arm a slot and press 'e' (or wait a tick after sampling).")
    else:
        conf_str = f"{best.confidence_factor * 100:.0f}%" if best.confidence_factor is not None else "n/a"
        print(f"WINNER: slot {best_n} \"{best.label}\" (final score {best.final_score}, confidence {conf_str})")
        print("Paste directly into MARKER_LOWER / MARKER_UPPER "
              "(debug_view.py, coordinate_mapper.py, ColorMarkerDetector's defaults in detection.py):")
        print(f"    MARKER_LOWER = {best.lower}")
        print(f"    MARKER_UPPER = {best.upper}")

        caveats = []
        if best.segmentation_score is None:
            caveats.append("live segmentation not yet tested")
        if best.background_score is None:
            caveats.append("background false-positive risk not yet tested")
        if best.hue_wraps:
            caveats.append("hue wraps the 0/180 seam -- ColorMarkerDetector can't use it as-is")
        caveats.append(f"only validated against camera index {state['camera_index']} this session -- "
                        "re-run with --camera <other index> to validate a second camera")
        print("    CAVEAT: " + "; ".join(caveats) + ".")
    print("=" * 72 + "\n")


# ----------------------------------------------------------------------
# HUD / display
# ----------------------------------------------------------------------
def _draw_hud(frame, state):
    """Draws directly onto the (already-copied) video frame: status line,
    keybinding cheatsheet, the armed slot's last-sampled rectangle
    (magenta), and the in-progress drag rectangle (yellow) while frozen.
    """
    armed = state["armed_slot"]
    if armed is None:
        armed_str = "ARMED: none (press 1-9)"
    else:
        cand = state["slots"][armed]
        if cand is not None:
            label = cand.label
        else:
            label = state["labels"][armed] or "(empty -- drag to sample)"
        armed_str = f"ARMED: {armed} {label}"

    frozen_str = "  |  FROZEN -- sampling..." if state["is_frozen"] else ""
    status = f"{armed_str}  |  k={state['k']:.2f}  |  backgrounds: {len(state['backgrounds'])}{frozen_str}"
    cv2.putText(frame, status, (10, 22), FONT, 0.55, (255, 255, 255), 2)

    cheatsheet = ("1-9 arm | drag: sample | b: bg capture | n: bg clear | e: re-evaluate | "
                  "l: label | c: clear slot | [ ]: k | m: mask | p: report | q: quit")
    cv2.putText(frame, cheatsheet, (10, frame.shape[0] - 10), FONT, 0.42, (180, 180, 180), 1)

    if armed is not None and state["slots"][armed] is not None:
        x1, y1, x2, y2 = state["slots"][armed].rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    if state["is_frozen"] and state["drag_start"] is not None and state["drag_current"] is not None:
        cv2.rectangle(frame, state["drag_start"], state["drag_current"], (0, 255, 255), 2)

    return frame


def _live_tag(cand):
    if not cand.live_tested:
        return "LIVE:pending"
    if cand.num_live_blobs == 0:
        return "LIVE:none"
    if (cand.extra_count or 0) > 0 or cand.position_ok is False or cand.oversized:
        return "LIVE:clutter"
    return "LIVE:ok"


def _build_scoreboard_panel(state, width):
    """Scoreboard panel stacked BELOW the video feed: one row per
    occupied slot, sorted by final_score descending (unscored slots sort
    last, tagged provisional). Armed slot's row gets a green border.
    """
    ranked = _sorted_slots(state)
    header_h = 26
    row_h = 50

    if not ranked:
        panel = np.zeros((header_h + 30, width, 3), dtype=np.uint8)
        cv2.putText(panel, "No candidates sampled yet -- arm a slot (1-9) and drag over a color.",
                    (10, header_h + 18), FONT, 0.5, (180, 180, 180), 1)
        return panel

    height = header_h + row_h * len(ranked) + 6
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(panel, "SCOREBOARD (best first)", (10, 18), FONT, 0.5, (255, 255, 255), 1)

    armed = state["armed_slot"]
    y = header_h
    for n, c in ranked:
        row_top, row_bottom = y, y + row_h

        th, tw = c.thumbnail.shape[:2]
        tx0, ty0 = 6, row_top + 5
        if ty0 + th <= height and tx0 + tw <= width:
            panel[ty0:ty0 + th, tx0:tx0 + tw] = c.thumbnail

        text_x = tx0 + THUMBNAIL_SIZE[0] + 12
        score_str = f"{c.final_score:.1f}" if c.final_score is not None else "n/a (provisional)"
        conf_str = f"{c.confidence_factor * 100:.0f}%" if c.confidence_factor is not None else "--"
        bg_tag = f"BG:{c.bg_tested_count} tested" if c.bg_tested_count > 0 else "BG:none"

        line1 = f"#{n} {c.label[:16]:16s} score={score_str}"
        line2 = f"conf={conf_str}  {_live_tag(c)}  {bg_tag}"
        cv2.putText(panel, line1, (text_x, row_top + 20), FONT, 0.48, (255, 255, 255), 1)
        cv2.putText(panel, line2, (text_x, row_top + 40), FONT, 0.45, (180, 180, 180), 1)

        if n == armed:
            cv2.rectangle(panel, (2, row_top + 2), (width - 3, row_bottom - 3), (0, 255, 0), 2)

        y = row_bottom

    return panel


def _sync_mask_window(state):
    """Show or hide the SEPARATE Mask Preview window for the armed slot,
    reflecting the 'm' toggle -- kept separate (not an in-place swap of
    the main feed) so the main window always stays drag-able for
    sampling. Always shows the armed slot's derived-range mask against
    the current (unfrozen) live frame.
    """
    armed = state["armed_slot"]
    cand = state["slots"].get(armed) if armed is not None else None

    if state["show_mask"] and cand is not None and cand.lower is not None and state["frame"] is not None:
        detector = ColorMarkerDetector(cand.lower, cand.upper, min_area=state["min_area"])
        _, mask = detector.detect(state["frame"], hsv=state["hsv"])
        cv2.imshow(MASK_WINDOW_NAME, mask)
    else:
        try:
            if cv2.getWindowProperty(MASK_WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(MASK_WINDOW_NAME)
        except cv2.error:
            pass


# ----------------------------------------------------------------------
# Mouse callback -- freeze-on-drag sampling with deferred live evaluation
# ----------------------------------------------------------------------
def _make_mouse_callback(state):
    def _clip(x, y):
        vh, vw = state["video_shape"]
        return max(0, min(x, vw - 1)), max(0, min(y, vh - 1))

    def on_mouse(event, x, y, flags, param):
        if state["video_shape"] is None or state["frame"] is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if state["armed_slot"] is None:
                print("Arm a slot (1-9) first.")
                return
            vh, vw = state["video_shape"]
            if not (0 <= x < vw and 0 <= y < vh):
                return  # click landed in the scoreboard panel, not the video area
            state["is_frozen"] = True
            state["frozen_frame"] = state["frame"].copy()
            state["frozen_hsv"] = state["hsv"].copy()
            state["drag_start"] = (x, y)
            state["drag_current"] = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if state["is_frozen"]:
                state["drag_current"] = _clip(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if not state["is_frozen"]:
                return
            state["is_frozen"] = False
            x1, y1 = state["drag_start"]
            x2, y2 = _clip(x, y)
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            if (x2 - x1) < MIN_RECT_W or (y2 - y1) < MIN_RECT_H:
                print(f"Drag at least {MIN_RECT_W}x{MIN_RECT_H}px over the sample -- rejected.")
                return

            mean_hsv, std_hsv, n_px = _sample_rectangle(state["frozen_hsv"], (x1, y1, x2, y2))
            n = state["armed_slot"]
            existing = state["slots"][n]
            label = state["labels"][n] or (existing.label if existing is not None else f"candidate {n}")
            thumb = cv2.resize(state["frozen_frame"][y1:y2, x1:x2], THUMBNAIL_SIZE)

            if existing is not None:
                print(f"Slot {n} re-sampled (previous sample discarded).")

            cand = Candidate(n, label, mean_hsv, std_hsv, n_px, (x1, y1, x2, y2), thumb)
            cand.rederive(state["k"])
            state["slots"][n] = cand

            print(f"Slot {n} \"{label}\": mean HSV=({mean_hsv[0]:.1f},{mean_hsv[1]:.1f},{mean_hsv[2]:.1f}) "
                  f"std=({std_hsv[0]:.1f},{std_hsv[1]:.1f},{std_hsv[2]:.1f}) n={n_px}px  "
                  f"MARKER_LOWER={cand.lower} MARKER_UPPER={cand.upper}")
            if n_px < CONFIDENCE_FULL_PIXELS:
                print(f"  (note: sample is smaller than the {CONFIDENCE_FULL_PIXELS}px full-confidence size -- "
                      f"score will be scaled down by a confidence factor until you drag a bigger area)")

            # DO NOT evaluate live segmentation here -- evaluating against
            # frozen_hsv right now would test the range against the exact
            # pixels it was derived from (near-circular). Schedule a
            # deferred eval for the next non-frozen frame instead.
            state["pending_live_eval"] = n

            # Background score has no such circularity problem (it's
            # independent of what's in front of the camera right now), so
            # it's safe to compute immediately if any references exist.
            if state["backgrounds"]:
                _evaluate_background(cand, state["backgrounds"], state["min_area"])
            _evaluate_consistency(cand)
            _maybe_finalize(cand)

    return on_mouse


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Empirically pick a corner-marker HSV range by sampling "
                    "physical color candidates from a live camera.")
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index to test candidates against (default: 0)")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA,
                        help="min blob area, must match production's ColorMarkerDetector "
                             f"min_area to test the real thing (default: {DEFAULT_MIN_AREA}, "
                             "ColorMarkerDetector's own default)")
    parser.add_argument("--k", type=float, default=DEFAULT_K,
                        help=f"initial range-width multiplier, mean +/- k*stddev "
                             f"(default: {DEFAULT_K})")
    args = parser.parse_args()

    state = {
        "slots": {n: None for n in range(1, 10)},
        "labels": {n: None for n in range(1, 10)},
        "armed_slot": None,
        "backgrounds": [],
        "k": args.k,
        "min_area": args.min_area,
        "frame": None,
        "hsv": None,
        "video_shape": None,      # (h, w) of the video portion, set each tick
        "is_frozen": False,
        "frozen_frame": None,
        "frozen_hsv": None,
        "drag_start": None,
        "drag_current": None,
        "pending_live_eval": None,
        "show_mask": False,
        "camera_index": args.camera,
    }

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _make_mouse_callback(state))

    digit_keys = {ord(str(d)): d for d in range(1, 10)}

    print("Color candidate tester running.")
    print("Press 1-9 to arm a slot, then drag a rectangle over a physical color sample "
          "in the live feed. Press 'b' over the empty tank for a background reference. "
          "Press 'p' any time for the ranking, 'q' to quit.\n")

    with Camera(args.camera) as cam:
        while True:
            raw = cam.read()

            if raw is not None:
                if not state["is_frozen"]:
                    blurred = cv2.GaussianBlur(raw, (5, 5), 0)
                    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                    state["frame"] = raw
                    state["hsv"] = hsv
                    state["video_shape"] = raw.shape[:2]
                # while frozen: cam.read() above still happened (keeps the
                # camera's own internal buffer from stalling/backing up),
                # but state["frame"]/state["hsv"] are deliberately left
                # untouched -- the frozen_* copies are what's shown/sampled.

                if state["pending_live_eval"] is not None and not state["is_frozen"]:
                    n = state["pending_live_eval"]
                    cand = state["slots"].get(n)
                    if cand is not None and cand.lower is not None:
                        _evaluate_live_segmentation(cand, state["frame"], state["hsv"], state["min_area"])
                        _maybe_finalize(cand)
                    state["pending_live_eval"] = None

                base = state["frozen_frame"] if state["is_frozen"] else state["frame"]
                if base is not None:
                    display_top = _draw_hud(base.copy(), state)
                    panel = _build_scoreboard_panel(state, display_top.shape[1])
                    display = np.vstack([display_top, panel])
                    cv2.imshow(WINDOW_NAME, display)

                _sync_mask_window(state)

            key = cv2.waitKey(1) & 0xFF

            if key == 255:  # no key pressed this tick
                continue

            if key == ord('q'):
                _print_report(state)
                break

            elif key in digit_keys:
                n = digit_keys[key]
                state["armed_slot"] = n
                existing = state["slots"][n]
                label = existing.label if existing is not None else (state["labels"][n] or "(empty)")
                print(f"Slot {n} armed -- {label}")

            elif key == ord('b'):
                if state["is_frozen"]:
                    print("Finish the current drag before capturing a background reference.")
                elif state["frame"] is None:
                    print("No frame available yet -- try again in a moment.")
                else:
                    state["backgrounds"].append((state["frame"].copy(), state["hsv"].copy()))
                    print(f"Background reference captured (now {len(state['backgrounds'])} total).")
                    for cand in state["slots"].values():
                        if cand is not None and cand.lower is not None:
                            _evaluate_background(cand, state["backgrounds"], state["min_area"])
                            _maybe_finalize(cand)

            elif key == ord('n'):
                state["backgrounds"] = []
                for cand in state["slots"].values():
                    if cand is not None:
                        cand.bg_tested_count = 0
                        cand.bg_worst_area = None
                        cand.bg_worst_blobs = None
                        cand.background_score = None
                        _maybe_finalize(cand)
                print("All background references cleared -- background risk is now unscored for every slot.")

            elif key == ord('e'):
                if state["backgrounds"]:
                    for cand in state["slots"].values():
                        if cand is not None and cand.lower is not None:
                            _evaluate_background(cand, state["backgrounds"], state["min_area"])
                            _maybe_finalize(cand)
                armed = state["armed_slot"]
                if armed is not None and state["slots"][armed] is not None:
                    state["pending_live_eval"] = armed
                    print(f"Background re-scored for all slots; live segmentation re-eval for "
                          f"slot {armed} scheduled for the next frame.")
                else:
                    print("Background re-scored for all slots (no sampled candidate is armed, "
                          "so there's nothing to re-test live segmentation on).")
                _print_report(state)

            elif key == ord('l'):
                n = state["armed_slot"]
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
                n = state["armed_slot"]
                if n is None:
                    print("Arm a slot (1-9) first.")
                else:
                    state["slots"][n] = None
                    print(f"Slot {n} cleared (label kept; slot number stays reserved).")

            elif key == ord('['):
                state["k"] = max(K_MIN, round(state["k"] - K_STEP, 2))
                for cand in state["slots"].values():
                    if cand is not None:
                        cand.rederive(state["k"])
                print(f"k = {state['k']:.2f} -- ranges re-derived for every sampled slot; "
                      "arm a slot and press 'e' to re-test it live.")

            elif key == ord(']'):
                state["k"] = min(K_MAX, round(state["k"] + K_STEP, 2))
                for cand in state["slots"].values():
                    if cand is not None:
                        cand.rederive(state["k"])
                print(f"k = {state['k']:.2f} -- ranges re-derived for every sampled slot; "
                      "arm a slot and press 'e' to re-test it live.")

            elif key == ord('m'):
                state["show_mask"] = not state["show_mask"]

            elif key == ord('p'):
                _print_report(state)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

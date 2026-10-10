"""Eye-gaze estimation — pure math over MediaPipe FaceMesh landmarks.

MediaPipe FaceMesh with ``refine_landmarks=True`` yields 478 landmarks, the
last ten being the two irises.  This module turns those landmarks into a
screen cursor position in three independent, individually-testable steps:

1. **raw**   — normalize each iris centre inside its eye's bounding box
   (eye corners horizontally, eyelids vertically) and average both eyes,
   giving a raw gaze point ``(rx, ry)`` in roughly ``[0, 1]^2``.
2. **fit_calibration** — a per-axis linear least-squares fit over recorded
   ``(raw, screen)`` pairs, producing gains/offsets that map raw gaze to
   screen pixels (plus an honest RMS residual in pixels).
3. **to_screen** — apply that calibration and clamp to the screen with a
   small edge margin.

No head-pose compensation in v1: the mapping assumes the head stays roughly
where it was during calibration, so large head translations shift the gaze
point (documented limitation — recalibrate after repositioning).  A partial
blink also compresses the vertical eye box; readings mid-blink are rejected
outright (see ``raw``) rather than emitted wrongly.

Everything here is pure stdlib (``math``) over objects exposing ``.x``/``.y``
— no cv2, no mediapipe, no Qt — so the whole pipeline is exercised by fast,
hardware-free tests.  Every function is total: bad input yields ``None`` (or
a safe default), never an exception.
"""

import math

# ── FaceMesh landmark indices (refine_landmarks=True, 478 points) ──────
LEFT_IRIS = (468, 469, 470, 471, 472)     # 468 = centre
RIGHT_IRIS = (473, 474, 475, 476, 477)    # 473 = centre
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
LEFT_LID_UPPER, LEFT_LID_LOWER = 159, 145
RIGHT_LID_UPPER, RIGHT_LID_LOWER = 386, 374

MIN_LANDMARKS = 478

# An eye box whose corners are closer than this (normalized units) is
# degenerate — no meaningful iris position can be read from it.
_DEGENERATE_CORNER_DIST = 1e-4

# Lid gap below this fraction of the corner distance = eye (nearly) closed:
# mid-blink the iris landmarks are unreliable, so refuse to read them.
_CLOSED_LID_RATIO = 0.2

# Gains beyond this map millimetres of raw movement to whole screens —
# the calibration samples had no real gaze spread; reject the fit.
_MAX_ABS_GAIN = 20000.0

# to_screen keeps the cursor at least this many pixels from every edge.
EDGE_MARGIN_PX = 5


def _eye_raw(lms, corner_a, corner_b, lid_upper, lid_lower, iris_idx):
    """Normalized (0..1, 0..1) iris position inside one eye's box, or None."""
    ca, cb = lms[corner_a], lms[corner_b]
    corner_dist = math.hypot(cb.x - ca.x, cb.y - ca.y)
    if corner_dist < _DEGENERATE_CORNER_DIST:
        return None
    lu, ll = lms[lid_upper], lms[lid_lower]
    lid_gap = math.hypot(ll.x - lu.x, ll.y - lu.y)
    if lid_gap < _CLOSED_LID_RATIO * corner_dist:
        return None                              # blink / closed eye

    # Iris centre = mean of the 5 iris landmarks (steadier than the single
    # centre point, which jitters more frame-to-frame).
    ix = sum(lms[i].x for i in iris_idx) / len(iris_idx)
    iy = sum(lms[i].y for i in iris_idx) / len(iris_idx)

    x0, x1 = min(ca.x, cb.x), max(ca.x, cb.x)    # corners: horizontal span
    y0, y1 = min(lu.y, ll.y), max(lu.y, ll.y)    # lids: vertical span
    if (x1 - x0) < _DEGENERATE_CORNER_DIST or (y1 - y0) < _DEGENERATE_CORNER_DIST:
        return None
    return (ix - x0) / (x1 - x0), (iy - y0) / (y1 - y0)


class GazeEstimator:
    """Iris-in-eye-box gaze: normalizes each iris centre inside its eye's
    bounding box (corners horizontally, lids vertically) and averages both
    eyes -> a raw (rx, ry) in ~[0,1]^2 that a linear calibration maps to
    screen pixels.  No head-pose compensation in v1 (documented limitation).
    """

    def __init__(self):
        self._ema = None    # EMA state for smoothed()

    # ── raw gaze ───────────────────────────────────────────────────────

    def raw(self, landmarks):
        """Raw gaze point ``(rx, ry)`` in ~[0,1]^2, or ``None`` when the
        frame carries no reliable reading: fewer than 478 landmarks, a
        degenerate eye box, or eyes closed (blink)."""
        try:
            if landmarks is None or len(landmarks) < MIN_LANDMARKS:
                return None
            left = _eye_raw(landmarks, LEFT_EYE_OUTER, LEFT_EYE_INNER,
                            LEFT_LID_UPPER, LEFT_LID_LOWER, LEFT_IRIS)
            right = _eye_raw(landmarks, RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
                             RIGHT_LID_UPPER, RIGHT_LID_LOWER, RIGHT_IRIS)
            if left is None or right is None:
                return None
            return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
        except Exception:
            return None     # malformed landmark objects → no reading

    # ── calibration → screen ───────────────────────────────────────────

    def to_screen(self, raw_xy, cal):
        """Map a raw gaze point through a calibration dict (from
        ``fit_calibration``) to integer screen pixels, clamped at least
        ``EDGE_MARGIN_PX`` from every edge.  Bad input → screen centre."""
        try:
            w = int(cal.get("screen_w", 1920))
            h = int(cal.get("screen_h", 1080))
        except Exception:
            w, h = 1920, 1080
        try:
            rx, ry = float(raw_xy[0]), float(raw_xy[1])
            sx = float(cal["x_gain"]) * rx + float(cal["x_off"])
            sy = float(cal["y_gain"]) * ry + float(cal["y_off"])
            if not (math.isfinite(sx) and math.isfinite(sy)):
                raise ValueError
        except Exception:
            sx, sy = w / 2.0, h / 2.0            # safe default: centre
        m = EDGE_MARGIN_PX
        sx = min(max(sx, m), max(w - m, m))
        sy = min(max(sy, m), max(h - m, m))
        return int(round(sx)), int(round(sy))

    # ── smoothing (simple EMA; integrator calls per frame) ─────────────

    def smoothed(self, raw_xy, alpha):
        """Exponential-moving-average of raw gaze points.  ``alpha`` in
        (0..1]: higher = follow new samples faster.  ``None`` input returns
        the current state unchanged (``None`` until the first sample)."""
        try:
            if raw_xy is None:
                return self._ema
            x, y = float(raw_xy[0]), float(raw_xy[1])
            a = min(max(float(alpha), 0.0), 1.0)
            if self._ema is None:
                self._ema = (x, y)
            else:
                px, py = self._ema
                self._ema = (a * x + (1.0 - a) * px,
                             a * y + (1.0 - a) * py)
            return self._ema
        except Exception:
            return self._ema

    def reset(self):
        """Forget the EMA state (call when the face is lost / re-acquired)."""
        self._ema = None


def _fit_axis(raws, screens):
    """Least-squares line s = gain*r + off; None when degenerate."""
    n = len(raws)
    rm = sum(raws) / n
    sm = sum(screens) / n
    var = sum((r - rm) ** 2 for r in raws) / n
    if var < 1e-12:
        return None                              # gaze never moved → no fit
    cov = sum((r - rm) * (s - sm) for r, s in zip(raws, screens)) / n
    gain = cov / var
    if not math.isfinite(gain) or abs(gain) > _MAX_ABS_GAIN:
        return None                              # exploding gain → degenerate
    return gain, sm - gain * rm


def fit_calibration(samples, screen_w, screen_h):
    """Least-squares per-axis linear fit over >=4 ``(raw, screen)`` samples.

    Returns ``{"x_gain","x_off","y_gain","y_off","screen_w","screen_h",
    "residual_px","n"}`` — ``residual_px`` is the RMS pixel error over the
    samples — or ``None`` when the fit is degenerate: fewer than 4 samples,
    raw x/y variance ~0 (the user's gaze didn't move, gains would explode),
    or ``|gain| > 20000``.
    """
    try:
        pts = [((float(r[0]), float(r[1])), (float(s[0]), float(s[1])))
               for r, s in samples]
        w, h = int(screen_w), int(screen_h)
    except Exception:
        return None
    if len(pts) < 4:
        return None
    fx = _fit_axis([r[0] for r, _ in pts], [s[0] for _, s in pts])
    fy = _fit_axis([r[1] for r, _ in pts], [s[1] for _, s in pts])
    if fx is None or fy is None:
        return None
    x_gain, x_off = fx
    y_gain, y_off = fy
    err2 = sum((x_gain * r[0] + x_off - s[0]) ** 2 +
               (y_gain * r[1] + y_off - s[1]) ** 2
               for r, s in pts) / len(pts)
    return {
        "x_gain": x_gain, "x_off": x_off,
        "y_gain": y_gain, "y_off": y_off,
        "screen_w": w, "screen_h": h,
        "residual_px": math.sqrt(err2),
        "n": len(pts),
    }


def quality_note(cal):
    """Honest one-line accuracy note for a calibration dict."""
    try:
        r = float(cal["residual_px"])
        if not math.isfinite(r) or r < 0:
            raise ValueError
    except Exception:
        return "uncalibrated"
    px = int(round(r))
    if r < 80:
        return f"good (~{px}px)"
    if r < 200:
        return f"usable for big targets (~{px}px)"
    return f"low accuracy (~{px}px) — best with dwell + large targets"

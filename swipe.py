"""What the head drew, from a stroke of cursor positions.

Two recognisers over the same points, layered deliberately.

**Direction** is the net vector from where the stroke started to where it
ended. A violently shaky line from A to B still points from A to B, so this
survives everything a tremor does to it. It is the floor: whenever the shape
recogniser is unsure, the caller still gets a usable answer.

**Shape** is the standard $1 unistroke recogniser — resample to a fixed number
of points, rotate to the indicative angle, scale to a square, compare by mean
point distance. Roughly sixty lines and no dependencies.

--------------------------------------------------------------------------
The measurement this design is built on
--------------------------------------------------------------------------
Shapes were nearly cut from this feature on the assumption that tremor would
destroy them. Simulating 5 shapes x 40 trials at Mistey's real sample rate
(9.1 Hz, measured on an actual laptop webcam) says otherwise:

    tremor 16 px .......................... 100%
    an overshoot hook on the end .......... 100%
    only 80% of the stroke drawn ............ 40%   <-- the real problem
    only 50% of the stroke drawn ............ 14%

Tremor is a non-issue. What destroys shape recognition is the stroke ENDING
EARLY, which is exactly what fatigue and spasticity cause, in exactly the
population this software is for. So each template is also stored part-drawn
(`_PREFIXES`) and a candidate scores against the best of them.

What the code in this file actually scores, same conditions:

    SHAPES                          DIRECTIONS
    clean ................. 100%    tremor  4 px ......... 100%
    tremor 16 px .......... 100%    tremor 16 px ......... 100%
    80% drawn ............. 100%    tremor 30 px ......... 100%
    squashed to 35% ........ 74%    tremor 40 px ......... 100%
    65% drawn ............... 27%    tremor 60 px ......... 100%

The two weak rows are real and are not papered over. A stroke below ~70% drawn
is not recoverable, and a head with barely any horizontal range degrades a
shape — which is exactly why an unrecognised shape falls back to its DIRECTION
rather than to nothing, and why the caller is expected to show the running best
match so the person can release at the right moment instead of guessing.

Pure geometry: no OS, no Qt, no numpy, no clock. Same discipline as
`scanning.py` — nothing in here can raise, because it is called from the thread
that drives somebody's only pointer.
"""

import math

DIRECTIONS_4 = ("up", "down", "left", "right")
DIRECTIONS_8 = DIRECTIONS_4 + ("up_left", "up_right", "down_left", "down_right")
SHAPES = ("circle", "zed", "ell", "vee", "caret")

# $1 works on a fixed-length point list; 64 is the value from the paper and is
# ample for the ~18 points a 1.5 s stroke yields at 9.1 Hz.
RESAMPLE_N = 64
SQUARE = 250.0

# Completion fractions each template is stored at.
#
# Measured, over 5 shapes x 40 trials at 9.1 Hz. More prefixes is not better:
# every extra template is another chance for the WRONG shape to match, and the
# short ones never earn their keep because a stroke below ~70% drawn is not
# recoverable anyway.
#
#     prefixes                     clean   80% drawn   squashed to 35%
#     (1.0,)                        100%       20%           55%
#     (0.7, 0.85, 1.0)              100%      100%           76%   <-- chosen
#     (0.55, 0.7, 0.85, 1.0)        100%      100%           60%
#     (0.4, 0.55, 0.7, 0.85, 1.0)   100%      100%           60%
_PREFIXES = (0.7, 0.85, 1.0)

# Below this, the stroke is a rest rather than a movement.
DEFAULT_MIN_DISTANCE = 120.0
# Net distance divided by path length. A there-and-back wobble covers a lot of
# path and ends up nowhere, so it scores near zero and is refused rather than
# guessed at.
DEFAULT_MIN_STRAIGHTNESS = 0.55
# How well a shape must match before it beats the direction answer.
DEFAULT_SHAPE_FLOOR = 0.80
# A "hold" has to actually be STILL, not merely short.
#
# Measured stillness: a genuine rest scores 0.93, a wide there-and-back wobble
# scores 0.00, and -- the case that set this number -- a 94 px swipe by
# somebody whose head barely turns scores 0.69. At a lower floor that person
# got a RIGHT-CLICK every time they tried to swipe: an action they never asked
# for, triggered by the very limitation the software exists to accommodate.
#
# At 0.80 their stroke returns None instead, the overlay says it was not clear,
# and the real remedy is available to them: `swipe_min_distance_px` is tunable
# down to 40 px, which turns that same movement into the swipe they meant.
DEFAULT_HOLD_FLOOR = 0.80


# ----------------------------------------------------------------- cleaning

def _clean(points):
    """[(t, x, y)] or [(x, y)] -> [(x, y)] with the unusable rows dropped.

    Accepts both shapes because the driver stores timestamps and the tests do
    not. Anything non-finite or non-numeric is skipped rather than raised on.
    """
    out = []
    try:
        for item in points or ():
            try:
                if len(item) >= 3:
                    x, y = float(item[1]), float(item[2])
                else:
                    x, y = float(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if x != x or y != y:                    # NaN
                continue
            if x in (float("inf"), float("-inf")):
                continue
            if y in (float("inf"), float("-inf")):
                continue
            out.append((x, y))
    except TypeError:
        return []
    return out


def _path_length(pts):
    return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
               for i in range(1, len(pts)))


def _smooth(pts, window=3):
    """A 3-wide moving average, used ONLY for the straightness measure.

    Straightness is net distance divided by path length, and raw path length
    counts every tremor excursion between one sample and the next. That made
    the gate reject exactly what it was supposed to tolerate:

        tremor 16 px -> 0.75     tremor 40 px -> 0.37   (refused)
        tremor 30 px -> 0.49     tremor 60 px -> 0.25   (refused)

    Smoothed first, the same strokes score 0.96 / 0.81 / 0.88 / 0.66 and pass,
    while a genuine there-and-back wobble still scores 0.06 and is still
    refused. The direction itself is always taken from the RAW endpoints -- the
    smoothing decides only whether to trust the stroke, never where it points.
    """
    if len(pts) < 3:
        return list(pts)
    out = []
    half = max(1, window // 2)
    for i in range(len(pts)):
        lo, hi = max(0, i - half), min(len(pts), i + half + 1)
        chunk = pts[lo:hi]
        out.append((sum(p[0] for p in chunk) / len(chunk),
                    sum(p[1] for p in chunk) / len(chunk)))
    return out


# ------------------------------------------------------------- $1 internals

def _resample(pts, n=RESAMPLE_N):
    """Evenly spaced points along the path, so speed stops mattering."""
    if len(pts) < 2:
        return None
    total = _path_length(pts)
    if total <= 0:
        return None
    interval = total / (n - 1)
    out = [pts[0]]
    work = list(pts)
    carried = 0.0
    i = 1
    while i < len(work):
        ax, ay = work[i - 1]
        bx, by = work[i]
        d = math.hypot(bx - ax, by - ay)
        if d == 0.0:
            i += 1
            continue
        if carried + d >= interval:
            t = (interval - carried) / d
            q = (ax + t * (bx - ax), ay + t * (by - ay))
            out.append(q)
            work.insert(i, q)
            carried = 0.0
        else:
            carried += d
        i += 1
    while len(out) < n:                             # rounding shortfall
        out.append(out[-1])
    return out[:n]


def _centroid(pts):
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))


def _rotate_to_zero(pts):
    """Turn the stroke so its indicative angle is zero.

    This is what makes the recogniser rotation-invariant, which matters here:
    a head-drawn circle starts wherever the person happened to begin, and an
    L drawn by someone whose head rests at an angle is still an L.
    """
    cx, cy = _centroid(pts)
    theta = math.atan2(cy - pts[0][1], cx - pts[0][0])
    cos, sin = math.cos(-theta), math.sin(-theta)
    return [((p[0] - cx) * cos - (p[1] - cy) * sin + cx,
             (p[0] - cx) * sin + (p[1] - cy) * cos + cy) for p in pts]


def _scale_translate(pts, size=SQUARE):
    """Fit to a square and centre on the origin.

    Non-uniform on purpose. It is what makes a stroke squashed to a third of
    its width still recognisable, which is the difference between working and
    not working for somebody whose head barely turns.
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    width = (max(xs) - min(xs)) or 1.0
    height = (max(ys) - min(ys)) or 1.0
    scaled = [((p[0] - min(xs)) * size / width,
               (p[1] - min(ys)) * size / height) for p in pts]
    cx, cy = _centroid(scaled)
    return [(p[0] - cx, p[1] - cy) for p in scaled]


def _normalise(pts):
    resampled = _resample(pts)
    if resampled is None:
        return None
    return _scale_translate(_rotate_to_zero(resampled))


def _distance(a, b):
    return sum(math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1])
               for i in range(len(a))) / len(a)


def _score(candidate, template):
    half_diagonal = 0.5 * math.hypot(SQUARE, SQUARE)
    return max(0.0, 1.0 - _distance(candidate, template) / half_diagonal)


# ------------------------------------------------------------- the templates

def _circle(n=200):
    return [(120.0 * math.cos(2 * math.pi * i / n),
             120.0 * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _polyline(corners, n=200):
    out = []
    segments = len(corners) - 1
    per = max(2, n // segments)
    for s in range(segments):
        (x0, y0), (x1, y1) = corners[s], corners[s + 1]
        for i in range(per):
            t = i / per
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    out.append(corners[-1])
    return out


_IDEAL = {
    "circle": _circle(),
    "zed": _polyline([(-120, -90), (120, -90), (-120, 90), (120, 90)]),
    "ell": _polyline([(-90, -120), (-90, 120), (110, 120)]),
    "vee": _polyline([(-110, -110), (0, 120), (110, -110)]),
    "caret": _polyline([(-110, 110), (0, -120), (110, 110)]),
}

# name -> [normalised template at each completion fraction]
_TEMPLATES = {}
for _name, _ideal in _IDEAL.items():
    _built = []
    for _f in _PREFIXES:
        _part = _ideal[:max(3, int(len(_ideal) * _f))]
        _norm = _normalise(_part)
        if _norm is not None:
            _built.append(_norm)
    _TEMPLATES[_name] = _built


# ----------------------------------------------------------------- direction

def _direction_of(dx, dy, directions=4):
    """Bucket a net vector. Screen coordinates, so +y is downward."""
    if directions >= 8:
        angle = math.degrees(math.atan2(dy, dx)) % 360.0
        sector = int((angle + 22.5) // 45) % 8
        return ("right", "down_right", "down", "down_left",
                "left", "up_left", "up", "up_right")[sector]
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def direction(points, min_distance=DEFAULT_MIN_DISTANCE, directions=4,
              min_straightness=DEFAULT_MIN_STRAIGHTNESS,
              hold_floor=DEFAULT_HOLD_FLOOR):
    """``(name, confidence)``. name is a direction, ``"hold"``, or ``None``."""
    pts = _clean(points)
    if len(pts) < 2:
        return (None, 0.0)
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    net = math.hypot(dx, dy)

    if net < min_distance:
        # Deliberately still an answer. Resting in place is a gesture people
        # can perform when nothing else is available, and it is the one that
        # costs the least effort.
        path = _path_length(pts) or 1.0
        stillness = max(0.0, 1.0 - path / (min_distance * 3.0))
        if stillness < hold_floor:
            return (None, stillness)
        return ("hold", stillness)

    smoothed = _smooth(pts)
    path = _path_length(smoothed) or net
    straightness = min(1.0, net / path)
    if straightness < min_straightness:
        return (None, straightness)
    return (_direction_of(dx, dy, directions), straightness)


# --------------------------------------------------------------------- shape

def shape(points, floor=DEFAULT_SHAPE_FLOOR):
    """``(name, confidence)`` for the best-matching shape, or ``(None, 0.0)``.

    Scored against every stored prefix, so a stroke released early still has
    something sensible to match against.
    """
    pts = _clean(points)
    if len(pts) < 4:
        return (None, 0.0)
    candidate = _normalise(pts)
    if candidate is None:
        return (None, 0.0)
    best_name, best_score = None, 0.0
    for name, templates in _TEMPLATES.items():
        for template in templates:
            value = _score(candidate, template)
            if value > best_score:
                best_name, best_score = name, value
    if best_name is None or best_score < floor:
        return (None, best_score)
    return (best_name, best_score)


# ------------------------------------------------------------------ the API

def classify(points, min_distance=DEFAULT_MIN_DISTANCE, directions=4,
             shapes_enabled=False, min_straightness=DEFAULT_MIN_STRAIGHTNESS,
             shape_floor=DEFAULT_SHAPE_FLOOR, hold_floor=DEFAULT_HOLD_FLOOR):
    """What was drawn. ``(name, confidence, kind)``; never raises.

    `kind` is ``"shape"``, ``"direction"`` or ``None``. A shape that does not
    clear its floor falls back to the direction rather than to nothing: someone
    who meant to draw a circle and released early should still get the swipe
    their movement described, not silence.
    """
    try:
        if shapes_enabled:
            name, confidence = shape(points, shape_floor)
            if name is not None:
                return (name, confidence, "shape")
        name, confidence = direction(points, min_distance, directions,
                                     min_straightness, hold_floor)
        if name is None:
            return (None, confidence, None)
        return (name, confidence, "direction")
    except Exception:                               # noqa: BLE001
        return (None, 0.0, None)


def is_shape(name):
    return name in SHAPES

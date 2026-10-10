"""Hand-pose + swipe recognition from MediaPipe hand landmarks.

Turns the 21 hand landmarks into a small set of gestures used for PC control,
following the common, well-proven mapping (index → cursor, pinch → click, open-
palm swipes → navigation):

  point      index finger only            → move the cursor
  pinch      thumb + index tips together  → click / drag
  palm       open hand (3–4 fingers up)   → the "swipe-ready" pose
  swipe L/R  quick palm swipe             → previous / next
  swipe U/D  quick palm swipe             → scroll up / down
  peace      index + middle up            → right-click
  thumbs_up  only the thumb up            → Enter / confirm
  fist       no fingers up                → pause (hold the cursor)

Poses are per-frame; swipes are motion over a short window with a cooldown so
one flick fires once.
"""

import math
from collections import deque


class Gesture:
    __slots__ = ("pose", "swipe", "pinch_dist", "engaged", "orientation",
                 "lclick", "rclick")

    def __init__(self, pose, swipe=None, pinch_dist=1.0, engaged=False,
                 orientation="front", lclick=False, rclick=False):
        self.pose = pose            # point/palm/peace/thumbs_up/fist/other/none
        self.swipe = swipe          # left/right/up/down or None (one-shot)
        self.pinch_dist = pinch_dist   # thumb–middle distance (for the click bar)
        self.engaged = engaged      # index pointing → actively controlling
        self.orientation = orientation  # 'front' (palm to camera) or 'back'
        self.lclick = lclick        # thumb + middle together → left click
        self.rclick = rclick        # thumb + ring together   → right click


class GestureRecognizer:
    CLICK = 0.090           # thumb–middle distance → left click (forgiving)
    RCLICK = 0.070          # thumb–ring distance   → right click (deliberate)
    SWIPE_DIST = 0.22       # clearer flick required (fewer accidental swipes)
    SWIPE_WINDOW = 0.30     # seconds to measure the swipe over
    SWIPE_COOLDOWN = 1.0    # seconds between swipes

    def __init__(self):
        self._hist = deque(maxlen=12)   # (t, palm_x, palm_y)
        self._last_swipe = 0.0

    def reset(self):
        self._hist.clear()

    @staticmethod
    def _dist(lm, a, b):
        return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

    def update(self, lm, now) -> Gesture:
        f = self._fingers(lm)
        n = f["index"] + f["middle"] + f["ring"] + f["pinky"]
        # Click uses DIFFERENT fingers than pointing, so clicking never moves the
        # cursor: thumb+middle = left click, thumb+ring = right click. Middle and
        # ring are adjacent, so a middle-pinch can also come near the ring — make
        # the two mutually exclusive by "closest finger wins" so a left-click never
        # accidentally trips a right-click.
        d_mid = self._dist(lm, 4, 12)
        d_ring = self._dist(lm, 4, 16)
        lclick = d_mid < self.CLICK and d_mid <= d_ring
        rclick = d_ring < self.RCLICK and d_ring < d_mid
        pose = self._pose(f, n, lm)
        swipe = self._swipe(lm, now, pose)
        engaged = bool(f["index"])           # index pointing = actively controlling
        orientation = self._orientation(lm)
        return Gesture(pose, swipe, d_mid, engaged,
                       orientation, lclick, rclick)

    @staticmethod
    def _orientation(lm):
        """Palm facing the camera ('front') vs back of the hand ('back'), from the
        winding of index-MCP → pinky-MCP around the wrist. Ambiguous → 'front'."""
        ax, ay = lm[5].x - lm[0].x, lm[5].y - lm[0].y
        bx, by = lm[17].x - lm[0].x, lm[17].y - lm[0].y
        nz = ax * by - ay * bx
        return "back" if nz < -0.02 else "front"

    # ── finger states ──────────────────────────────────────
    @staticmethod
    def _fingers(lm):
        f = {}
        for name, tip, pip in (("index", 8, 6), ("middle", 12, 10),
                               ("ring", 16, 14), ("pinky", 20, 18)):
            f[name] = lm[tip].y < lm[pip].y - 0.015     # tip clearly above the joint
        # Thumb: tip splayed away from the index knuckle.
        f["thumb"] = math.hypot(lm[4].x - lm[5].x, lm[4].y - lm[5].y) > 0.12
        return f

    @staticmethod
    def _pose(f, n, lm):
        if n == 0:
            # Thumbs-up = only the thumb, and it points up (tip above its joint).
            thumb_up = f["thumb"] and lm[4].y < lm[3].y - 0.02
            return "thumbs_up" if thumb_up else "fist"
        if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
            return "peace"
        if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
            return "point"
        if n >= 4:                 # a deliberate, fully-open hand (for swipes)
            return "palm"
        return "other"

    # ── swipe (open-palm motion) ───────────────────────────
    def _swipe(self, lm, now, pose):
        cx, cy = lm[9].x, lm[9].y      # palm centre (middle-finger MCP)
        self._hist.append((now, cx, cy))
        if pose != "palm" or now - self._last_swipe < self.SWIPE_COOLDOWN:
            return None
        old = None
        for t, x, y in self._hist:
            if now - t <= self.SWIPE_WINDOW:
                old = (x, y)
                break
        if old is None:
            return None
        dx, dy = cx - old[0], cy - old[1]
        if abs(dx) > abs(dy) and abs(dx) > self.SWIPE_DIST:
            self._last_swipe = now
            return "right" if dx > 0 else "left"
        if abs(dy) > self.SWIPE_DIST:
            self._last_swipe = now
            return "down" if dy > 0 else "up"
        return None

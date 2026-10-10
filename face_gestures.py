"""Mouth-gesture recognition from MediaPipe face-mesh landmarks.

The head-only user clicks with the mouth: the vertical gap between the
inner-lip landmarks (13 and 14) versus ``mouth_open_threshold`` detects an
open mouth, and the *duration* the mouth is held open selects the action:

  < left_click_max (1s)     -> LEFT click  (double on a rapid second tap)
  [left_click_max, grab)    -> RIGHT click
  >= grab_threshold (3s)    -> GRAB (drag start); close mouth to DROP
  drag > max_drag_seconds   -> safety AUTO-RELEASE (button never sticks)

This module is pure logic — no pyautogui, no Qt, no camera — extracted
verbatim from the state machine that used to live inline in
``vision_mouse._vision_worker`` so it can be unit-tested deterministically.
``VisionMouse`` maps the returned :class:`FaceGesture` to mouse side-effects.

Wave-2 adds an OPTIONAL second stage riding the same FaceMesh landmarks:
discrete :class:`FaceEvent` s — deliberate blink / double-blink / wink /
eyebrow-raise / head-tilt — produced by :meth:`FaceGestureRecognizer.
update_events`. It is cleanly separated from the mouth machine and emits
NOTHING unless ``events_enabled`` is turned on (a default AbilityProfile
leaves it off, so default behaviour is byte-identical to Wave-1).
"""

import math


# ══════════════════════════════════════════════════════════════════════
# Landmark indices for the optional event stage (ALL base-mesh indices;
# refine_landmarks=False is enough — nothing here needs the iris points).
# ══════════════════════════════════════════════════════════════════════
# Eye-Aspect-Ratio points: (p1, p2, p3, p4, p5, p6) with p1/p4 the horizontal
# corners and (p2,p6)/(p3,p5) the vertical lid pairs.
_LEFT_EYE_EAR = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)
# Brow-raise: vertical gap brow→upper-lid, one pair per side.
_BROW_PAIRS = ((105, 159), (334, 386))
# Outer eye corners: inter-ocular normalisation + head-roll (tilt) line.
_EYE_OUTER_L = 33
_EYE_OUTER_R = 263
# Highest landmark index the event stage reads. update_events() no-ops on
# shorter lists (e.g. the mouth-only synthetic landmarks used in tests).
_MAX_EVENT_LANDMARK = 387


class FaceEvent:
    """One discrete face-gesture event from the optional event stage."""

    __slots__ = ("kind",)

    KINDS = ("blink", "double_blink", "wink_l", "wink_r", "brow_raise",
             "tilt_left", "tilt_right", "tilt_end")

    def __init__(self, kind):
        self.kind = kind

    def __repr__(self):  # pragma: no cover - debugging nicety
        return f"FaceEvent({self.kind!r})"


class FaceGestureParams:
    """Tunables + per-gesture enables for the event stage (ALL default OFF)."""

    __slots__ = (
        "blink_enabled", "wink_enabled", "brow_enabled", "tilt_enabled",
        "blink_ear_threshold", "blink_min_hold_s", "double_blink_window_s",
        "wink_margin", "brow_raise_ratio", "brow_hold_s", "brow_rearm_ratio",
        "brow_baseline_tau_s", "tilt_angle_deg", "tilt_hold_s",
        "tilt_release_hyst_deg",
    )

    def __init__(self, blink_enabled=False, wink_enabled=False,
                 brow_enabled=False, tilt_enabled=False,
                 blink_ear_threshold=0.21, blink_min_hold_s=0.35,
                 double_blink_window_s=0.7, wink_margin=0.06,
                 brow_raise_ratio=1.25, brow_hold_s=0.2,
                 brow_rearm_ratio=1.05, brow_baseline_tau_s=3.0,
                 tilt_angle_deg=18.0, tilt_hold_s=0.5,
                 tilt_release_hyst_deg=5.0):
        self.blink_enabled = blink_enabled
        self.wink_enabled = wink_enabled
        self.brow_enabled = brow_enabled
        self.tilt_enabled = tilt_enabled
        self.blink_ear_threshold = blink_ear_threshold   # EAR below = closed
        self.blink_min_hold_s = blink_min_hold_s         # shorter = natural blink
        self.double_blink_window_s = double_blink_window_s
        self.wink_margin = wink_margin                   # other eye must be this far OPEN
        self.brow_raise_ratio = brow_raise_ratio         # gap > baseline*ratio = raised
        self.brow_hold_s = brow_hold_s                   # sustain before the event
        self.brow_rearm_ratio = brow_rearm_ratio         # re-arm below baseline*this
        self.brow_baseline_tau_s = brow_baseline_tau_s   # slow-EMA time constant
        self.tilt_angle_deg = tilt_angle_deg             # roll beyond = tilted
        self.tilt_hold_s = tilt_hold_s                   # sustain before the event
        self.tilt_release_hyst_deg = tilt_release_hyst_deg  # end below angle-hyst


class FaceGesture:
    """One frame's mouth-gesture result."""

    __slots__ = ("action", "mouth_open", "held", "state")

    def __init__(self, action=None, mouth_open=False, held=0.0, state="move"):
        self.action = action          # None|"click"|"double"|"right"|"drag"|"drop"|"auto_release"
        self.mouth_open = mouth_open  # gated mouth-open detection this frame
        self.held = held              # seconds the mouth has been held open
        self.state = state            # "move"|"click"|"right"|"drag" (head-guide HUD)


class FaceGestureRecognizer:
    """Timing state machine for mouth open/close gestures.

    Owns the mouth-hold timer, the double-click window, and the drag latch.
    ``update()`` is called once per processed frame with the face landmarks
    and the current time; it returns a :class:`FaceGesture` describing any
    action that fired on this frame plus the live HUD state.
    """

    def __init__(self, mouth_open_threshold=10, left_click_max=1.0,
                 grab_threshold=3.0, double_click_threshold=0.6,
                 max_drag_seconds=8.0):
        self.mouth_open_threshold = mouth_open_threshold
        self.left_click_max = left_click_max          # release before this = left click
        self.grab_threshold = grab_threshold          # hold this long = grab/drag
        self.double_click_threshold = double_click_threshold
        self.max_drag_seconds = max_drag_seconds

        # Swipe mode. When on, the mouth stops being a click/drag control and
        # becomes the hold that brackets a gesture: open starts the stroke,
        # close ends it. Nothing here can reach `grab_threshold` while it is
        # set, which is the point -- it makes a stuck mouse button structurally
        # impossible rather than merely unlikely. Off by default; a run that
        # never sets it behaves exactly as before.
        self.swipe_mode = False
        self._swiping = False      # safety auto-release

        # ----- Mouth gesture timing state -----
        self.mouth_open_start = None
        self.last_click_time = 0
        self._dragging = False
        self._drag_started_at = 0.0

        # ----- Optional discrete-event stage (Wave-2; default OFF) -----
        # With events_enabled False (the default), update_events() always
        # returns [] and none of the state below is ever consulted — the
        # recognizer behaves byte-identically to Wave-1.
        self.events_enabled = False
        self._event_params = FaceGestureParams()
        self.reset_events()

    # ------------------------------------------------------------------
    @property
    def is_dragging(self) -> bool:
        return self._dragging

    @is_dragging.setter
    def is_dragging(self, value) -> None:
        # Writable so external teardown paths (and tests) can force the latch.
        self._dragging = bool(value)

    def reset(self) -> None:
        """Clear mouth timing + drag state (face loss / stop / click gate-off).

        Deliberately does NOT clear ``last_click_time`` — the original
        ``_release_drag`` never did, so a click just before a face loss can
        still pair into a double-click on return, exactly as before.
        """
        self.mouth_open_start = None
        self._dragging = False
        self._swiping = False
        self._drag_started_at = 0.0
        # Face loss / stop also clears any half-finished discrete event.
        self.reset_events()

    def reset_events(self) -> None:
        """Clear the event stage's in-flight state (face loss / stop).

        The brow baseline is dropped too — it re-learns in a couple of
        seconds and a face that left the frame may come back differently lit
        or posed, so a stale baseline is worse than none.
        """
        # Eye closure in flight: None | "blink" | "wink_l" | "wink_r"
        self._eye_close_kind = None
        self._eye_close_start = 0.0
        self._last_blink_time = 0.0        # arms the double-blink window
        # Brow-raise machine: idle | pending | triggered
        self._brow_baseline = None
        self._brow_state = "idle"
        self._brow_pending_since = 0.0
        self._brow_last_ts = None
        # Head-tilt machine.
        self._tilt_active = None           # None | "left" | "right"
        self._tilt_pending = None
        self._tilt_pending_since = 0.0

    # ------------------------------------------------------------------
    # Mouth opening detection (lifted verbatim from VisionMouse)
    # ------------------------------------------------------------------
    def detect_mouth_open(self, landmarks, screen_h) -> bool:
        """Detect mouth opening using MediaPipe face mesh landmarks."""
        try:
            upper = landmarks[13].y * screen_h
            lower = landmarks[14].y * screen_h
            return (lower - upper) > self.mouth_open_threshold
        except Exception as e:
            print(f"[VISION] Mouth detection error: {e}", flush=True)
            return False

    # ------------------------------------------------------------------
    # State machine — one step per processed frame
    # ------------------------------------------------------------------
    def update(self, landmarks, now, screen_h, enabled=True) -> FaceGesture:
        """Advance the mouth state machine one frame; pure logic, no side effects.

        ``enabled=False`` mirrors the original click gating: detection is
        forced to "closed" but the machine still steps, so a gesture that was
        in progress when the gate turned off releases exactly as it used to.
        """
        mouth_open = self.detect_mouth_open(landmarks, screen_h) if enabled else False

        action = None
        held = 0.0

        if self.swipe_mode:
            return self._step_swipe(mouth_open, now)

        if mouth_open:
            if self.mouth_open_start is None:
                self.mouth_open_start = now
            held = now - self.mouth_open_start

            # Held past the grab threshold → start a drag (grab).
            if held >= self.grab_threshold and not self._dragging:
                self._dragging = True
                self._drag_started_at = now
                self.last_click_time = 0
                action = "drag"
            # Safety: auto-release a drag that has run too long so a noisy
            # mouth signal can never leave the button stuck down.
            elif self._dragging and (now - self._drag_started_at) > self.max_drag_seconds:
                self._dragging = False
                self.mouth_open_start = None
                action = "auto_release"

        else:
            if self.mouth_open_start is not None:
                held = now - self.mouth_open_start
                if self._dragging:
                    # mouth closed while dragging → DROP
                    self._dragging = False
                    action = "drop"
                elif held < self.left_click_max:
                    # quick (<1s) → left click (double on a rapid second tap)
                    if now - self.last_click_time <= self.double_click_threshold:
                        self.last_click_time = 0
                        action = "double"
                    else:
                        self.last_click_time = now
                        action = "click"
                else:
                    # medium hold (1s..3s) released before grabbing → right click
                    self.last_click_time = 0
                    action = "right"
                self.mouth_open_start = None

        # Live gesture state for the on-screen head guide — computed after
        # the transition, exactly like the old inline emit block.
        if self._dragging:
            state = "drag"
        elif mouth_open:
            h = now - (self.mouth_open_start or now)
            state = "right" if h >= self.left_click_max else "click"
        else:
            state = "move"

        return FaceGesture(action=action, mouth_open=mouth_open, held=held,
                           state=state)

    def _step_swipe(self, mouth_open, now):
        """The mouth as a bracket rather than a button.

        Open starts a stroke, close ends it, and the hold in between produces
        no action at all however long it lasts. A drag can never begin here, so
        the caller does not need a timeout to protect against one.
        """
        action = None
        held = 0.0
        if mouth_open:
            if self.mouth_open_start is None:
                self.mouth_open_start = now
                self._swiping = True
                action = "swipe_start"
            held = now - self.mouth_open_start
        elif self.mouth_open_start is not None:
            held = now - self.mouth_open_start
            self.mouth_open_start = None
            self._swiping = False
            action = "swipe_end"
        return FaceGesture(action=action, mouth_open=mouth_open, held=held,
                           state="swipe" if self._swiping else "move")

    # ══════════════════════════════════════════════════════════════════
    # Optional discrete-event stage (Wave-2) — blink / wink / brow / tilt
    # ══════════════════════════════════════════════════════════════════

    def set_event_params(self, params: FaceGestureParams) -> None:
        """Swap in new event-stage tunables (from an AbilityProfile)."""
        self._event_params = params

    @property
    def tilt_active(self):
        """The currently *held* head tilt: "left" / "right" / None."""
        return self._tilt_active

    def update_events(self, landmarks, now) -> list:
        """Advance the event stage one frame; returns a list of FaceEvent.

        Always ``[]`` unless ``events_enabled`` — the default recognizer is
        therefore byte-identical to Wave-1. Safe with short landmark lists
        (mouth-only callers/tests): if the eye/brow indices are missing it
        no-ops. Per-gesture enables gate inside, so e.g. blink events only
        fire when ``blink_enabled``.

        Suppression policy (documented choice — it lives HERE, not in
        VisionMouse): blink/wink/brow events are suppressed, and their
        in-flight state cleared, while the mouth is open or a drag is
        latched, so the mouth state machine always wins. Tilt keeps working
        (tilt-scroll during a drag is useful and cannot collide with mouth
        clicks).
        """
        if not self.events_enabled:
            return []
        try:
            count = len(landmarks)
        except TypeError:
            return []
        if count <= _MAX_EVENT_LANDMARK:
            return []

        p = self._event_params
        events = []
        suppress = (self.mouth_open_start is not None) or self._dragging
        if p.blink_enabled or p.wink_enabled:
            self._step_eyes(landmarks, now, p, events, suppress)
        if p.brow_enabled:
            self._step_brow(landmarks, now, p, events, suppress)
        if p.tilt_enabled:
            self._step_tilt(landmarks, now, p, events)
        return events

    # ── shared landmark math ────────────────────────────────────────
    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    @classmethod
    def _ear(cls, lm, idx) -> float:
        """Eye Aspect Ratio — (|p2-p6| + |p3-p5|) / (2*|p1-p4|)."""
        p1, p2, p3, p4, p5, p6 = (lm[i] for i in idx)
        horiz = cls._dist(p1, p4)
        if horiz <= 1e-9:
            return 1.0          # degenerate face — treat as open (no event)
        return (cls._dist(p2, p6) + cls._dist(p3, p5)) / (2.0 * horiz)

    # ── blink / double-blink / wink ─────────────────────────────────
    def _step_eyes(self, lm, now, p, events, suppress) -> None:
        if suppress:
            # Mouth gesture in progress: never let a squint from the mouth
            # movement fire an eye event; drop any half-finished closure.
            self._eye_close_kind = None
            return
        ear_l = self._ear(lm, _LEFT_EYE_EAR)
        ear_r = self._ear(lm, _RIGHT_EYE_EAR)
        thr = p.blink_ear_threshold

        if self._eye_close_kind is None:
            # Closure onset. Wink needs the OTHER eye clearly open (margin)
            # so half-lidded ambiguity never mis-fires a wink.
            if ear_l < thr and ear_r < thr:
                self._eye_close_kind = "blink"
                self._eye_close_start = now
            elif ear_l < thr and ear_r > thr + p.wink_margin:
                self._eye_close_kind = "wink_l"
                self._eye_close_start = now
            elif ear_r < thr and ear_l > thr + p.wink_margin:
                self._eye_close_kind = "wink_r"
                self._eye_close_start = now
            return

        if ear_l < thr and ear_r < thr and self._eye_close_kind != "blink":
            # Asymmetric onset: a "wink" that deepened into both eyes closed
            # was really a blink. Upgrade (never downgrade), keep the start.
            self._eye_close_kind = "blink"
            return
        if ear_l < thr or ear_r < thr:
            return              # still (partly) closed — keep holding

        # Fully reopened → the closure ends; a long-enough hold emits.
        kind = self._eye_close_kind
        held = now - self._eye_close_start
        self._eye_close_kind = None
        if held < p.blink_min_hold_s:
            return              # natural blink — rejected
        if kind == "blink":
            if not p.blink_enabled:
                return
            if (self._last_blink_time
                    and now - self._last_blink_time <= p.double_blink_window_s):
                # Second deliberate blink inside the window → double_blink
                # INSTEAD of a second plain blink.
                self._last_blink_time = 0.0
                events.append(FaceEvent("double_blink"))
            else:
                self._last_blink_time = now
                events.append(FaceEvent("blink"))
        elif p.wink_enabled:
            events.append(FaceEvent(kind))

    # ── eyebrow raise ───────────────────────────────────────────────
    def _step_brow(self, lm, now, p, events, suppress) -> None:
        if suppress:
            # Opening the mouth distorts the brow line: drop any pending
            # trigger and pause baseline learning (keep the learned value).
            if self._brow_state == "pending":
                self._brow_state = "idle"
            self._brow_last_ts = now
            return
        iod = self._dist(lm[_EYE_OUTER_L], lm[_EYE_OUTER_R])
        if iod <= 1e-9:
            return
        # Mean brow→upper-lid gap, normalised by inter-ocular distance so it
        # is invariant to how close the user sits.
        gap = sum(self._dist(lm[a], lm[b]) for a, b in _BROW_PAIRS) / (2.0 * iod)
        if self._brow_baseline is None:
            self._brow_baseline = gap
            self._brow_last_ts = now
            return
        dt = max(0.0, now - (self._brow_last_ts if self._brow_last_ts is not None else now))
        self._brow_last_ts = now

        if self._brow_state == "idle":
            if gap > self._brow_baseline * p.brow_raise_ratio:
                self._brow_state = "pending"
                self._brow_pending_since = now
            else:
                # Slow EMA baseline (tau ≈ 3 s), only while no event is in
                # flight — tracks lighting/pose drift, not the raise itself.
                if p.brow_baseline_tau_s > 0:
                    k = min(1.0, dt / p.brow_baseline_tau_s)
                else:
                    k = 1.0
                self._brow_baseline += k * (gap - self._brow_baseline)
        elif self._brow_state == "pending":
            if gap <= self._brow_baseline * p.brow_raise_ratio:
                self._brow_state = "idle"          # dipped early — no event
            elif now - self._brow_pending_since >= p.brow_hold_s:
                self._brow_state = "triggered"
                events.append(FaceEvent("brow_raise"))
        else:  # "triggered" — re-arm only after settling near baseline
            if gap < self._brow_baseline * p.brow_rearm_ratio:
                self._brow_state = "idle"

    # ── head tilt (roll) ────────────────────────────────────────────
    def _step_tilt(self, lm, now, p, events) -> None:
        a = lm[_EYE_OUTER_L]
        b = lm[_EYE_OUTER_R]
        # Roll of the outer-eye-corner line 33→263, in degrees. Convention:
        # positive roll = landmark 263 LOWER in the image → "right";
        # negative → "left". (Direction→action is remappable anyway via
        # face_gesture_map, so the convention just needs to be stable.)
        roll = math.degrees(math.atan2(b.y - a.y, b.x - a.x))
        if roll > p.tilt_angle_deg:
            side = "right"
        elif roll < -p.tilt_angle_deg:
            side = "left"
        else:
            side = None

        if self._tilt_active is not None:
            # Held tilt ends with hysteresis: only once the roll (toward the
            # held side) falls under tilt_angle_deg - hysteresis. A sign flip
            # counts too (held_roll goes negative).
            release = max(0.0, p.tilt_angle_deg - p.tilt_release_hyst_deg)
            held_roll = roll if self._tilt_active == "right" else -roll
            if held_roll < release:
                self._tilt_active = None
                self._tilt_pending = None
                events.append(FaceEvent("tilt_end"))
            return

        if side is None:
            self._tilt_pending = None
            return
        if self._tilt_pending != side:
            self._tilt_pending = side
            self._tilt_pending_since = now
            return
        if now - self._tilt_pending_since >= p.tilt_hold_s:
            self._tilt_active = side
            self._tilt_pending = None
            events.append(FaceEvent("tilt_left" if side == "left" else "tilt_right"))

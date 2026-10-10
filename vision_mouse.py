import os
import sys
import time
import math
from collections import deque

# Support running this module directly from its file location.
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

import mistey.config as config
from mistey.core.supervised_thread import SupervisedThread
from mistey.input.face_gestures import FaceGestureRecognizer, FaceGestureParams
from mistey.input import gesture_actions
from mistey.input import smoothing
from mistey.input.gesture_actions import DEFAULT_FACE_MAP
from mistey.health import struggle as struggle_log



# A flush every few minutes turns one long sitting into several data points,
# which is what a day-to-day trend needs, and caps memory at a few MB however
# long someone works.
HEALTH_FLUSH_SECONDS = 300.0
# The FIRST window of a session is short on purpose. Five minutes of seeing
# nothing happen is indistinguishable from a broken feature, and the person
# watching for it may have had to work to turn it on at all. One minute is still
# well above MIN_SAMPLES, so the measurement is real and not a placeholder.
HEALTH_FIRST_FLUSH_SECONDS = 60.0
HEALTH_MIN_SAMPLES = 120          # under ~10s of use there is nothing to measure
# ...and the same floor expressed as the time it was meant to be. The count was
# chosen when the loop ran near 12.5 Hz, where 120 samples is 9.6 seconds. At
# the ~29 Hz that One-Euro unlocks it would be four seconds, which is not long
# enough to fit a tremor period to, so the intent has to be stated in seconds
# or it quietly weakens the moment the loop speeds up. Equivalent at the old
# rate, so nothing changes for a profile that stays on the EMA.
HEALTH_MIN_SECONDS = 9.5
HEALTH_MAX_SAMPLES = 12000        # ~19 min at 10.4 Hz, ~7 min at 29 Hz

# A dead-zone rejection only counts as *blocked intent* if the movement was
# substantial. Somebody sitting still produces a delta near zero every frame,
# and counting that would report thousands of "struggles" from a person who
# was reading. The upper three-quarters of the dead zone is movement large
# enough to be deliberate and still thrown away.
DEADZONE_INTENT_FRACTION = 0.25

# ===== The dead-zone's missing twin =====
# The dead-zone refuses movement SMALLER than its radius, so tremor does not
# jitter the pointer. Nothing refused movement that was far too LARGE, so a
# flexor spasm or a knock to the camera throws the cursor across the screen.
# Together the two are a band-pass on cursor displacement, and they belong
# documented as a pair.
#
# The ceiling comes from the person's own movement, not a constant: a jump in
# pixels per frame means something different at every screen size, gain, and
# frame rate, and the rate alone now varies between 10 Hz and 29 Hz depending
# on the filter. Measured against their own 95th-percentile frame-to-frame
# movement, the same setting behaves the same way for everybody.
FAST_JUMP_OVER_USUAL = 8.0
# ...and a floor, which only ever makes the clamp MORE permissive. It covers
# two cases. The first seconds of a session have no distribution to take a
# percentile of. And somebody whose head barely moves has a tiny 95th
# percentile, so a bare multiple of it would hold back their ordinary pointing
# -- the limitation this software exists to accommodate deciding what it
# refuses, which is the same mistake the swipe hold-floor was added to fix.
#
# As a fraction of the screen width rather than a pixel count, because the
# thing being caught is a jump "across the screen": 400 px is a quarter of a
# 1366 laptop and a tenth of a 4K monitor, so a constant would be a different
# feature on different hardware.
#
# THIS NUMBER IS NOT YET SETTLED, and the honest reason is that the two
# distributions overlap. A spasm throwing the pointer at a screen edge produces
# roughly 500-1000 px in one frame; a deliberate fast flick produces perhaps
# 200-400. There is no measurement of either on a real affected person, only
# arithmetic, so a quarter of the width is a starting point and not a finding.
#
# Three things make that acceptable rather than reckless: it is off by default,
# a hold is bounded to FAST_JUMP_MAX_HOLD_FRAMES so the worst case is a brief
# delay rather than a refusal, and every hold is counted as `fast_jump_held` --
# which is the instrument that will say whether this is set right, from the
# people it is set for, rather than from a developer's guess about them.
FAST_JUMP_FLOOR_FRACTION = 0.25
# How many frames of history before the percentile is trusted at all.
FAST_JUMP_MIN_HISTORY = 60

# And the escape hatch, which is the whole reason this is safe to ship.
#
# Holding a movement back cannot be allowed to persist, or a head that really
# did move fast and STAYS there leaves the cursor frozen at the old place
# forever -- every subsequent frame is the same large jump from the same stale
# point. A spasm is brief and a deliberate movement is not, so after this many
# consecutive held frames the movement is accepted. Worst case is a delay of
# about a tenth of a second on a genuine fast flick; the alternative is a
# pointer that stops, which for somebody whose head is their only input is the
# worst failure available.
FAST_JUMP_MAX_HOLD_FRAMES = 3
# Consecutive blocked frames before this is worth a captured moment: about a
# second of pushing against the filter with nothing happening.
DEADZONE_EVIDENCE_STREAK = 12
# A dwell that re-arms early is just moving to a new target. One that re-arms
# after most of the hold is a click the person nearly had and lost, which is
# the thing worth counting.
DWELL_NEARLY_FRACTION = 0.5


def face_span(landmarks) -> float:
    """Apparent face size: the span between the outer eye corners.

    In normalised frame units, so it shrinks as you sit further back — which
    makes it a usable stand-in for distance without any depth sensor. The outer
    eye corners (33, 263) are used rather than the face oval because they sit
    on rigid bone: expressions, speech and mouth gestures do not move them.

    Returns 0.0 if the landmarks are unusable, which callers read as "no idea".
    """
    try:
        left, right = landmarks[33], landmarks[263]
        return math.hypot(right.x - left.x, right.y - left.y)
    except Exception:
        return 0.0



def head_roll(landmarks) -> float:
    """Head tilt in degrees, from the same two rigid landmarks as face_span.

    Positive is the person's right ear toward their right shoulder as the image
    is displayed. Read from the outer eye corners (33, 263) for the reason
    face_span uses them: they sit on bone, so speaking or a mouth-click gesture
    cannot register as a change of posture.

    Returns 0.0 when the landmarks are unusable — indistinguishable from "level",
    which is the right way to be wrong here: a report should never claim a tilt
    that was really a dropped frame.
    """
    try:
        left, right = landmarks[33], landmarks[263]
        return math.degrees(math.atan2(right.y - left.y, right.x - left.x))
    except Exception:
        return 0.0

def distance_scale(span, reference=None, lo=None, hi=None) -> float:
    """How much to amplify cursor gain for a face of this apparent size.

    1.0 at the reference distance, above 1 further away, below 1 closer — so
    the same physical head movement produces the same cursor travel wherever
    the person is sitting. Clamped at both ends: one bad frame must never
    launch the cursor across the screen.
    """
    reference = config.FACE_SPAN_REFERENCE if reference is None else reference
    lo = config.FACE_SPAN_SCALE_MIN if lo is None else lo
    hi = config.FACE_SPAN_SCALE_MAX if hi is None else hi
    if not span or span <= 0:
        return 1.0                      # unknown distance: change nothing
    return max(lo, min(hi, reference / span))


def _phase(message):
    """Record a milestone to the startup log.

    The shipped build is windowed, so print() goes nowhere. Anything needed to
    explain "the head cursor isn't moving" has to reach a file a caregiver can
    send us, not a console nobody can see.
    """
    try:
        from mistey.diagnostics import phase
        phase("vision: " + message)
    except Exception:
        pass


class VisionMouse:
    def __init__(self, resource_guard, show_window: bool = True, click_callback=None,
                 frame_callback=None):
        self.guard = resource_guard
        self.show_window = show_window
        self.click_callback = click_callback
        self.frame_callback = frame_callback     # optional live preview (BGR frames)
        self._last_frame_emit = 0.0
        self.gesture_callback = None             # (state, 0.0) for the head guide
        # Called with a signal name when something worth adapting to happens.
        # Wired by the controller to the orchestrator. Without it the adapter
        # cannot see tracking loss at all, which is why its face_lost counter
        # sat at zero for every user who ever ran this.
        self.signal_callback = None
        self._last_gesture_emit = 0.0

        # Supervised worker (auto-restarts on transient failures, never silent)
        self._supervisor = None
        self._cap = None
        self._holds_camera = False

        # Screen
        try:
            import pyautogui
        except ImportError as exc:
            raise ImportError(
                "pyautogui is required for VisionMouse control. Install it with `pip install pyautogui`."
            ) from exc

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.screen_w, self.screen_h = pyautogui.size()

        # EMA state
        self.prev_x = None
        self.prev_y = None

        # Which smoothing runs, and the One-Euro instance if it is selected.
        # "ema" is the default everywhere, and the EMA branch below is left
        # exactly as it was, so a profile-less run and every saved profile are
        # byte-identical. See smoothing.py for why the choice is not cosmetic:
        # the EMA's smoothing depends on the frame rate the loop happens to
        # achieve, and One-Euro's does not.
        self.filter_mode = "ema"
        self._one_euro = None
        # Where the head's resting position is taken to be. None means the
        # middle of the screen, which is what the mapping always assumed.
        self._origin_x = None
        self._origin_y = None
        self._recentre_pending = False
        self._recentre_source = ""
        # Hold back a movement too fast to be voluntary. Off by default, like
        # every other optional input behaviour here, so a profile-less run and
        # every saved profile are byte-identical.
        self.spasm_clamp_enabled = False
        self._jump_samples = deque(maxlen=600)   # per-FRAME movement, px
        self._last_target_x = None
        self._last_target_y = None
        self._jump_hold_frames = 0
        # Set by the controller to the orchestrator's speech. Optional, because
        # the driver has to work with no orchestrator at all -- the tests build
        # it that way and so does a bare --direct run that fails preflight.
        self.say_callback = None
        # How often the mesh runs, which follows the filter. The EMA value is
        # the one the loop has always used; One-Euro lifts the throttle because
        # it can survive the faster rate and the EMA cannot.
        self._mp_interval = config.FACE_MP_INTERVAL

        # ===== Motor parameters =====
        # Seeded from config.py (the "Default" preset) so a profile-less run is
        # byte-identical to before. apply_motor_profile() overrides these live.
        self.alpha = config.FACE_EMA_ALPHA
        self.sensitivity_factor = config.FACE_SENSITIVITY_FACTOR
        self.diagonal_sensitivity_factor = config.FACE_DIAGONAL_SENSITIVITY_FACTOR
        self.max_face_not_visible_frames = config.MAX_FACE_NOT_VISIBLE_FRAMES
        self.dead_zone_radius = config.FACE_DEAD_ZONE_RADIUS

        # ----- Mouth gesture state machine (extracted, unit-testable) -----
        # Owns the mouth timing thresholds + the drag latch:
        #   < left_click_max        -> LEFT click (double on a rapid second tap)
        #   [left_click_max, grab)  -> RIGHT click
        #   >= grab_threshold       -> DRAG (grab); close mouth to drop
        # mouth_open_threshold lives here too; the property below keeps the
        # old `self.mouth_open_threshold` attribute working (config seeding,
        # apply_motor_profile) with a single source of truth.
        self._face = FaceGestureRecognizer(
            mouth_open_threshold=config.MOUTH_OPEN_THRESHOLD)

        # Click modes (profile-driven)
        self.mouth_click_enabled = True
        self.dwell_click_enabled = False
        self.dwell_time = 1.0
        self.dwell_radius = 40

        # ----- Click state -----
        self.face_not_visible_counter = 0

        # ----- Cursor / dead-zone state -----
        self._committed_x = None
        self._committed_y = None

        # ----- Jitter estimate (for live adaptation) -----
        self._jitter_samples = deque(maxlen=60)

        # ----- Dwell-click state -----
        self._dwell_anchor = None
        self._dwell_start = 0.0
        self._dwell_fired = False

        # ----- Click gating (mouth timing itself lives in self._face) -----
        self.clicks_enabled = True    # master gate (Scroll Scope pauses clicking)
        self.lost_release_frames = 5  # genuine face-absent frames before dropping a drag

        # ----- Switch/scanning integration (Wave-2) — default OFF -----
        # When scanning drives the cursor, this driver keeps computing
        # landmarks + face gestures (they feed the switch) but never moves
        # the OS cursor (cursor_enabled False) and one face gesture becomes
        # the SWITCH: switch_gesture in {"blink", "mouth", None}. While a
        # gesture is the switch it fires switch_callback() INSTEAD of its
        # normal action. Independent of clicks_enabled — scanning replaces
        # clicking entirely.
        self.cursor_enabled = True
        self.switch_callback = None       # zero-arg fn set by the controller
        self.switch_gesture = None        # None | "blink" | "mouth"
        self._switch_mouth_prev = False   # rising-edge memory (mouth switch)
        self._switch_mouth_ts = 0.0       # 600 ms debounce anchor

        # ----- Optional face-gesture events (Wave-2) — ALL default OFF -----
        # blink/double-blink/wink/brow-raise/head-tilt ride the same FaceMesh
        # frame. A default AbilityProfile leaves them disabled, so a
        # profile-less run stays byte-identical to before (zero regression).
        self._face_events_on = False
        self._face_map = dict(DEFAULT_FACE_MAP)
        self._tilt_repeat_ts = 0.0    # throttles the held-tilt repeat action

        # ----- Movement record (mistey/health) — OFF unless asked for -----
        # Sampled raw, flushed on a timer so a four-hour sitting becomes several
        # data points rather than one averaged smear, and so memory is bounded
        # however long someone works.
        self.health_monitoring_enabled = False
        # Matches UserProfile.user_id so a driver that never meets an
        # AbilityEngine still writes where the report will look.
        self.health_user_id = "default_user"
        self._health_samples = deque(maxlen=HEALTH_MAX_SAMPLES)
        self._health_started = 0.0
        self._health_last_flush = 0.0
        self._health_flushes = 0       # first one comes round sooner
        # Where and when each click landed, which is what turns the stream into
        # a series of reaches: the person moved the pointer somewhere and
        # clicked, so that is what they were aiming at. Without it
        # motor_metrics.acquisition_time has no target and never runs -- which
        # is why acquisition_s and overshoots had never been recorded once,
        # despite the report carrying labels for them.
        #
        # Far smaller than the sample buffer because clicks are rare: a long
        # window holds a few hundred, and 2000 cannot be reached in one.
        self._click_marks = deque(maxlen=2000)
        # Posture and reach over the sitting:
        #     (t, nose_x, nose_y, roll_deg, face_span)
        # every one of which is already computed on every analysed frame into
        # _last_pose and thrown away unless a moment of difficulty happened to
        # be captured.
        #
        # face_span rides along because it is what makes the other three mean
        # anything across sittings: a nose excursion in frame fractions is
        # larger when somebody sits closer, so dividing by the apparent size of
        # their face is what turns "how far the head turned" into a quantity
        # that can be compared between days. Same geometry distance_scale uses
        # on the cursor path, for the same reason.
        #
        # A separate buffer rather than widening the movement sample, for the
        # same reason _swipe_points is separate: _health_samples is read by
        # _clean() and everything downstream of it, and changing the tuple
        # shape to carry two more fields would put the cursor's own measurement
        # path at risk for the sake of a nice-to-have.
        self._pose_samples = deque(maxlen=HEALTH_MAX_SAMPLES)

        # ----- Struggle record — rides the same flush, same opt-in ---------
        # Counting what did NOT work costs one dict increment per event. The
        # counter is created unconditionally because an empty one is free; it
        # is only ever fed when health_monitoring_enabled is true.
        self._struggle = struggle_log.Counter()
        self._dz_streak = 0            # consecutive dead-zone rejections
        self._last_pose = {}           # nose/roll/span of the last good frame
        self._at_ceiling = []          # motor params pinned at their bounds

        # ----- Swipe gestures (mistey/input/swipe.py) -- OFF unless asked ----
        # A separate buffer is unavoidable: _health_samples is gated behind the
        # health opt-in, nose-only, and emptied on every flush, and
        # _jitter_samples holds magnitudes with no direction. 150 points is
        # ~16 s at 9.1 Hz, far more than any stroke.
        self.swipe_gestures_enabled = False
        self.swipe_shapes_enabled = False
        self.swipe_min_distance_px = 120
        self.swipe_directions = 4
        self.swipe_shape_floor = 0.80
        self._swipe_map = {}
        self._swipe_points = deque(maxlen=150)
        self._swiping = False
        self._swipe_hint = ""          # what the overlay is currently naming
        self._swipe_hint_ts = 0.0
        self.swipe_callback = None     # (points, label) -> None, for the overlay

        # ----- Captured moments — a further, separate opt-in --------------
        self.health_evidence_enabled = False
        self.health_photos_enabled = False
        self._evidence = None          # built lazily; None means capture nothing

        # ----- Distance compensation (see config.FACE_DISTANCE_COMPENSATION) --
        self.distance_compensation = config.FACE_DISTANCE_COMPENSATION
        self._span_ema = None          # smoothed apparent face size
        self._too_far_since = 0.0      # when the face first became too small

        # ----- Eye-gaze cursor (Wave-2) — default OFF ("nose"), zero regression -----
        # cursor_source selects what drives the cursor: "nose" (default, the
        # original head cursor) or "gaze" (iris tracking; needs a calibration).
        # apply_motor_profile validates + activates it; the worker rebuilds the
        # FaceMesh in place when the source flips (iris landmarks cost extra).
        self.cursor_source = "nose"
        self._gaze = None                 # lazy GazeEstimator (built in the worker)
        self._gaze_cal = {}               # fit_calibration dict; empty = uncalibrated
        self.gaze_smoothing_alpha = 0.12
        self._needs_iris = False          # refine_landmarks only when gaze selected
        self._graph_dirty = False         # worker rebuilds FaceMesh (no restart)

    # ----- Delegates into the mouth-gesture recognizer -----------------
    # External readers (drag hand-off, teardown paths, tests) keep using
    # the original attribute names on VisionMouse.
    @property
    def is_dragging(self) -> bool:
        return self._face.is_dragging

    @is_dragging.setter
    def is_dragging(self, value) -> None:
        self._face.is_dragging = bool(value)

    @property
    def mouth_open_start(self):
        return self._face.mouth_open_start

    @property
    def mouth_open_threshold(self):
        return self._face.mouth_open_threshold

    @mouth_open_threshold.setter
    def mouth_open_threshold(self, value) -> None:
        self._face.mouth_open_threshold = value

    def _record_movement(self, now, x, y):
        """Buffer one raw sample, and flush a session when the timer comes round.

        Wrapped whole: a health record is a nice-to-have and the cursor is not.
        For many of these users it is their only way to operate the computer, so
        nothing in this path may ever propagate.
        """
        try:
            if not self._health_started:
                self._health_started = now
                self._health_last_flush = now
            self._health_samples.append((now, float(x), float(y)))
            # Posture rides the same buffer timer and the same opt-in. getattr
            # because this method is wrapped whole: a missing buffer must cost
            # the posture figures, never the movement record behind them.
            pose = self._last_pose
            poses = getattr(self, "_pose_samples", None)
            # The opt-in is checked HERE and not only at the call site. Today
            # this method is reached only while recording, so the flag is
            # redundant -- and a redundant check is the right kind: it makes
            # the opt-in a property of the code that collects, not of where it
            # happens to be called from. An earlier draft of the struggle
            # counter accumulated by default and checked at flush time, which
            # would have collected from somebody who never switched it on.
            if poses is not None and pose and self.health_monitoring_enabled:
                try:
                    poses.append((now,
                                  float(pose.get("nose_x", 0.0)),
                                  float(pose.get("nose_y", 0.0)),
                                  float(pose.get("roll_deg", 0.0)),
                                  float(pose.get("span", 0.0))))
                except (TypeError, ValueError):
                    pass
            # getattr, not attribute access. This whole method is wrapped, so
            # one missing attribute does not raise — it silently stops the
            # person's record from ever being written, which is the failure
            # mode hardest to notice and easiest to introduce.
            due = (HEALTH_FIRST_FLUSH_SECONDS
                   if not getattr(self, "_health_flushes", 0)
                   else HEALTH_FLUSH_SECONDS)
            if now - self._health_last_flush >= due:
                self._flush_movement(now)
        except Exception:
            pass

    def _pinned_params(self, profile):
        """Motor settings sitting at the edge of their allowed range.

        A parameter at its ceiling is the single most actionable thing in the
        technical report: live adaptation has nothing left to give, so continued
        difficulty cannot be tuned away and needs a product change instead.
        """
        pinned = []
        try:
            bounds = getattr(profile, "bounds", None) or {}
            for name, span in bounds.items():
                try:
                    lo, hi = float(span[0]), float(span[1])
                    value = float(getattr(profile, name))
                except (TypeError, ValueError, IndexError, AttributeError):
                    continue
                edge = abs(hi - lo) * 0.001
                if value >= hi - edge:
                    pinned.append(name)
                elif value <= lo + edge:
                    pinned.append(name + ":floor")
        except Exception:
            return []
        return pinned[:12]

    def _sync_evidence_recorder(self):
        """Build, rebuild or tear down the recorder to match the current flags."""
        try:
            want = (self.health_monitoring_enabled
                    and self.health_evidence_enabled)
            if not want:
                if self._evidence is not None:
                    self._evidence.stop()
                    self._evidence = None
                return
            from mistey.health import evidence
            if (self._evidence is None
                    or self._evidence.user_id != self.health_user_id
                    or self._evidence.photos != bool(self.health_photos_enabled)):
                if self._evidence is not None:
                    self._evidence.stop()
                self._evidence = evidence.Recorder(
                    self.health_user_id, enabled=True,
                    photos=self.health_photos_enabled)
                # Recorder re-reads the consent file, so this may come back
                # False even though the profile asked for photographs. Mirror
                # its decision rather than our request — otherwise the hot path
                # would keep copying frames the recorder throws away.
                self.health_photos_enabled = self._evidence.photos
        except Exception:
            self._evidence = None

    def _note_struggle(self, kind, weight=1):
        """One dict increment, wrapped. Called from the frame loop.

        The health record is opt-in; the adaptation signal is not, because it
        changes how the cursor behaves for the person right now rather than
        keeping anything about them. So the two are reported separately and
        only the first is gated.
        """
        try:
            if self.health_monitoring_enabled:
                self._struggle.note(kind, weight)
        except Exception:
            pass
        try:
            callback = self.signal_callback
            if callback is not None:
                callback(kind)
        except Exception:
            pass

    def _trail(self):
        """The last few seconds of pointer path, as screen fractions.

        Reuses the movement buffer rather than keeping a second one — those
        samples already exist, are already raw (pre-smoothing, pre-dead-zone),
        and normalising to fractions means nothing about the display resolution
        travels with an exported report.
        """
        try:
            from mistey.health.evidence import TRAIL_POINTS
            w = float(self.screen_w or 1) or 1.0
            h = float(self.screen_h or 1) or 1.0
            return [(x / w, y / h)
                    for _, x, y in list(self._health_samples)[-TRAIL_POINTS:]]
        except Exception:
            return []

    def _capture_moment(self, kind, frame=None):
        """Offer this moment to the evidence recorder, if there is one.

        The recorder owns the debounce and the disk budget; this only supplies
        what it cannot see from here. Wrapped whole for the usual reason: the
        cursor outranks the record of it.
        """
        try:
            rec = self._evidence
            if rec is None or not self.health_monitoring_enabled:
                return
            rec.note(kind, time.time(), self._last_pose, self._trail(),
                     frame if self.health_photos_enabled else None,
                     {"dead_zone_radius": self.dead_zone_radius,
                      "dwell_time_s": self.dwell_time})
        except Exception:
            pass

    def _struggle_context(self):
        """The settings that were in force, so a count can be acted on.

        "1,400 movements too small to register" is a complaint. The same number
        beside "dead_zone_radius 220, already at its ceiling" is a decision:
        this person has run out of tuning and needs different software.
        """
        # Read field by field. The first version built one dict literal, so a
        # single wrong attribute name raised and the whole context came back
        # empty — silently, because the caller swallows. The context is the part
        # that makes a count actionable, so losing all of it to one typo was the
        # worst possible failure here, and it happened: `smoothing_alpha` and
        # `sensitivity` are called `alpha` and `sensitivity_factor` on the
        # driver, and every struggle record shipped with `"context": {}`.
        fields = (("dead_zone_radius", "dead_zone_radius"),
                  ("smoothing_alpha", "alpha"),
                  ("sensitivity", "sensitivity_factor"),
                  ("diagonal_sensitivity", "diagonal_sensitivity_factor"),
                  ("dwell_time_s", "dwell_time"),
                  ("dwell_radius_px", "dwell_radius"),
                  ("dwell_click_enabled", "dwell_click_enabled"),
                  ("mouth_click_enabled", "mouth_click_enabled"),
                  ("mouth_open_threshold", "mouth_open_threshold"),
                  ("cursor_source", "cursor_source"),
                  ("distance_compensation", "distance_compensation"))
        out = {}
        for name, attr in fields:
            try:
                value = getattr(self, attr)
            except Exception:
                continue
            if value is not None:
                out[name] = value
        try:
            out["at_ceiling"] = list(self._at_ceiling)
        except Exception:
            pass
        return out

    def _flush_struggle(self, started, now):
        """Write the window's counts beside the movement record."""
        try:
            counts = self._struggle.drain()
            if counts:
                struggle_log.record_window(
                    self.health_user_id, started, now, counts,
                    self._struggle_context(), modality="head")
            if self._evidence is not None:
                self._evidence.flush()
        except Exception:
            _phase("struggle record skipped (cursor unaffected)")

    def _flush_movement(self, now=None):
        """Summarise the buffer into one stored session and start the next."""
        try:
            now = now or time.time()
            samples = list(self._health_samples)
            self._health_samples.clear()
            # getattr, like _health_flushes below and for the same reason: this
            # method is wrapped so it can never hurt the cursor, which means an
            # AttributeError here does not raise -- it quietly stops recording
            # for the rest of the session. test_a_missing_internal_counter_
            # cannot_silence_the_record pins it, having caught this a third time.
            marks = getattr(self, "_click_marks", None)
            clicks = list(marks) if marks is not None else []
            if marks is not None:
                marks.clear()
            poses = getattr(self, "_pose_samples", None)
            pose_rows = list(poses) if poses is not None else []
            if poses is not None:
                poses.clear()
            started, self._health_last_flush = self._health_started, now
            self._health_started = now
            # The struggle counts are flushed on their own terms: a window can
            # be full of blocked movement and still hold too few samples to
            # measure tremor from, and that window is exactly the interesting
            # one. Tying it to the movement threshold would drop it.
            self._health_flushes = getattr(self, "_health_flushes", 0) + 1
            if started:
                self._flush_struggle(started, now)
            if len(samples) < HEALTH_MIN_SAMPLES or not started:
                return
            if (now - started) < HEALTH_MIN_SECONDS:
                # Enough samples but not enough time, which only happens on a
                # fast loop. Measuring a 3 Hz tremor from four seconds of
                # pointer movement is barely one period.
                return
            from mistey.health import motor_metrics, session_store
            # getattr-with-default, not self.dwell_radius: this whole method is
            # wrapped in a bare except, so a missing attribute here does not
            # raise -- it silently switches the movement record off and says
            # "measurement failed" once. Nothing on this path may depend on an
            # attribute being present.
            radius = float(getattr(self, "dwell_radius", 40) or 40)
            metrics = motor_metrics.flat_metrics(motor_metrics.summarise(
                samples, clicks=clicks, radius=radius, poses=pose_rows))
            if not metrics:
                return
            session_store.record_session(self.health_user_id, started, now,
                                         metrics, modality="head")
        except Exception:
            _phase("movement record skipped (measurement failed, cursor unaffected)")

    def _note_face_distance(self, span) -> None:
        """Say so when the face is too small to track, once, not every frame.

        Someone whose cursor has stopped responding has no way to know the
        camera simply cannot see them well enough — and if head control is
        their only input, they cannot go and read a log about it either. The
        guide rail already carries head-state hints, so it goes there.
        """
        too_far = span < config.FACE_SPAN_TOO_FAR
        now = time.time()
        if not too_far:
            if self._too_far_since:
                self._too_far_since = 0.0
                self._emit_gesture("face_ok")
            return
        if self._too_far_since == 0.0:
            self._too_far_since = now
            return
        # Sustained, not a one-frame glitch: say it, then stay quiet for a while.
        if now - self._too_far_since > 2.0:
            self._too_far_since = now + 15.0     # re-arm, don't nag
            _phase(f"face is small in frame (span {span:.3f}) - user may be too "
                   f"far from the camera for reliable tracking")
            self._note_struggle("too_far")
            self._capture_moment("too_far")
            self._emit_gesture("too_far")

    def _too_fast(self, jump) -> bool:
        """Is this one frame's movement far beyond what this person ever does?

        The threshold is their own 95th-percentile frame-to-frame movement
        times a multiplier, with a floor until there is enough history to take
        a percentile of. An absolute pixels-per-frame ceiling would be a
        different setting on every screen, at every gain, and at every frame
        rate -- and the rate alone now varies between about 10 Hz and 29 Hz
        depending on which filter the profile selected.

        Returns False on anything unexpected: the only consequence of saying
        "not too fast" is that the pointer moves, which is what it is for.
        """
        try:
            floor = FAST_JUMP_FLOOR_FRACTION * float(self.screen_w or 1920)
            ceiling = floor
            if len(self._jump_samples) >= FAST_JUMP_MIN_HISTORY:
                usual = sorted(self._jump_samples)[
                    int(len(self._jump_samples) * 0.95)]
                # max(), so the person's own movement can only ever RAISE the
                # ceiling. Somebody who moves a lot is not clamped for moving
                # the way they move; somebody who barely moves is protected by
                # the floor rather than measured down to it.
                ceiling = max(floor, usual * FAST_JUMP_OVER_USUAL)
            return float(jump) > ceiling
        except Exception:                      # noqa: BLE001
            return False

    def _announce_recentre(self) -> None:
        """Say that the middle has moved, out loud and on screen.

        Out loud matters more than usual here. Somebody re-centres because the
        cursor has drifted away from where their head rests, and at that moment
        they are quite likely looking away from the screen or down -- which is
        how the posture slipped in the first place. A confirmation they have to
        read is a confirmation they may not get.

        Wrapped whole and silent on failure: this runs on the thread driving
        somebody's only pointer, and a missing speech service is not a reason
        for the cursor to stop.
        """
        try:
            _phase("head cursor re-centred (%s)" % (self._recentre_source or "?"))
        except Exception:                      # noqa: BLE001
            pass
        try:
            self._emit_gesture("recentred")
        except Exception:                      # noqa: BLE001
            pass
        say = self.say_callback
        if say is not None:
            try:
                say("Re-centred. Where your head is now is the middle.")
            except Exception:                  # noqa: BLE001
                pass

    def _emit_gesture(self, state) -> None:
        cb = self.gesture_callback
        if cb is None:
            return
        try:
            cb(state, 0.0)
        except Exception:
            pass

    def apply_motor_profile(self, profile) -> None:
        """Apply an AbilityProfile's motor params live (no loop restart needed).

        The loop reads these instance attributes every frame, so updates take
        effect immediately — enabling calibration and live adaptation.
        """
        # Read defensively: an older saved profile predates this field, and an
        # absent flag must mean "not recording", never "record by default".
        was_on = self.health_monitoring_enabled
        self.health_monitoring_enabled = bool(
            getattr(profile, "health_monitoring_enabled", False))
        self.health_evidence_enabled = bool(
            getattr(profile, "health_evidence_enabled", False))
        # Photographs are gated twice on purpose. The profile flag is what the
        # person switched on; evidence.Recorder re-reads the stored consent and
        # will refuse regardless of the flag. A stale True in a saved profile
        # must never be able to resurrect photography after consent was
        # withdrawn — the file on disk is the authority, not the setting.
        self.health_photos_enabled = bool(
            getattr(profile, "health_photos_enabled", False))
        self._at_ceiling = self._pinned_params(profile)
        self._sync_evidence_recorder()
        if was_on and not self.health_monitoring_enabled:
            # Turned off mid-session: keep what was already measured rather than
            # discarding it, then stop. Switching it off is not a request to
            # erase — purge() is, and it is a separate, explicit act.
            self._flush_movement()
        # Keep whatever id was already set if the profile does not carry one.
        # AbilityProfile has no user_id field, so the old `or "default"` reset
        # this on every profile apply — including immediately after AbilityEngine
        # had set it to the real id. The driver then wrote `default.health.jsonl`
        # while the report read `default_user.*`, so a person could record for
        # hours and be told nothing had been recorded.
        self.health_user_id = str(getattr(profile, "user_id", "")
                                  or self.health_user_id or "default_user")

        self.alpha = float(profile.smoothing_alpha)
        self.sensitivity_factor = float(profile.sensitivity)
        self.diagonal_sensitivity_factor = float(profile.diagonal_sensitivity)
        self.dead_zone_radius = int(profile.dead_zone_radius)
        self.mouth_open_threshold = int(profile.mouth_open_threshold)
        self.max_face_not_visible_frames = int(profile.max_face_not_visible_frames)
        self.mouth_click_enabled = bool(profile.mouth_click_enabled)
        self.dwell_click_enabled = bool(profile.dwell_click_enabled)
        self.dwell_time = float(profile.dwell_time_s)
        self.dwell_radius = int(profile.dwell_radius_px)

        # ----- Which smoothing (Wave-4) -----
        # getattr-with-default like everything else here, so a profile saved
        # before this field existed loads as "ema" — which is what it had.
        # Anything not in the whitelist becomes "ema" too: a typo in a saved
        # profile must not be able to leave somebody without a pointer.
        # getattr-with-default: a profile saved before this existed loads with
        # the hold off, which is what it had.
        self.spasm_clamp_enabled = bool(
            getattr(profile, "spasm_clamp_enabled", False))

        mode = str(getattr(profile, "filter_mode", "ema") or "ema").lower()
        self.filter_mode = mode if mode in ("ema", "one_euro") else "ema"
        # The sample rate follows the filter. Raising it under the EMA would
        # divide a tremor user's smoothing by the same factor, and the
        # compensating alpha is below its own bound -- so the faster loop is
        # only offered to the filter that can take it.
        self._mp_interval = (config.FACE_MP_INTERVAL_ONE_EURO
                             if self.filter_mode == "one_euro"
                             else config.FACE_MP_INTERVAL)
        if self.filter_mode == "one_euro":
            if getattr(profile, "filter_tuning_auto", True):
                # Translate the preset's own smoothing_alpha rather than
                # inventing numbers, so a preset authored as "heavy smoothing
                # for a tremor" keeps the smoothing it was authored with and
                # only its dependence on the frame rate goes away.
                mincutoff = smoothing.ema_equivalent_cutoff(self.alpha)
                # Responsiveness while moving, scaled the way hand_mouse already
                # scales it: heavier smoothing (lower alpha) means a steadier
                # filter. NOTE the magnitude is not yet verified on the head
                # path — the filter runs BEFORE the centre-relative gain, so its
                # velocity estimate is in pre-gain pixels and `beta` does not
                # mean here what it means on the hand. That is what the live
                # preview in the tuning panel is for.
                beta = max(0.003, 0.03 * self.alpha)
            else:
                mincutoff = float(getattr(profile, "filter_mincutoff", 0.37))
                beta = float(getattr(profile, "filter_beta", 0.006))
            if self._one_euro is None:
                self._one_euro = smoothing.OneEuro2D(
                    mincutoff=mincutoff, beta=beta)
            else:
                # tune() rather than a rebuild: a clinician moving a slider
                # while the person is pointing must not drop the filter's state
                # and jump the cursor.
                self._one_euro.tune(mincutoff=mincutoff, beta=beta)
            print("[VISION] filter -> one_euro (mincutoff=%.3f beta=%.4f, "
                  "from alpha=%.2f)" % (mincutoff, beta, self.alpha))

        # ----- Optional face-gesture events (Wave-2) -----
        # Every read is getattr-with-default so profiles saved before these
        # fields existed keep working — and the defaults are all OFF.
        # Swipe mode. Read defensively like everything else here: a profile
        # saved before this existed must load with it off, never on.
        self.swipe_gestures_enabled = bool(
            getattr(profile, "swipe_gestures_enabled", False))
        self.swipe_shapes_enabled = bool(
            getattr(profile, "swipe_shapes_enabled", False))
        try:
            self.swipe_min_distance_px = int(
                getattr(profile, "swipe_min_distance_px", 120))
            self.swipe_directions = int(getattr(profile, "swipe_directions", 4))
            self.swipe_shape_floor = float(
                getattr(profile, "swipe_shape_floor", 0.80))
        except (TypeError, ValueError):
            pass
        from mistey.input import gesture_actions as _ga
        self._swipe_map = _ga.validate_swipe_map(
            getattr(profile, "swipe_map", None))
        if self._face.swipe_mode != self.swipe_gestures_enabled:
            # Changing mode mid-stroke would leave the pointer frozen.
            self._end_swipe()
        self._face.swipe_mode = self.swipe_gestures_enabled

        self._face_events_on = bool(getattr(profile, "face_gestures_enabled", False))
        self._face.events_enabled = self._face_events_on
        self._face.set_event_params(FaceGestureParams(
            blink_enabled=bool(getattr(profile, "blink_click_enabled", False)),
            wink_enabled=bool(getattr(profile, "wink_gestures_enabled", False)),
            brow_enabled=bool(getattr(profile, "brow_gesture_enabled", False)),
            tilt_enabled=bool(getattr(profile, "tilt_gestures_enabled", False)),
            blink_ear_threshold=float(getattr(profile, "blink_ear_threshold", 0.21)),
            blink_min_hold_s=float(getattr(profile, "blink_min_hold_s", 0.35)),
            double_blink_window_s=float(getattr(profile, "double_blink_window_s", 0.7)),
            brow_raise_ratio=float(getattr(profile, "brow_raise_ratio", 1.25)),
            tilt_angle_deg=float(getattr(profile, "tilt_angle_deg", 18.0)),
            tilt_hold_s=float(getattr(profile, "tilt_hold_s", 0.5)),
        ))
        self._face_map = gesture_actions.validate_map(
            getattr(profile, "face_gesture_map", None))

        # ----- Eye-gaze cursor (Wave-2) -----
        # getattr-with-default so profiles saved before these fields existed
        # keep working; the whitelist rejects any bad saved string.
        src = str(getattr(profile, "cursor_source", "nose") or "nose").lower()
        src = src if src in ("nose", "gaze") else "nose"
        cal = dict(getattr(profile, "gaze_calibration", {}) or {})
        # gaze without a calibration falls back to nose (never a dead cursor)
        effective = "gaze" if (src == "gaze" and cal) else "nose"
        if effective != self.cursor_source:
            self._graph_dirty = True      # worker rebuilds FaceMesh in place
        self.cursor_source = effective
        self._gaze_cal = cal
        self.gaze_smoothing_alpha = float(getattr(profile, "gaze_smoothing_alpha", 0.12))
        # Iris landmarks (refine_landmarks=True) only when gaze drives the
        # cursor — everyone else keeps the cheaper 468-point mesh.
        self._needs_iris = (self.cursor_source == "gaze")
        if effective == "gaze":
            print("[VISION] cursor source: GAZE (eye tracking, calibrated)")
        elif src == "gaze":
            print("[VISION] gaze requested but uncalibrated — staying on nose cursor")

        if self._face_events_on:
            print("[VISION] face gestures ON "
                  f"(blink={'on' if getattr(profile, 'blink_click_enabled', False) else 'off'}, "
                  f"wink={'on' if getattr(profile, 'wink_gestures_enabled', False) else 'off'}, "
                  f"brow={'on' if getattr(profile, 'brow_gesture_enabled', False) else 'off'}, "
                  f"tilt={'on' if getattr(profile, 'tilt_gestures_enabled', False) else 'off'})")

        print(f"[VISION] motor profile applied (alpha={self.alpha}, "
              f"sens={self.sensitivity_factor}, dead_zone={self.dead_zone_radius}, "
              f"dwell={'on' if self.dwell_click_enabled else 'off'}, "
              f"mouth={'on' if self.mouth_click_enabled else 'off'})")

    # Actions this driver performs itself, rather than by driving the OS.
    #
    # gesture_actions.ACTIONS entries are all fn(pyautogui) -- a pure OS
    # side-effect. Re-centring has to reach into the mapping this object owns,
    # so it is intercepted here instead of widening every other action's
    # signature to pass a driver that nothing else needs.
    _DRIVER_ACTIONS = ("recentre",)

    def _do_action(self, action, pyautogui, label_cb=None):
        """Run one named action, whether it belongs to the OS or to us.

        One funnel for all three gesture channels -- swipes, a held tilt, and
        the discrete face events -- so re-centring became available to every
        one of them at once, and so the next driver-local action needs no new
        plumbing either.
        """
        if action in self._DRIVER_ACTIONS:
            if action == "recentre":
                self.request_recentre("gesture")
            label = gesture_actions.ACTIONS.get(action, (None, False, None))[2]
            cb = label_cb if label_cb is not None else self._notify_click
            if label and cb is not None:
                try:
                    cb(label)
                except Exception:          # noqa: BLE001
                    pass
            return label
        return gesture_actions.dispatch(
            action, pyautogui,
            label_cb if label_cb is not None else self._notify_click)

    def request_recentre(self, source="voice"):
        """Take wherever the head is now as the middle of the screen.

        Safe to call from any thread: it sets a flag and the cursor loop does
        the work on its next frame, in the coordinate space the mapping
        actually uses. Nothing here touches the mapping directly, because the
        caller is usually the voice thread or a hotkey hook.

        This exists because of a circular dependency. Posture slips during a
        long sitting, the mapped centre stops matching where the head rests,
        and the cursor drifts to an edge -- which is exactly the moment the
        pointer cannot be used to reach a button that would fix it. So every
        trigger has to work without the pointer: a face gesture the person
        chooses, a spoken command, or a hotkey somebody helping can press.
        """
        self._recentre_pending = True
        self._recentre_source = str(source or "")
        return True

    def _notify_click(self, click_type: str = "click"):
        # Every click from every source arrives here -- mouth, dwell, blink,
        # tilt, swipe -- so one hook covers them all, and a new click source
        # gets measured without anybody remembering to wire it.
        #
        # Two deque appends behind the opt-in. A "drag" is not an aim: the
        # pointer is about to carry something rather than having arrived, and
        # counting it as a reach would score the pick-up as the target.
        if (self.health_monitoring_enabled
                and click_type != "drag"
                and self._committed_x is not None):
            try:
                self._click_marks.append(
                    (time.time(), float(self._committed_x),
                     float(self._committed_y)))
            except Exception:          # noqa: BLE001 - never at the cursor's expense
                pass
        if self.click_callback:
            try:
                self.click_callback(click_type)
            except Exception as e:
                print(f"[VISION] Click callback error: {e}")

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------
    def start(self):
        if self.running:
            return

        if not self.guard.acquire_camera():
            print("[VISION] Camera busy")
            _phase("NOT started - camera lock already held in this process")
            return
        self._holds_camera = True

        # A webcam that is busy for a few seconds is the NORMAL case here, not a
        # fault: the guided dashboard is still exiting and Windows does not hand
        # the device over instantly. The default budget (3 tries, ~7s) expired
        # during that handoff and parked the cursor in ERROR *for the whole
        # session* — head mode worked all through setup and was then dead in the
        # cockpit, silently.
        #
        # For someone whose only input is head movement, a cursor that gives up
        # is the difference between a working computer and none. Keep trying for
        # about a minute instead.
        self._supervisor = SupervisedThread(
            name="VisionMouse",
            worker=self._vision_worker,
            cleanup=self._release_capture,
            max_restarts=15,
            base_backoff=0.7,
            max_backoff=5.0,
        )
        self._supervisor.start()
        print("[VISION] Started")
        _phase("head/face cursor starting")

    def stop(self):
        # Flush before tearing down: the last stretch of use is often the most
        # interesting one, since fatigue shows up at the end of a session.
        if self.health_monitoring_enabled:
            self._flush_movement()
        # getattr, not attribute access: stop() is a teardown path and is
        # reached after a failed __init__ too. A driver that cannot be torn
        # down leaks the camera, and then head control — someone's only input —
        # cannot be restarted without killing the process.
        recorder = getattr(self, "_evidence", None)
        if recorder is not None:
            recorder.stop()
            self._evidence = None
        if self._supervisor:
            self._supervisor.stop()
            self._supervisor = None
        self._release_capture()
        # Only release the camera lock if we actually hold it — otherwise we'd
        # free a lock another cursor driver (HandMouse) is using.
        if self._holds_camera:
            self.guard.release_camera()
            self._holds_camera = False
        print("[VISION] Stopped")

    @property
    def running(self) -> bool:
        return self._supervisor.running if self._supervisor else False

    def get_health(self):
        if self._supervisor:
            return self._supervisor.get_health()
        return {"status": "stopped"}

    def current_jitter(self) -> float:
        """Mean per-frame cursor movement magnitude — a proxy for tremor/jitter.

        Used by the live adaptation loop to detect rising involuntary movement
        (e.g. fatigue) relative to the user's calibrated baseline.
        """
        if not self._jitter_samples:
            return 0.0
        return sum(self._jitter_samples) / len(self._jitter_samples)

    def set_clicks_enabled(self, on: bool):
        """Master click gate — the Scroll Scope pauses clicking while the head is
        scrolling (moving toward screen edges would otherwise dwell/mouth-click
        whatever is under the cursor). Cursor movement is unaffected."""
        self.clicks_enabled = bool(on)
        if not on:
            self._release_drag()      # never leave a drag latched while gated
            self._end_swipe()         # nor the pointer frozen mid-stroke

    def set_cursor_enabled(self, on: bool):
        """Cursor-motion gate (Wave-2 switch/scanning): when False the worker
        keeps computing landmarks and face gestures as usual (they feed the
        scanning switch) but SKIPS the dead-zone commit / moveTo — the
        ScanningController drives the OS cursor instead. Dwell is skipped too
        (dwell is meaningless without cursor motion). Default True.
        Independent of set_clicks_enabled."""
        self.cursor_enabled = bool(on)

    def _fire_switch(self):
        """Deliver one switch activation to the scanning integrator.

        Guarded: a controller-side error must never kill the vision loop
        (mirrors _notify_click's contract)."""
        cb = self.switch_callback
        if cb is None:
            return
        try:
            cb()
        except Exception as e:
            print(f"[VISION] switch callback error: {e}")

    def _end_swipe(self):
        """Abandon any stroke in flight and unfreeze the pointer.

        Called from every teardown path there is. A frozen cursor that never
        thaws is the worst failure this feature can produce: for many of these
        users the pointer is the only way to reach the control that would turn
        the mode back off.
        """
        try:
            self._swiping = False
            self._swipe_points.clear()
            self._swipe_hint = ""
            if getattr(self, "_face", None) is not None:
                self._face._swiping = False
            cb = self.swipe_callback
            if cb is not None:
                cb([], "")
        except Exception:
            pass

    def _swipe_label(self, name, kind):
        """The word the overlay shows, taken from the action it will run.

        Sourced from the action table rather than written out again here, so
        the overlay can never promise something different from what fires.
        """
        try:
            if not name:
                return ""
            from mistey.input import gesture_actions as ga
            action = (self._swipe_map or ga.DEFAULT_SWIPE_MAP).get(name, "none")
            entry = ga.ACTIONS.get(action)
            if entry and entry[2]:
                return entry[2]
            return ""
        except Exception:
            return ""

    def _swipe_track(self, now, x, y):
        """One point, plus a throttled re-read so the overlay can name it."""
        try:
            self._swipe_points.append((now, float(x), float(y)))
            if now - self._swipe_hint_ts < 0.15:
                return
            self._swipe_hint_ts = now
            from mistey.input import swipe as swipe_mod
            name, _conf, kind = swipe_mod.classify(
                list(self._swipe_points),
                min_distance=float(self.swipe_min_distance_px),
                directions=self.swipe_directions,
                shapes_enabled=self.swipe_shapes_enabled,
                shape_floor=self.swipe_shape_floor)
            self._swipe_hint = self._swipe_label(name, kind)
            cb = self.swipe_callback
            if cb is not None:
                w = float(self.screen_w or 1) or 1.0
                h = float(self.screen_h or 1) or 1.0
                cb([(px / w, py / h) for _, px, py in self._swipe_points],
                   self._swipe_hint)
        except Exception:
            pass

    def _fire_swipe(self, pyautogui):
        """Classify the finished stroke and run whatever it named."""
        try:
            from mistey.input import gesture_actions as ga
            from mistey.input import swipe as swipe_mod
            points = list(self._swipe_points)
            name, _conf, kind = swipe_mod.classify(
                points,
                min_distance=float(self.swipe_min_distance_px),
                directions=self.swipe_directions,
                shapes_enabled=self.swipe_shapes_enabled,
                shape_floor=self.swipe_shape_floor)
            self._end_swipe()
            if name is None:
                # Never silent. A gesture that does nothing and says nothing is
                # indistinguishable from a broken feature.
                self._note_struggle("swipe_unclear")
                self._emit_gesture("swipe_unclear")
                return
            if self.swipe_shapes_enabled and kind != "shape":
                # They had shapes on and got a direction: the stroke was not
                # complete enough. A rising rate here is direct evidence the
                # hold is too long for this person.
                self._note_struggle("swipe_shape_fell_back")
            action = (self._swipe_map or ga.DEFAULT_SWIPE_MAP).get(name, "none")
            self._do_action(action, pyautogui)
        except Exception:
            self._end_swipe()

    def _release_drag(self):
        """Release a held left-button drag — prevents a stuck button / desktop hang.

        Called whenever the camera releases (stop, restart, crash) and when the
        face is lost, so a drag in progress can never leave the mouse button down.
        """
        if self._face.is_dragging:
            try:
                import pyautogui
                pyautogui.mouseUp(button="left")
            except Exception:
                pass
        # Clears the drag latch + mouth timing (never last_click_time,
        # matching the original behaviour).
        self._face.reset()

    def _release_capture(self):
        """Release the camera handle + any preview window between runs/restarts.

        The ResourceGuard camera lock is held across restarts and only released
        in stop(); this just frees the cv2 device so reopening can succeed.
        """
        # Never leave a drag button held when the camera goes away.
        self._release_drag()
        self._end_swipe()
        # Close the MediaPipe graph on every teardown (crash/restart/stop) — leaking
        # it holds native buffers and blocks the next FaceMesh from initialising.
        try:
            if getattr(self, "_face_mesh", None) is not None:
                self._face_mesh.close()
        except Exception:
            pass
        finally:
            self._face_mesh = None
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        finally:
            self._cap = None
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ---------------------------------------------------------
    # Core Vision Loop (runs inside SupervisedThread)
    # ---------------------------------------------------------
    def _vision_worker(self, should_run):
        import cv2
        import mediapipe as mp
        import pyautogui

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        # Reset smoothing / cursor / dwell state on each (re)start.
        self.prev_x = None
        self.prev_y = None
        self._last_target_x = None
        self._last_target_y = None
        self._jump_hold_frames = 0
        if self._one_euro is not None:
            self._one_euro.reset()
        self._committed_x = None
        self._committed_y = None
        self._dwell_anchor = None
        self._dwell_fired = False
        self._gaze = None                  # gaze EMA rebuilt lazily (fresh state)
        self._switch_mouth_prev = False    # mouth-switch edge state (fresh run)

        # Throttle MediaPipe
        last_mp_time = 0
        # Read from the instance each iteration, not captured once: the
        # profile can change mid-session and the rate follows the filter.
        # (See config.FACE_MP_INTERVAL for why the two are tied.)

        mp_face = mp.solutions.face_mesh

        def build_face_mesh():
            # refine_landmarks only when the eye-gaze cursor is selected. The
            # default nose cursor reads just the nose (2) and inner lips (13/14),
            # all in the base 468-point mesh — and iris refinement roughly
            # doubles FaceMesh CPU cost, so it stays OFF for everyone else
            # (preserving the 4 GB low-end budget; a big saving there).
            self._needs_iris = (self.cursor_source == "gaze")
            return mp_face.FaceMesh(
                max_num_faces=1,
                refine_landmarks=self._needs_iris,
                # A lower tracking threshold keeps MediaPipe in cheap *tracking* mode
                # instead of dropping the face and re-running full *detection* every
                # frame (that re-detection churn is what makes the cursor freeze-then-
                # jump and feel laggy, especially in low light).
                # Lowered for small faces. Someone sitting back scores lower
                # on both, and at 0.5/0.4 MediaPipe kept dropping them and
                # re-running full detection every frame - the freeze-then-jump
                # the comment above describes, at its worst exactly when the
                # person can least afford it.
                min_detection_confidence=0.4,
                min_tracking_confidence=0.3,
            )

        face_mesh = build_face_mesh()
        # The build above already reflects the current cursor_source, so any
        # dirty flag raised while the driver was stopped is stale — clear it.
        self._graph_dirty = False

        if sys.platform == "win32":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
        self._cap = cap

        if not cap.isOpened():
            # Raise so the supervisor records the error and retries (the camera
            # may come back); after max retries it parks in ERROR, visibly.
            _phase("camera busy - will retry (another app or the setup window "
                   "may still hold it)")
            raise RuntimeError("Camera not accessible")

        # Size comes from config: the pixels ACROSS THE FACE decide landmark
        # quality, and someone sitting back has very few of them. Left
        # unconstrained a webcam may hand us 1080p, where FaceMesh costs far
        # more for no extra detail on the face itself.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        _phase("capture %dx%d requested, got %dx%d"
               % (config.CAMERA_WIDTH, config.CAMERA_HEIGHT,
                  cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        print("[VISION] Camera opened")
        _phase("camera opened - head cursor live")
        self._face_mesh = face_mesh      # so _release_capture can close it on any exit

        self.face_not_visible_counter = 0

        while should_run():
            # ── Live cursor-source switch (nose ↔ gaze) ──
            # apply_motor_profile marks the graph dirty when the source flips;
            # rebuild FaceMesh in place with the right refine setting — no
            # camera reopen, no thread restart, cursor pauses for one build.
            if self._graph_dirty:
                self._graph_dirty = False
                try:
                    face_mesh.close()
                except Exception:
                    pass
                face_mesh = build_face_mesh()
                self._face_mesh = face_mesh
                self._gaze = None            # rebuilt lazily with fresh EMA state
                # Reset smoothing/commit state so the switch can never lurch the
                # cursor from one mapping's coordinates into the other's.
                self.prev_x = None
                self.prev_y = None
                if self._one_euro is not None:
                    self._one_euro.reset()
                # The re-centre origin is in nose-mapped screen coordinates, so
                # it means nothing to the gaze mapping. Cleared here and
                # deliberately NOT cleared on a supervised restart: a camera
                # hiccup does not change where somebody's head rests, and
                # throwing their re-centre away would undo the fix at the only
                # moment they needed it.
                self._origin_x = None
                self._origin_y = None
                self._last_target_x = None
                self._last_target_y = None
                self._jump_hold_frames = 0
                self._committed_x = None
                self._committed_y = None
                self._dwell_anchor = None

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            now = time.time()
            if now - last_mp_time >= self._mp_interval:
                # Low-light boost (user-tunable via the cockpit's Camera panel).
                from mistey.input import camera_tuning
                proc = camera_tuning.maybe_boost(rgb, cv2)
                # Read-only lets MediaPipe skip an internal copy (documented speedup).
                proc.flags.writeable = False
                result = face_mesh.process(proc)
                last_mp_time = now
            else:
                result = None

            faces = result.multi_face_landmarks if result else None

            if faces:
                self.face_not_visible_counter = 0
                landmarks = faces[0].landmark

                # ── Cursor target: nose (default) or eye-gaze ──
                nose = landmarks[2]

                # Posture snapshot for the movement record. Three scalars, taken
                # only while recording is on: where the head is in frame, how far
                # it is tilted, how large it appears. Deliberately NOT the face
                # mesh — 468 landmarks are biometric, and a report is a file
                # people are meant to be able to hand to a clinic.
                if self.health_monitoring_enabled:
                    try:
                        self._last_pose = {
                            "nose_x": nose.x, "nose_y": nose.y,
                            "roll_deg": head_roll(landmarks),
                            "span": self._span_ema if self._span_ema
                            else face_span(landmarks),
                            "jitter": self.current_jitter(),
                        }
                    except Exception:
                        pass
                if self.cursor_source == "gaze":
                    # Eye-gaze (Wave-2): iris-in-eye-box raw gaze → EMA →
                    # calibrated screen mapping. A None raw (blink / graph not
                    # yet refined) yields no target this frame — the cursor
                    # simply holds its last committed position.
                    if self._gaze is None:
                        from mistey.input.gaze import GazeEstimator
                        self._gaze = GazeEstimator()
                    target_x = target_y = None
                    r = self._gaze.raw(landmarks)
                    if r is not None:
                        sm = self._gaze.smoothed(r, self.gaze_smoothing_alpha)
                        target_x, target_y = self._gaze.to_screen(sm, self._gaze_cal)
                else:
                    # ── Nose → cursor ── (the default path — unchanged)
                    screen_x = nose.x * self.screen_w
                    screen_y = nose.y * self.screen_h

                    # Movement record: sample HERE, before the EMA and before the
                    # dead-zone commit below. Both of those exist precisely to
                    # suppress tremor, so measuring after them would measure our
                    # own filters doing their job rather than the person's hand.
                    # A parkinsons profile with a 150 px dead-zone would report a
                    # beautifully steady user who is in fact shaking.
                    if self.health_monitoring_enabled:
                        self._record_movement(now, screen_x, screen_y)

                    # The stroke is sampled from the SAME raw stream the
                    # movement record uses: before the EMA and before the
                    # dead-zone. Both of those exist to suppress tremor, and a
                    # gesture read after them would be a gesture the filters
                    # had already flattened.
                    if self._swiping:
                        self._swipe_track(now, screen_x, screen_y)

                    # One branch or the other, never both. The EMA arm is
                    # untouched: with filter_mode "ema" — the default, and what
                    # every saved profile has — this block computes exactly what
                    # it computed before, to the bit.
                    if self.filter_mode == "one_euro" and self._one_euro is not None:
                        screen_x, screen_y = self._one_euro(screen_x, screen_y, now)
                    elif self.prev_x is not None and self.prev_y is not None:
                        screen_x = self.prev_x + self.alpha * (screen_x - self.prev_x)
                        screen_y = self.prev_y + self.alpha * (screen_y - self.prev_y)

                    # Kept current in both arms, so switching filter mid-session
                    # cannot lurch the cursor from one filter's idea of where it
                    # was into the other's.
                    self.prev_x = screen_x
                    self.prev_y = screen_y

                    cx = self.screen_w / 2
                    cy = self.screen_h / 2

                    # A re-centre requested from anywhere -- a face gesture, a
                    # spoken command, a hotkey -- lands here, on the cursor
                    # thread, in the coordinate space the mapping uses. The
                    # trigger only ever sets a flag, so none of those callers
                    # touches the mapping from another thread.
                    if self._recentre_pending:
                        self._recentre_pending = False
                        self._origin_x, self._origin_y = screen_x, screen_y
                        # Without this the first frame after re-centring is a
                        # large jump: the filter still holds the old position
                        # and the committed point is where the cursor was. Any
                        # future clamp on sudden movement would then reject the
                        # very correction the person just asked for.
                        if self._one_euro is not None:
                            self._one_euro.reset()
                        self._committed_x = None
                        self._committed_y = None
                        self._dwell_anchor = None
                        # The clamp compares against the previous frame's
                        # target, which is now meaningless -- the whole mapping
                        # just moved. Leaving it would make the first frame
                        # after a re-centre look like a spasm.
                        self._last_target_x = None
                        self._last_target_y = None
                        self._jump_hold_frames = 0
                        self._announce_recentre()

                    # The head's resting place is subtracted, and the cursor is
                    # still PLACED at the middle -- so re-centring moves where
                    # "middle" is for the head without moving the screen.
                    ox = cx if self._origin_x is None else self._origin_x
                    oy = cy if self._origin_y is None else self._origin_y
                    dx = screen_x - ox
                    dy = screen_y - oy

                    if abs(dx) > abs(dy):
                        gain = self.sensitivity_factor
                    else:
                        gain = self.diagonal_sensitivity_factor

                    # Scale by how far away the person is sitting. Without this
                    # the cursor's whole reachable range shrinks with distance,
                    # and a preset dead-zone (parkinsons uses 150 px) can exceed
                    # that range entirely — the cursor then never moves at all,
                    # which reads as "head control is broken" rather than "you
                    # are sitting too far back".
                    if self.distance_compensation:
                        span = face_span(landmarks)
                        if span > 0:
                            self._span_ema = (
                                span if self._span_ema is None
                                else self._span_ema
                                + config.FACE_SPAN_EMA * (span - self._span_ema))
                            gain *= distance_scale(self._span_ema)
                            self._note_face_distance(self._span_ema)

                    new_x = cx + dx * gain
                    new_y = cy + dy * gain

                    SAFE = 10
                    target_x = max(SAFE, min(self.screen_w - SAFE, int(new_x)))
                    target_y = max(SAFE, min(self.screen_h - SAFE, int(new_y)))

                # ── Dead-zone ── (shared commit block — both cursor sources)
                # Ignore movement smaller than the radius so tremor / involuntary
                # motion does not jitter the cursor. radius 0 (Default) moves every
                # frame, exactly as before; presets raise it to steady the cursor.
                # (target is None only on a gaze no-reading frame: skip movement.)
                # cursor_enabled False (switch/scanning owns the cursor) skips
                # the whole commit — the head can never fight the scan cursor.
                if target_x is not None and self.cursor_enabled and not self._swiping:
                    if self._committed_x is None:
                        self._committed_x, self._committed_y = target_x, target_y
                    # ── Velocity clamp ── the dead-zone's twin, at the other
                    # end of the same quantity. Measured PER FRAME, from the
                    # previous target and deliberately NOT from the committed
                    # point: while the dead-zone holds the cursor still and the
                    # head drifts, the distance from the committed point grows
                    # legitimately, and a clamp reading that would fire on
                    # somebody slowly looking across their screen.
                    if self._last_target_x is not None:
                        jump = math.hypot(target_x - self._last_target_x,
                                          target_y - self._last_target_y)
                        self._jump_samples.append(jump)
                        if self.spasm_clamp_enabled and self._too_fast(jump):
                            self._jump_hold_frames += 1
                            if self._jump_hold_frames <= FAST_JUMP_MAX_HOLD_FRAMES:
                                # Hold the pointer where it was. Counted, so
                                # the movement record shows how often this
                                # person's movement is being held back -- which
                                # is the signal that the ceiling is too low for
                                # them, and invisible from anywhere else.
                                self._note_struggle("fast_jump_held")
                                target_x = self._last_target_x
                                target_y = self._last_target_y
                            else:
                                # Three frames of the same large movement is
                                # not a spasm, it is somebody who moved. Let it
                                # through rather than freeze their pointer.
                                self._jump_hold_frames = 0
                        else:
                            self._jump_hold_frames = 0
                    self._last_target_x, self._last_target_y = target_x, target_y

                    delta = math.hypot(target_x - self._committed_x,
                                       target_y - self._committed_y)
                    self._jitter_samples.append(delta)
                    if delta >= self.dead_zone_radius:
                        self._committed_x, self._committed_y = target_x, target_y
                        pyautogui.moveTo(target_x, target_y)
                        self._dz_streak = 0
                    elif (self.dead_zone_radius > 0
                          and delta >= self.dead_zone_radius
                          * DEADZONE_INTENT_FRACTION):
                        # Real movement, thrown away by the steadiness filter.
                        # This is the measurement that tells us the filter has
                        # stopped protecting the person and started blocking
                        # them — invisible from anywhere else, because from the
                        # cursor's point of view nothing happened.
                        self._note_struggle("deadzone_reject")
                        self._dz_streak += 1
                        if self._dz_streak == DEADZONE_EVIDENCE_STREAK:
                            self._capture_moment("deadzone_reject", frame)
                    else:
                        self._dz_streak = 0
                cursor_x, cursor_y = self._committed_x, self._committed_y

                current_time = time.time()

                # ── Dwell click ──
                # Resting the cursor within dwell_radius for dwell_time triggers a
                # click — a low-effort alternative for users who cannot hold a
                # mouth gesture (e.g. Parkinson's, Cerebral Palsy). Suspended while
                # dragging so the two click paths never collide. (cursor is None
                # only before the first gaze reading — nothing to dwell on yet.)
                if (self.clicks_enabled and self.dwell_click_enabled
                        and self.cursor_enabled
                        and not self.is_dragging and cursor_x is not None):
                    self._handle_dwell(cursor_x, cursor_y, current_time,
                                       pyautogui, frame)

                # ── Switch/scanning routing (Wave-2; default OFF) ──
                # When the scanning integrator owns a face gesture
                # (switch_gesture set AND a callback wired), that gesture is
                # the SWITCH and never performs its normal action. Deliberately
                # independent of clicks_enabled: scanning replaces clicking.
                switch_mode = (self.switch_gesture
                               if self.switch_callback is not None else None)

                if switch_mode == "mouth":
                    # The open mouth IS the switch: fire on the open rising
                    # edge (600 ms debounce); the click machine below is gated
                    # off so a switch press can never also mouth-click.
                    raw_open = self._face.detect_mouth_open(landmarks, self.screen_h)
                    if (raw_open and not self._switch_mouth_prev
                            and current_time - self._switch_mouth_ts >= 0.6):
                        self._switch_mouth_ts = current_time
                        self._fire_switch()
                    self._switch_mouth_prev = raw_open

                # ── Mouth (lips) gestures — the head-only mouse ──
                # Duration the mouth is held open decides the action:
                #   < 1s            -> LEFT click  (double on a rapid second tap)
                #   1s .. 3s        -> RIGHT click
                #   >= 3s           -> GRAB (drag start); close mouth to DROP
                # The timing state machine lives in FaceGestureRecognizer; this
                # block only maps its result to mouse side-effects. Gating off
                # forces "mouth closed" but still steps the machine — identical
                # to the old inline `mouth_open = ... else False` behaviour.
                g = self._face.update(
                    landmarks, current_time, self.screen_h,
                    enabled=(self.mouth_click_enabled and self.clicks_enabled
                             and self.cursor_enabled       # scanning replaces clicking
                             and switch_mode != "mouth"))

                if g.action == "swipe_start":
                    self._swiping = True
                    self._swipe_points.clear()
                    self._swipe_hint = ""
                    self._emit_gesture("swipe_start")
                elif g.action == "swipe_end":
                    self._fire_swipe(pyautogui)
                elif g.action == "drag":
                    pyautogui.mouseDown(button="left")
                    self._notify_click("drag")
                elif g.action == "auto_release":
                    # Safety: auto-release a drag that has run too long so a noisy
                    # mouth signal can never leave the button stuck down. Same
                    # swallow-errors mouseUp as the old _release_drag path, and
                    # deliberately no click notify.
                    try:
                        pyautogui.mouseUp(button="left")
                    except Exception:
                        pass
                    self._note_struggle("drag_timeout")
                    self._capture_moment("drag_timeout", frame)
                elif g.action == "drop":
                    # mouth closed while dragging → DROP
                    pyautogui.mouseUp(button="left")
                    self._notify_click("drop")
                elif g.action == "double":
                    pyautogui.doubleClick()
                    self._notify_click("double")
                elif g.action == "click":
                    pyautogui.click()
                    self._note_struggle("click_mouth")
                    self._notify_click("click")
                elif g.action == "right":
                    pyautogui.rightClick()
                    self._note_struggle("click_mouth")
                    self._notify_click("right")

                # ── Optional face-gesture events (Wave-2; default OFF) ──
                # blink/double-blink/wink/brow-raise/head-tilt → mapped OS
                # actions. Suppression of blink/wink/brow while the mouth is
                # open or dragging lives inside update_events (documented
                # there), so the mouth machine above always wins.
                # When scanning owns the blink (switch_mode == "blink") the
                # event stage runs regardless of the profile's face-gesture
                # switches — the blink is the user's only switch — and blink/
                # double-blink fire the switch INSTEAD of their mapped action.
                # cursor_enabled False = scanning owns the session: non-switch
                # gesture actions are suspended too (a scanning user wants ONE
                # reliable signal, not surprise brow/tilt actions mid-scan).
                events_on = (self._face_events_on and self.clicks_enabled
                             and self.cursor_enabled)
                if events_on or switch_mode == "blink":
                    forced = None
                    if switch_mode == "blink" and not (
                            self._face.events_enabled
                            and self._face._event_params.blink_enabled):
                        # Arm blink recognition for this frame only; restore
                        # right after so leaving scanning can never leave
                        # blink-clicks enabled behind the profile's back.
                        forced = (self._face.events_enabled,
                                  self._face._event_params.blink_enabled)
                        self._face.events_enabled = True
                        self._face._event_params.blink_enabled = True
                    evs = self._face.update_events(landmarks, current_time)
                    if forced is not None:
                        (self._face.events_enabled,
                         self._face._event_params.blink_enabled) = forced
                    for ev in evs:
                        if (switch_mode == "blink"
                                and ev.kind in ("blink", "double_blink")):
                            # Each deliberate blink = one switch press
                            # (double_blink replaces a second plain blink),
                            # never the mapped click action.
                            self._fire_switch()
                            continue
                        if events_on:
                            self._dispatch_face_event(ev, pyautogui, current_time)
                if events_on:
                    # A HELD tilt auto-repeats its action (continuous scroll)
                    # — but only actions marked repeatable, so a tilt mapped
                    # to a click can never machine-gun clicks.
                    ta = self._face.tilt_active
                    if ta and current_time - self._tilt_repeat_ts >= 0.4:
                        act = self._face_map.get(f"tilt_{ta}", "none")
                        if gesture_actions.repeatable(act):
                            self._tilt_repeat_ts = current_time
                            self._do_action(act, pyautogui)

                # Live gesture state for the on-screen head guide.
                if self.gesture_callback:
                    self._emit_gesture_state(g.state, current_time)

                # Debug draw
                h, w, _ = frame.shape
                cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 3, (255, 0, 0), -1)

            elif result is None:
                # Throttled frame — MediaPipe did not run this iteration. Preserve
                # all state (especially an active drag); do nothing. (Without this
                # guard a held drag was released ~12x/sec, breaking drag-and-drop.)
                pass

            else:
                # MediaPipe ran but found no face → genuine loss.
                self.face_not_visible_counter += 1
                # Release a held drag only after several genuinely face-absent
                # frames, so one dropped detection can't break a drag-and-drop.
                if self.is_dragging and self.face_not_visible_counter >= self.lost_release_frames:
                    self._release_drag()
                    self._end_swipe()
                # Subject lost: hold the cursor and keep waiting. Do NOT self-
                # terminate — that would leak the camera lock (stop() releases it,
                # not the worker) and force the user to re-enable. Resumes on return.
                if self.face_not_visible_counter == self.max_face_not_visible_frames:
                    print("[VISION] Face not visible — pausing cursor (will resume)")
                    # Counted once per loss, at the point the cursor actually
                    # pauses — not once per frame, which would turn a single
                    # walk-away into hundreds of "struggles".
                    self._note_struggle("face_lost")
                    self._capture_moment("face_lost", frame)
                if self.gesture_callback and self.face_not_visible_counter >= self.lost_release_frames:
                    self._emit_gesture_state("idle", time.time())

            if self.frame_callback:
                self._emit_frame(frame, cv2)
            if self.show_window:
                cv2.imshow("Agent Mistey Vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                cv2.waitKey(1)

            time.sleep(0.01)

        # Teardown (face_mesh.close / cap.release / destroyAllWindows) is handled
        # by _release_capture, which the supervisor runs after every worker attempt
        # — clean stop, crash, or restart alike. No duplicate cleanup here.

    def _dispatch_face_event(self, ev, pyautogui, now):
        """Map one FaceEvent through the profile's gesture map to an OS action."""
        if ev.kind in ("tilt_left", "tilt_right"):
            # Arm the repeat throttle so the hold doesn't double-fire on the
            # same frame the tilt event lands.
            self._tilt_repeat_ts = now
        label = self._do_action(self._face_map.get(ev.kind, "none"), pyautogui)
        if label is not None:
            # An event-driven action (e.g. blink-click) re-arms dwell so the
            # dwell path can never stack a second click on top of it.
            self._dwell_anchor = None

    def _emit_frame(self, frame, cv2):
        """Push a small preview frame to the UI (throttled ~8 fps)."""
        now = time.time()
        if now - self._last_frame_emit < 0.12:
            return
        self._last_frame_emit = now
        try:
            small = cv2.resize(frame, (224, 168))
            self.frame_callback(small)
        except Exception:
            pass

    def _emit_gesture_state(self, state, now, force=False):
        if not force and now - self._last_gesture_emit < 0.1:
            return
        self._last_gesture_emit = now
        try:
            self.gesture_callback(state, 0.0)
        except Exception:
            pass

    # ---------------------------------------------------------
    # Dwell-to-click
    # ---------------------------------------------------------
    def _handle_dwell(self, x, y, now, pyautogui, frame=None):
        """Click when the cursor rests within dwell_radius for dwell_time seconds."""
        if self._dwell_anchor is None:
            self._dwell_anchor = (x, y)
            self._dwell_start = now
            self._dwell_fired = False
            return

        ax, ay = self._dwell_anchor
        if math.hypot(x - ax, y - ay) > self.dwell_radius:
            # Cursor moved away → re-arm at the new resting point.
            # Only a LATE re-arm counts as a difficulty. Re-arming early is just
            # moving to a new target, which is ordinary use; re-arming when the
            # hold was most of the way done is a click the person nearly had and
            # lost, and counting the two together would bury the signal in
            # normal pointing.
            if (not self._dwell_fired and self.dwell_time > 0
                    and (now - self._dwell_start)
                    >= self.dwell_time * DWELL_NEARLY_FRACTION):
                self._note_struggle("dwell_rearm")
                # The frame is passed through because a dwell that slips is the
                # most useful of these to actually see: it is usually posture,
                # and a photograph shows the head support that a schematic
                # cannot. It is still only kept if photographs were agreed to.
                self._capture_moment("dwell_rearm", frame)
            self._dwell_anchor = (x, y)
            self._dwell_start = now
            self._dwell_fired = False
        elif not self._dwell_fired and (now - self._dwell_start) >= self.dwell_time:
            pyautogui.click()
            self._dwell_fired = True
            self._note_struggle("click_dwell")
            # A dwell click arms the mouth double-click window, as before.
            self._face.last_click_time = now
            self._notify_click("click")

    # ---------------------------------------------------------
    # Mouth opening detection (logic lives in FaceGestureRecognizer)
    # ---------------------------------------------------------
    def detect_mouth_opening(self, landmarks):
        """Detect mouth opening using MediaPipe face mesh landmarks."""
        return self._face.detect_mouth_open(landmarks, self.screen_h)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    try:
        from mistey.core.resource_guard import ResourceGuard
    except ImportError as exc:
        print(f"Unable to import ResourceGuard: {exc}")
        sys.exit(1)

    guard = ResourceGuard()
    vm = VisionMouse(guard, show_window=False, click_callback=lambda c: print('[CLICK]', c))

    try:
        vm.start()
        print("VisionMouse running. Move your head to control the cursor. Ctrl+C to stop.")
        while vm.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping VisionMouse...")
    finally:
        vm.stop()

"""
HandMouse — cursor control by HAND (distinct from head/face control).

Why this exists
---------------
The app previously had only `VisionMouse` (nose/head → cursor). The "hand gesture"
and "head" options both started it, so they behaved identically. This module is a
genuinely different modality: the index fingertip moves the cursor and a pinch
(thumb + index) clicks. It shares the camera with VisionMouse via ResourceGuard,
so the two are mutually exclusive cursor drivers (managed by BackendController).

It reuses the same motor parameters as VisionMouse (smoothing, sensitivity,
dead-zone, dwell) so the AbilityProfile / calibration / live adaptation apply
identically. Pinch-to-click only (no held drag) — by design, so it cannot
reproduce the stuck-button drag hang.
"""

import os
import sys
import time
import math
from collections import deque

package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

import mistey.config as config
from mistey.core.supervised_thread import SupervisedThread
from mistey.input.smoothing import OneEuro2D
from mistey.input.hand_gestures import GestureRecognizer

# MediaPipe hand landmark indices.
_INDEX_TIP = 8
_THUMB_TIP = 4


class HandMouse:
    def __init__(self, resource_guard, show_window: bool = True, click_callback=None,
                 frame_callback=None, gesture_callback=None):
        self.guard = resource_guard
        self.show_window = show_window
        self.click_callback = click_callback
        self.frame_callback = frame_callback     # optional live preview (BGR frames)
        self.gesture_callback = gesture_callback  # (state, pinch_closeness 0..1)
        self._last_frame_emit = 0.0
        self._last_gesture_emit = 0.0
        # One-Euro filter → smooth, jitter-free fingertip → cursor tracking.
        self._smooth = OneEuro2D(freq=20.0, mincutoff=0.8, beta=0.008)
        # Absolute mapping: a comfortable box of the camera view maps to the whole
        # screen (whole screen reachable, no gain amplification of jitter). Smaller
        # box = faster. Set from the profile's sensitivity in apply_motor_profile.
        self.active_box = 0.30      # half-size around centre → region 0.20..0.80
        # Richer gesture set (swipes → next/previous/scroll, peace → right-click…).
        self._gestures = GestureRecognizer()
        self._prev_pose = "none"
        self._pose_since = 0.0
        self._pose_fired = False
        self._last_action = 0.0
        self.gesture_action_callback = None    # optional (label) on a discrete action

        self._supervisor = None
        self._cap = None
        self._holds_camera = False

        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.screen_w, self.screen_h = pyautogui.size()

        # EMA / cursor state
        self.prev_x = None
        self.prev_y = None
        self._committed_x = None
        self._committed_y = None

        # Motor parameters (seeded from config; overridden by apply_motor_profile)
        self.alpha = config.FACE_EMA_ALPHA
        self.sensitivity_factor = config.FACE_SENSITIVITY_FACTOR
        self.diagonal_sensitivity_factor = config.FACE_DIAGONAL_SENSITIVITY_FACTOR
        self.dead_zone_radius = config.FACE_DEAD_ZONE_RADIUS
        self.max_hand_not_visible_frames = config.MAX_FACE_NOT_VISIBLE_FRAMES

        # Click modes
        self.pinch_threshold = 0.08   # normalized thumb-index distance (easier)
        self.dwell_click_enabled = False
        self.dwell_time = 1.0
        self.dwell_radius = 40

        # Click / drag state
        self.clicks_enabled = True      # master gate (Scroll Scope pauses clicking)
        self.last_click_time = 0
        self._pinching = False          # thumb+middle held (left click/drag)
        self._pinch_start = 0.0
        self._rclick_active = False      # thumb+ring right-click latch
        self._scroll_prev_y = None       # back-of-hand scroll reference
        self.hand_not_visible_counter = 0
        self.is_dragging = False
        self._drag_started_at = 0.0
        self.drag_threshold = 0.9     # hold a pinch this long (s) to start dragging
                                      # (longer = a normal click won't turn into a drag)
        self.max_drag_seconds = 8.0   # safety auto-release so the button never sticks
        self.lost_release_frames = 5  # genuine hand-absent frames before dropping a drag

        # Dwell state
        self._dwell_anchor = None
        self._dwell_start = 0.0
        self._dwell_fired = False

        # Jitter estimate (for live adaptation)
        self._jitter_samples = deque(maxlen=60)

    # ---- profile ----
    def apply_motor_profile(self, profile) -> None:
        self.alpha = float(profile.smoothing_alpha)
        self.sensitivity_factor = float(profile.sensitivity)
        self.diagonal_sensitivity_factor = float(profile.diagonal_sensitivity)
        self.dead_zone_radius = int(profile.dead_zone_radius)
        self.max_hand_not_visible_frames = int(profile.max_face_not_visible_frames)
        self.dwell_click_enabled = bool(profile.dwell_click_enabled)
        self.dwell_time = float(profile.dwell_time_s)
        self.dwell_radius = int(profile.dwell_radius_px)
        # Map the profile's smoothing into One-Euro: heavier smoothing (low alpha)
        # → steadier (lower beta); lighter smoothing → more responsive.
        self._smooth.tune(mincutoff=0.8, beta=max(0.003, 0.03 * self.alpha))
        # Sensitivity → active-region size: higher sensitivity = smaller box = faster.
        self.active_box = max(0.18, min(0.42, 0.5 - 0.045 * self.sensitivity_factor))
        print(f"[HAND] motor profile applied (alpha={self.alpha}, "
              f"sens={self.sensitivity_factor}, dead_zone={self.dead_zone_radius}, "
              f"dwell={'on' if self.dwell_click_enabled else 'off'})")

    def _notify_click(self, click_type: str = "click"):
        if self.click_callback:
            try:
                self.click_callback(click_type)
            except Exception as e:
                print(f"[HAND] Click callback error: {e}")

    # ---- lifecycle ----
    def start(self):
        if self.running:
            return
        if not self.guard.acquire_camera():
            print("[HAND] Camera busy")
            return
        self._holds_camera = True
        self._supervisor = SupervisedThread(
            name="HandMouse",
            worker=self._hand_worker,
            cleanup=self._release_capture,
        )
        self._supervisor.start()
        print("[HAND] Started")

    def stop(self):
        if self._supervisor:
            self._supervisor.stop()
            self._supervisor = None
        self._release_capture()
        if self._holds_camera:
            self.guard.release_camera()
            self._holds_camera = False
        print("[HAND] Stopped")

    @property
    def running(self) -> bool:
        return self._supervisor.running if self._supervisor else False

    def get_health(self):
        if self._supervisor:
            return self._supervisor.get_health()
        return {"status": "stopped"}

    def current_jitter(self) -> float:
        if not self._jitter_samples:
            return 0.0
        return sum(self._jitter_samples) / len(self._jitter_samples)

    def _release_drag(self):
        """Release a held pinch-drag — prevents a stuck button on stop/lost/crash."""
        if self.is_dragging:
            try:
                import pyautogui
                pyautogui.mouseUp(button="left")
            except Exception:
                pass
            self.is_dragging = False
        self._pinching = False

    def set_clicks_enabled(self, on: bool):
        """Master click gate — the Scroll Scope pauses clicking while scrolling.
        Cursor movement/gestures are unaffected."""
        self.clicks_enabled = bool(on)
        if not on:
            self._release_drag()
            self._pinching = False
            self._rclick_active = False

    def _release_capture(self):
        # Never leave a pinch-drag button held when the camera goes away.
        self._release_drag()
        # Close the MediaPipe Hands graph on every teardown (crash/restart/stop),
        # not just the clean-exit path — a leak blocks the next Hands init.
        try:
            if getattr(self, "_hands", None) is not None:
                self._hands.close()
        except Exception:
            pass
        finally:
            self._hands = None
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

    # ---- core loop ----
    def _hand_worker(self, should_run):
        import cv2
        import mediapipe as mp
        import pyautogui

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        self.prev_x = self.prev_y = None
        self._committed_x = self._committed_y = None
        self._dwell_anchor = None
        self._dwell_fired = False
        self.hand_not_visible_counter = 0
        self.is_dragging = False
        self._pinching = False

        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._hands = hands      # so _release_capture can close it on any exit

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
        self._cap = cap
        if not cap.isOpened():
            raise RuntimeError("Camera not accessible")
        print("[HAND] Camera opened")

        last_mp_time = 0
        MP_INTERVAL = 0.05  # ~20 FPS

        while should_run():
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            now = time.time()
            if now - last_mp_time >= MP_INTERVAL:
                # Low-light boost (user-tunable via the cockpit's Camera panel).
                from mistey.input import camera_tuning
                proc = camera_tuning.maybe_boost(rgb, cv2)
                proc.flags.writeable = False
                result = hands.process(proc)
                last_mp_time = now
            else:
                result = None

            hand_list = result.multi_hand_landmarks if result else None

            if hand_list:
                self.hand_not_visible_counter = 0
                lm = hand_list[0].landmark
                idx = lm[_INDEX_TIP]
                current_time = now

                gest = self._gestures.update(lm, now)
                cursor_x, cursor_y = self._committed_x, self._committed_y

                if gest.orientation == "back":
                    # Back of the hand shown → SCROLL mode: raise/lower the hand.
                    # Release any in-progress click/drag first — otherwise the left
                    # button stays latched (the release logic lives in the front
                    # branch and would be skipped here).
                    if self.is_dragging or self._pinching:
                        self._release_drag()
                    self._scroll_by_hand(lm, pyautogui)
                    self._smooth.reset()
                else:
                    self._scroll_prev_y = None
                    # Cursor moves only when ENGAGED (index pointing) and not doing
                    # an open-palm swipe — so a relaxed hand or fist pauses.
                    if gest.engaged and gest.pose != "palm":
                        # Absolute mapping: fingertip in the active box → whole
                        # screen, linearly (full reach, jitter not amplified).
                        fx, fy = self._smooth(idx.x, idx.y, now)
                        lo = 0.5 - self.active_box
                        span = 2.0 * self.active_box
                        nx = min(1.0, max(0.0, (fx - lo) / span))
                        ny = min(1.0, max(0.0, (fy - lo) / span))
                        tx = max(2, min(self.screen_w - 2, int(nx * self.screen_w)))
                        ty = max(2, min(self.screen_h - 2, int(ny * self.screen_h)))
                        if self._committed_x is None:
                            self._committed_x, self._committed_y = tx, ty
                        delta = math.hypot(tx - self._committed_x, ty - self._committed_y)
                        self._jitter_samples.append(delta)
                        if delta >= self.dead_zone_radius:
                            self._committed_x, self._committed_y = tx, ty
                            pyautogui.moveTo(tx, ty)
                        cursor_x, cursor_y = self._committed_x, self._committed_y
                    else:
                        self._smooth.reset()

                    # Discrete gestures — swipes + thumbs/peace.
                    self._handle_discrete(gest, now, pyautogui)

                    # LEFT click / drag — thumb + MIDDLE finger (index stays free to
                    # point, so clicking never nudges the cursor). Hold to drag.
                    if gest.lclick and self.clicks_enabled:
                        if not self._pinching:
                            self._pinching = True
                            self._pinch_start = current_time
                        elif (current_time - self._pinch_start) >= self.drag_threshold and not self.is_dragging:
                            pyautogui.mouseDown(button="left")
                            self.is_dragging = True
                            self._drag_started_at = current_time
                            self._notify_click("drag")
                        elif self.is_dragging and (current_time - self._drag_started_at) > self.max_drag_seconds:
                            self._release_drag()
                    else:
                        if self._pinching:
                            if self.is_dragging:
                                pyautogui.mouseUp(button="left")
                                self.is_dragging = False
                                self._notify_click("drop")
                            elif (current_time - self._pinch_start) < self.drag_threshold:
                                pyautogui.click()
                                self.last_click_time = current_time
                                self._notify_click("click")
                            self._pinching = False
                        elif (self.clicks_enabled and self.dwell_click_enabled
                              and not self.is_dragging):
                            self._handle_dwell(cursor_x, cursor_y, current_time, pyautogui)

                    # RIGHT click — thumb + RING finger (one-shot).
                    if gest.rclick and self.clicks_enabled and not self._rclick_active:
                        pyautogui.click(button="right")
                        self._notify_click("right")
                        self._rclick_active = True
                    elif not gest.rclick:
                        self._rclick_active = False

                # Live gesture state for the on-screen guide.
                if self.gesture_callback:
                    if gest.orientation == "back":
                        state = "scroll"
                    elif self.is_dragging:
                        state = "drag"
                    elif gest.lclick or self._pinching:
                        state = "click"
                    elif gest.rclick:
                        state = "rclick"
                    elif gest.pose in ("palm", "thumbs_up", "fist"):
                        state = {"palm": "palm", "thumbs_up": "thumb", "fist": "fist"}[gest.pose]
                    elif gest.engaged:
                        state = "move"
                    else:
                        state = "idle"        # hand relaxed → not controlling
                    CLOSE = 0.16
                    # Bar fills to 1.0 exactly at the click threshold so the user
                    # sees when a left-click will fire.
                    click_at = self._gestures.CLICK
                    closeness = max(0.0, min(1.0, (CLOSE - gest.pinch_dist) / (CLOSE - click_at)))
                    self._emit_gesture(state, closeness, current_time)

                if self.show_window or self.frame_callback:
                    h, w, _ = frame.shape
                    cv2.circle(frame, (int(idx.x * w), int(idx.y * h)), 8, (0, 255, 128), -1)   # index → cursor
                    mid = lm[12]
                    cv2.circle(frame, (int(mid.x * w), int(mid.y * h)), 6, (0, 200, 255), -1)   # middle → click
                    tb = lm[_THUMB_TIP]
                    cv2.circle(frame, (int(tb.x * w), int(tb.y * h)), 6, (255, 200, 0), -1)      # thumb
            elif result is None:
                # Throttled frame — MediaPipe did not run. Preserve pinch/drag
                # state (without this, the pinch reset ~20x/sec and drag broke).
                pass

            else:
                # MediaPipe ran but found no hand → genuine loss.
                self.hand_not_visible_counter += 1
                if self.hand_not_visible_counter >= self.lost_release_frames:
                    if self.is_dragging:
                        self._release_drag()   # drop a held drag, never stick
                    self._pinching = False
                    self._rclick_active = False  # clear the right-click latch on loss
                    self._smooth.reset()       # avoid a jump when the hand returns
                    self._gestures.reset()
                    self._prev_pose = "none"
                    if self.gesture_callback:
                        self._emit_gesture("none", 0.0, time.time())
                # Subject lost: hold the cursor and keep waiting (do NOT self-
                # terminate — that would leak the camera lock). Resumes on return.
                if self.hand_not_visible_counter == self.max_hand_not_visible_frames:
                    print("[HAND] Hand not visible — pausing cursor (will resume)")

            if self.frame_callback:
                self._emit_frame(frame, cv2)
            if self.show_window:
                cv2.imshow("Agent Mistey Hand", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                cv2.waitKey(1)
            time.sleep(0.01)

        # Teardown (hands.close / cap.release / destroyAllWindows) is handled by
        # _release_capture, which the supervisor runs after every worker attempt
        # — clean stop, crash, or restart alike. No duplicate cleanup here.

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

    def _emit_gesture(self, state, closeness, now, force=False):
        if not force and now - self._last_gesture_emit < 0.08:
            return
        self._last_gesture_emit = now
        try:
            self.gesture_callback(state, float(closeness))
        except Exception:
            pass

    def _notify_action(self, label):
        if self.gesture_action_callback:
            try:
                self.gesture_action_callback(label)
            except Exception:
                pass

    def _handle_discrete(self, gest, now, pyautogui):
        """Swipes → next/previous/scroll; peace → right-click; thumbs-up → Enter."""
        if gest.swipe:
            table = {"right": ("press", "right", "next", "swipe_right"),
                     "left":  ("press", "left", "previous", "swipe_left"),
                     "up":    ("scroll", 320, "scroll up", "swipe_up"),
                     "down":  ("scroll", -320, "scroll down", "swipe_down")}
            act, arg, label, hud = table[gest.swipe]
            if act == "press":
                pyautogui.press(arg)
            else:
                pyautogui.scroll(arg)
            self._notify_action(label)
            if self.gesture_callback:
                self._emit_gesture(hud, 0.0, now, force=True)
            self._prev_pose = gest.pose
            return
        # Pose-hold actions — must HOLD peace/thumbs-up briefly so a hand shape
        # you pass through on the way to pointing can't accidentally fire them.
        if gest.pose != self._prev_pose:
            self._pose_since = now
            self._pose_fired = False
        self._prev_pose = gest.pose
        if self._pose_fired or (now - self._pose_since) < 0.45:
            return
        if gest.pose == "peace":
            pyautogui.hotkey("win", "shift", "s")     # screenshot (right-click is
            self._notify_action("screenshot"); self._pose_fired = True   # thumb+ring)
        elif gest.pose == "thumbs_up":
            pyautogui.press("enter")
            self._notify_action("enter"); self._pose_fired = True

    def _scroll_by_hand(self, lm, pyautogui):
        """Back-of-hand scroll: raising/lowering the hand scrolls the page."""
        py = lm[9].y
        if self._scroll_prev_y is not None:
            dy = py - self._scroll_prev_y
            if abs(dy) > 0.008:
                pyautogui.scroll(int(-dy * 2600))     # hand up → scroll up
        self._scroll_prev_y = py
        if self.gesture_callback:
            self._emit_gesture("scroll", 0.0, time.time(), force=True)

    def _handle_dwell(self, x, y, now, pyautogui):
        if self._dwell_anchor is None:
            self._dwell_anchor = (x, y)
            self._dwell_start = now
            self._dwell_fired = False
            return
        ax, ay = self._dwell_anchor
        if math.hypot(x - ax, y - ay) > self.dwell_radius:
            self._dwell_anchor = (x, y)
            self._dwell_start = now
            self._dwell_fired = False
        elif not self._dwell_fired and (now - self._dwell_start) >= self.dwell_time:
            pyautogui.click()
            self._dwell_fired = True
            self.last_click_time = now
            self._notify_click("click")

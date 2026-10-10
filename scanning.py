"""Switch/scanning access — the pure state machine + switch-trigger abstractions.

Scanning access lets a user who can produce only ONE reliable signal (a single
switch press, a sound, a blink relayed by the integrator) operate the whole
screen: regions are highlighted one after another on a timer, and the switch
"activates" whichever region is currently lit.

This module is deliberately PURE:
  * ``ScanningController`` never touches the OS, a camera, Qt, or the clock —
    the integrator feeds it ``tick(now)`` / ``activate(now)`` and reacts to its
    callbacks (draw a highlight, move the cursor, perform a click).  That makes
    the whole state machine fake-clock testable, mirroring the dwell logic in
    ``vision_mouse._handle_dwell``.
  * Triggers (``KeyTrigger`` / ``SoundTrigger`` / ``CallableTrigger``) import
    their soft dependencies (pynput / sounddevice) lazily inside ``start`` and
    NEVER raise — a missing library or microphone just leaves ``available``
    False, exactly like ``ui/cockpit/mic_level.MicLevel``.

State flow (each level scanned on a timer, switch press descends):

    IDLE ──start/activate──▶ GROUPS ──▶ ITEMS ──▶ ACTION ──execute──▶ GROUPS…
                                                    │
                                                    └─"Finer"─▶ FINER ─▶ ACTION

  * GROUPS — 3 horizontal screen bands + a 4th "Mistey bar" group (the cockpit
    bottom bar, the last ``MISTEY_BAR_H`` px of the screen).
  * ITEMS  — the chosen band split into ``scan_cols`` cells, left → right.
  * ACTION — 5-item menu [Click, Double, Right click, Drag toggle, Finer].
    "Finer" re-scans the selected cell as a 2×2 grid, then returns to ACTION
    for the chosen sub-cell (repeatable for ever finer targeting).
  * Two full highlight loops WITHOUT an activation at any level ascend one
    level (FINER→ACTION, ACTION→ITEMS, ITEMS→GROUPS, GROUPS→IDLE rest).  A
    switch press while resting resumes at GROUPS.

The machine stays dumb: it only *emits* action names ("click", "double",
"right_click", "drag_toggle") with screen coordinates — the integrator decides
what a drag toggle means for the running drag state.
"""

import time
from dataclasses import dataclass

# Height (px) of the cockpit bottom bar — the 4th scan group, so the user can
# always reach Mistey's own controls through the switch.
MISTEY_BAR_H = 78

# The ACTION menu, in scan order.  Labels are UI-facing; names are what
# ``on_action`` receives.
_ACTIONS = [
    ("Click", "click"),
    ("Double", "double"),
    ("Right click", "right_click"),
    ("Drag toggle", "drag_toggle"),
    ("Finer", "finer"),
]

# States (also the strings handed to ``on_state``).
IDLE = "idle"
GROUPS = "groups"
ITEMS = "items"
ACTION = "action"
FINER = "finer"


@dataclass
class Region:
    """A highlightable rectangle in SCREEN coordinates."""
    x: int
    y: int
    w: int
    h: int
    label: str = ""
    kind: str = "cell"          # "cell" | "group" | "action"

    @property
    def centre(self):
        return self.x + self.w // 2, self.y + self.h // 2


def _clamp_params(params):
    """Merge user params over the defaults and clamp to sane bounds so a bad
    profile value can never make scanning unusable (interval 0 == strobe)."""
    p = {
        "scan_interval_s": 1.2,     # highlight dwell per region
        "scan_rows": 3,             # horizontal bands in GROUPS
        "scan_cols": 4,             # cells per band in ITEMS
        "rest_after_loops": 2,      # full loops w/o activation before ascending
    }
    p.update(params or {})
    p["scan_interval_s"] = min(10.0, max(0.3, float(p["scan_interval_s"])))
    p["scan_rows"] = min(8, max(1, int(p["scan_rows"])))
    p["scan_cols"] = min(8, max(1, int(p["scan_cols"])))
    p["rest_after_loops"] = min(10, max(1, int(p["rest_after_loops"])))
    return p


class ScanningController:
    """Pure, fake-clock-testable scanning state machine.

    The integrator owns the clock: call ``tick(now)`` frequently (any cadence —
    the controller advances only when ``scan_interval_s`` has elapsed) and
    ``activate(now)`` whenever the user's switch fires.  All side effects go
    through the four callbacks; each is guarded so an overlay bug can never
    kill the scan loop.
    """

    def __init__(self, screen_w, screen_h, params=None,
                 on_highlight=None,   # fn(region|None, state, index) — overlay hook
                 on_move=None,        # fn(x, y) — move OS cursor to cell centre
                 on_action=None,      # fn(action_name, x, y) — perform at point
                 on_state=None):      # fn(state_name) — telemetry / HUD
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.params = _clamp_params(params)
        self.on_highlight = on_highlight
        self.on_move = on_move
        self.on_action = on_action
        self.on_state = on_state

        self._active = False
        self._state = IDLE
        self._regions = []          # regions being scanned in the current state
        self._index = 0             # currently highlighted region
        self._loops = 0             # completed full loops without activation
        self._last_advance = None   # fake-clock anchor (None = await first tick)
        self.target = None          # the cell the ACTION menu will act on

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def active(self):
        return self._active

    @property
    def state(self):
        return self._state

    @property
    def regions(self):
        """The region list currently being scanned (read-only view)."""
        return tuple(self._regions)

    def start(self):
        """Begin scanning at the GROUPS level."""
        self._active = True
        self.target = None
        self._enter(GROUPS, self._build_groups())

    def stop(self):
        """Stop scanning entirely and clear the highlight."""
        self._active = False
        self._state = IDLE
        self._regions = []
        self._index = 0
        self._loops = 0
        self._last_advance = None
        self.target = None
        self._emit_highlight(None)
        self._emit_state()

    def current(self):
        """The currently highlighted Region, or None when idle/stopped."""
        if not self._active or self._state == IDLE or not self._regions:
            return None
        return self._regions[self._index]

    # ------------------------------------------------------------------
    # Clock + switch inputs
    # ------------------------------------------------------------------
    def tick(self, now):
        """Advance the highlight once ``scan_interval_s`` has elapsed.

        Same anchor/fire pattern as the dwell click: the first tick after a
        (re)start only sets the time anchor, so the first region always gets a
        full interval of highlight before moving on.
        """
        if not self._active or self._state == IDLE:
            return
        if self._last_advance is None:
            self._last_advance = now
            return
        if now - self._last_advance < self.params["scan_interval_s"]:
            return
        self._last_advance = now

        nxt = self._index + 1
        if nxt < len(self._regions):
            self._index = nxt
            self._emit_highlight(self._regions[self._index])
            return

        # Wrapped — one full loop without an activation.
        self._loops += 1
        if self._loops >= self.params["rest_after_loops"]:
            self._ascend()
        else:
            self._index = 0
            self._emit_highlight(self._regions[self._index])

    def activate(self, now):
        """The switch fired: descend into / select / execute the highlight."""
        if not self._active:
            return
        self._last_advance = now      # fresh interval at whatever comes next

        if self._state == IDLE:
            # Resting after auto-ascend — a press resumes scanning.
            self._enter(GROUPS, self._build_groups())
            return

        region = self._regions[self._index]

        if self._state == GROUPS:
            self._enter(ITEMS, self._build_items(region))

        elif self._state in (ITEMS, FINER):
            # A cell was chosen: it becomes the action target, the OS cursor
            # moves there so the user can see what they are about to act on.
            self.target = region
            self._emit_move(*region.centre)
            self._enter(ACTION, self._build_actions(region))

        elif self._state == ACTION:
            _, name = _ACTIONS[self._index]
            if name == "finer":
                self._enter(FINER, self._build_finer(self.target))
            else:
                x, y = self.target.centre
                self._emit_action(name, x, y)
                # Executed — start over at the top level for the next task.
                self._enter(GROUPS, self._build_groups())

    # ------------------------------------------------------------------
    # Internal state transitions
    # ------------------------------------------------------------------
    def _enter(self, state, regions):
        self._state = state
        self._regions = regions
        self._index = 0
        self._loops = 0
        self._emit_state()
        self._emit_highlight(self._regions[0] if self._regions else None)

    def _ascend(self):
        """Two silent loops — the user probably overshot: back up one level."""
        if self._state == FINER:
            self._enter(ACTION, self._build_actions(self.target))
        elif self._state == ACTION:
            # Re-scan the cells of the band the target belongs to.
            self._enter(ITEMS, self._build_items(self._band_of(self.target)))
        elif self._state == ITEMS:
            self._enter(GROUPS, self._build_groups())
        else:  # GROUPS → rest (still active; a switch press resumes)
            self._state = IDLE
            self._regions = []
            self._index = 0
            self._loops = 0
            self._last_advance = None
            self._emit_highlight(None)
            self._emit_state()

    # ------------------------------------------------------------------
    # Region geometry (integer tiling: no gaps, no overlap)
    # ------------------------------------------------------------------
    def _build_groups(self):
        """3 (scan_rows) horizontal bands over the desktop + the Mistey bar."""
        rows = self.params["scan_rows"]
        usable_h = self.screen_h - MISTEY_BAR_H
        groups = []
        for i in range(rows):
            y0 = round(i * usable_h / rows)
            y1 = round((i + 1) * usable_h / rows)
            groups.append(Region(0, y0, self.screen_w, y1 - y0,
                                 label=f"Band {i + 1}", kind="group"))
        groups.append(Region(0, usable_h, self.screen_w, MISTEY_BAR_H,
                             label="Mistey bar", kind="group"))
        return groups

    def _build_items(self, band):
        """Split a band into scan_cols cells, left → right."""
        cols = self.params["scan_cols"]
        cells = []
        for i in range(cols):
            x0 = band.x + round(i * band.w / cols)
            x1 = band.x + round((i + 1) * band.w / cols)
            cells.append(Region(x0, band.y, x1 - x0, band.h,
                                label=f"Cell {i + 1}", kind="cell"))
        return cells

    def _build_finer(self, cell):
        """Re-scan a chosen cell as a 2×2 grid (quarter cells)."""
        subs = []
        for r in range(2):
            y0 = cell.y + round(r * cell.h / 2)
            y1 = cell.y + round((r + 1) * cell.h / 2)
            for c in range(2):
                x0 = cell.x + round(c * cell.w / 2)
                x1 = cell.x + round((c + 1) * cell.w / 2)
                subs.append(Region(x0, y0, x1 - x0, y1 - y0,
                                   label=f"Quarter {r * 2 + c + 1}",
                                   kind="cell"))
        return subs

    def _build_actions(self, cell):
        """The 5-item action menu.  Geometry is a courtesy for the overlay:
        equal vertical strips across the target cell (the overlay is free to
        draw the menu however it likes — only label/kind are load-bearing)."""
        n = len(_ACTIONS)
        actions = []
        for i, (label, _) in enumerate(_ACTIONS):
            x0 = cell.x + round(i * cell.w / n)
            x1 = cell.x + round((i + 1) * cell.w / n)
            actions.append(Region(x0, cell.y, x1 - x0, cell.h,
                                  label=label, kind="action"))
        return actions

    def _band_of(self, region):
        """The GROUPS band that contains a region's centre (for ascending)."""
        _, cy = region.centre
        for band in self._build_groups():
            if band.y <= cy < band.y + band.h:
                return band
        return self._build_groups()[0]

    # ------------------------------------------------------------------
    # Guarded callback emitters — an overlay bug must never kill scanning
    # ------------------------------------------------------------------
    def _emit_highlight(self, region):
        if self.on_highlight is None:
            return
        try:
            self.on_highlight(region, self._state, self._index)
        except Exception:
            pass

    def _emit_move(self, x, y):
        if self.on_move is None:
            return
        try:
            self.on_move(x, y)
        except Exception:
            pass

    def _emit_action(self, name, x, y):
        if self.on_action is None:
            return
        try:
            self.on_action(name, x, y)
        except Exception:
            pass

    def _emit_state(self):
        if self.on_state is None:
            return
        try:
            self.on_state(self._state)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# Switch triggers — HOW the single activation signal reaches the machine
# ══════════════════════════════════════════════════════════════════════

class SwitchTrigger:
    """Base switch trigger.

    ``start(on_activate)`` arms the trigger; ``stop()`` disarms it.  Neither
    ever raises.  IMPORTANT: ``on_activate`` is invoked on the trigger's OWN
    thread (pynput listener / sounddevice audio callback) — the integrator is
    responsible for marshalling into its event loop (e.g. a Qt signal) before
    touching the controller or any UI.
    """

    def __init__(self, debounce_s=0.0):
        self.debounce_s = max(0.0, float(debounce_s))
        self.available = False
        self._on_activate = None
        self._last_fire = None

    def start(self, on_activate):
        self._on_activate = on_activate
        self._last_fire = None
        self.available = True

    def stop(self):
        self._on_activate = None
        self.available = False

    def _emit(self, now=None):
        """Debounced, guarded delivery of one activation."""
        cb = self._on_activate
        if cb is None:
            return
        if now is None:
            now = time.monotonic()
        if self._last_fire is not None and now - self._last_fire < self.debounce_s:
            return
        self._last_fire = now
        try:
            cb()
        except Exception:
            pass


class CallableTrigger(SwitchTrigger):
    """Integrator-driven switch: anything that already detects an event
    (a blink from the face loop, a voice intent, a UI button) just calls
    ``.fire()``.  Default debounce 0 — fires on every call; pass
    ``debounce_s`` to rate-limit noisy sources."""

    def fire(self, now=None):
        self._emit(now)


class KeyTrigger(SwitchTrigger):
    """Global keyboard switch via pynput (soft dependency, imported lazily).

    ``key=None`` treats ANY key as the switch (classic single-switch setup on
    a big-button USB keyboard); otherwise pass a key name ("space", "f9", "a").
    400 ms debounce absorbs key-repeat and tremor double-presses.
    """

    _warned = False   # log the missing-pynput situation once per process

    def __init__(self, key=None, debounce_s=0.4):
        super().__init__(debounce_s=debounce_s)
        self.key = key.lower() if isinstance(key, str) else key
        self._listener = None

    def start(self, on_activate):
        self._on_activate = on_activate
        self._last_fire = None
        try:
            from pynput import keyboard   # soft dep: absent → unavailable
        except Exception:
            if not KeyTrigger._warned:
                print("[scanning] pynput not installed — key switch unavailable")
                KeyTrigger._warned = True
            self.available = False
            return
        try:
            def _on_press(key):
                try:
                    if self._matches(key):
                        self._emit()
                except Exception:
                    pass

            self._listener = keyboard.Listener(on_press=_on_press)
            self._listener.start()
            self.available = True
        except Exception as exc:
            print(f"[scanning] key switch unavailable: {exc}")
            self.available = False

    def _matches(self, key):
        if self.key is None:
            return True
        name = getattr(key, "name", None) or getattr(key, "char", None)
        return isinstance(name, str) and name.lower() == self.key

    def stop(self):
        try:
            if self._listener is not None:
                self._listener.stop()
        except Exception:
            pass
        self._listener = None
        super().stop()


class SoundTrigger(SwitchTrigger):
    """Any-loud-sound switch via sounddevice (soft dependency, lazy import).

    Fires when the mic RMS level (scaled to the same lively 0..1 range the
    voice HUD equalizer shows — see ``ui/cockpit/mic_level``) exceeds
    ``threshold``.  600 ms debounce turns one vocalisation into one press.
    Fails quietly when there is no microphone (``available`` stays False).
    """

    def __init__(self, threshold=0.6, debounce_s=0.6):
        super().__init__(debounce_s=debounce_s)
        self.threshold = min(1.0, max(0.0, float(threshold)))
        self._stream = None

    def start(self, on_activate):
        self._on_activate = on_activate
        self._last_fire = None
        try:
            import numpy as np
            import sounddevice as sd

            def _cb(indata, frames, time_info, status):
                try:
                    rms = float(np.sqrt(np.mean(np.square(indata))))
                    level = min(1.0, (rms * 9.0) ** 0.8)
                    if level > self.threshold:
                        self._emit()
                except Exception:
                    pass

            self._stream = sd.InputStream(
                channels=1, samplerate=16000, blocksize=512, callback=_cb)
            self._stream.start()
            self.available = True
        except Exception as exc:
            print(f"[scanning] sound switch unavailable: {exc}")
            self.available = False

    def stop(self):
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        super().stop()

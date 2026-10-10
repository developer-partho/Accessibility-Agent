"""Face-gesture → OS-action dispatch table (Wave-2, optional).

Maps the discrete :class:`~mistey.input.face_gestures.FaceEvent` kinds
(blink / double-blink / wink / eyebrow-raise / head-tilt) to concrete mouse
and keyboard side-effects. Pure logic with the pyautogui handle injected per
call, so it is unit-testable without a display — mirroring hand_mouse's
open-palm swipe table (back/forward = alt+left / alt+right).

Everything here is inert unless an AbilityProfile turns
``face_gestures_enabled`` on: the recognizer emits no events by default, so
in a default run this module is never consulted (zero regression).
"""


def _left_click(pg):
    pg.click()


def _double_click(pg):
    pg.doubleClick()


def _right_click(pg):
    pg.rightClick()


def _scroll_up(pg):
    pg.scroll(120)


def _scroll_down(pg):
    pg.scroll(-120)


def _back(pg):
    pg.hotkey("alt", "left")


def _forward(pg):
    pg.hotkey("alt", "right")


def _scroll_left(pg):
    pg.hscroll(-120)


def _scroll_right(pg):
    pg.hscroll(120)


def _page_up(pg):
    pg.press("pageup")


def _page_down(pg):
    pg.press("pagedown")


def _switch_app(pg):
    pg.hotkey("alt", "tab")


def _refresh(pg):
    pg.press("f5")


def _escape(pg):
    pg.press("escape")


def _tap(pg):
    pg.click()


def _noop(pg):
    pass


# name → (side-effect fn(pyautogui), repeatable while held, HUD/notify label)
# "repeatable" marks actions safe to auto-repeat while a tilt is HELD
# (continuous scroll); one-shot actions like clicks must fire once per event.
ACTIONS = {
    "left_click":   (_left_click,   False, "click"),
    "double_click": (_double_click, False, "double"),
    "right_click":  (_right_click,  False, "right"),
    "scroll_up":    (_scroll_up,    True,  "scroll up"),
    "scroll_down":  (_scroll_down,  True,  "scroll down"),
    "back":         (_back,         False, "back"),
    "forward":      (_forward,      False, "forward"),
    "none":         (_noop,         False, None),
    # Added for swipe gestures. Every one of them is reversible or harmless.
    #
    # There is deliberately NO close-window, no delete and no power action
    # here. A swipe can be misread -- the recogniser says so itself, and a
    # person with a tremor will misfire one -- and an irreversible action
    # behind a misreadable gesture is not a trade worth making. Destructive
    # commands stay on the voice path, which asks for confirmation and counts
    # down before it acts.
    "scroll_left":  (_scroll_left,  True,  "scroll left"),
    "scroll_right": (_scroll_right, True,  "scroll right"),
    # Paging is NOT repeatable, though scrolling is. A held tilt auto-repeats
    # every 0.4 s, and a scroll notch is a few lines while a page is a whole
    # screen -- the same hold would throw somebody through a document far
    # faster than they could read where they had got to.
    "page_up":      (_page_up,      False, "page up"),
    "page_down":    (_page_down,    False, "page down"),
    "switch_app":   (_switch_app,   False, "switch app"),
    "refresh":      (_refresh,      False, "refresh"),
    "escape":       (_escape,       False, "escape"),
    "tap":          (_tap,          False, "tap"),
    # Re-centre the head cursor: take where the head is NOW as the middle.
    #
    # The function is a no-op on purpose. Every other entry here is a pure OS
    # side-effect taking pyautogui, and this one has to reach back into the
    # driver that owns the mapping -- so VisionMouse._do_action intercepts the
    # name before dispatch() is ever called. The entry exists so that
    # validate_map accepts it, so a person can map any face gesture to it, and
    # so the HUD has a word for it. Widening every action's signature to pass a
    # driver nothing else needs would be the worse trade.
    #
    # It belongs on a gesture where close-window and delete deliberately do
    # not: re-centring is harmless and instantly undone by re-centring again,
    # which is the test the rest of this table is chosen by.
    "recentre":     (_noop,         False, "re-centred"),
}

# Default event-kind → action map. Winks default to "none" (hard for many
# users and easy to fire accidentally); everything is remappable per profile
# via AbilityProfile.face_gesture_map (which mirrors this literal —
# ability/ must not import input/, so keep the two in sync).
DEFAULT_FACE_MAP = {
    "blink": "left_click",
    "double_blink": "double_click",
    "wink_l": "none",
    "wink_r": "none",
    "brow_raise": "right_click",
    "tilt_left": "scroll_up",
    "tilt_right": "scroll_down",
    "tilt_end": "none",
}


# Default swipe → action map. Mirrored by AbilityProfile._DEFAULT_SWIPE_MAP,
# for the same reason DEFAULT_FACE_MAP is — ability/ must not import input/, so
# keep the two literals in step (there is a test).
#
# The four cardinals cover what a person actually reaches for; diagonals are
# recognised but map to "none" until somebody asks, because eight targets is a
# lot to hit with a head. "hold" is the one gesture that needs no range at all,
# which is why it carries right-click.
DEFAULT_SWIPE_MAP = {
    "left": "back",
    "right": "forward",
    "up": "scroll_up",
    "down": "scroll_down",
    "hold": "right_click",
    "up_left": "none",
    "up_right": "none",
    "down_left": "none",
    "down_right": "none",
    # Shapes — a separate opt-in, and none of them destructive.
    "circle": "refresh",
    "zed": "escape",
    "ell": "switch_app",
    "vee": "page_down",
    "caret": "page_up",
}


def repeatable(action_name) -> bool:
    """True if the named action may auto-repeat while a tilt is held."""
    entry = ACTIONS.get(action_name)
    return bool(entry and entry[1])


def dispatch(action_name, pyautogui, notify=None):
    """Run one named action; returns its label, or None for none/unknown.

    A pyautogui failure is swallowed (with a log line) and returns None —
    an OS hiccup must never crash the supervised vision loop. ``notify``
    (VisionMouse._notify_click) gets the label on success; its own errors
    are swallowed too, matching _notify_click's contract.
    """
    fn, _repeat, label = ACTIONS.get(action_name, ACTIONS["none"])
    if label is None:
        return None
    try:
        fn(pyautogui)
    except Exception as e:
        print(f"[GESTURE] action '{action_name}' failed: {e}", flush=True)
        return None
    if notify is not None:
        try:
            notify(label)
        except Exception:
            pass
    return label


def validate_map(m) -> dict:
    """Sanitize a profile's face_gesture_map into a safe, complete copy.

    - unknown event kinds are dropped;
    - unknown/invalid action names become "none";
    - missing kinds fall back to DEFAULT_FACE_MAP;
    - non-dict input yields a fresh copy of DEFAULT_FACE_MAP.
    Always returns a NEW dict (never the input or the default by reference).
    """
    out = dict(DEFAULT_FACE_MAP)
    if not isinstance(m, dict):
        return out
    for kind, action in m.items():
        if kind not in DEFAULT_FACE_MAP:
            continue
        out[kind] = action if isinstance(action, str) and action in ACTIONS else "none"
    return out


def validate_swipe_map(m) -> dict:
    """Sanitize a profile's swipe_map into a safe, complete copy.

    Same contract as validate_map: unknown gestures dropped, unknown actions
    become "none", missing gestures fall back to the default, and the result is
    always a new dict. A corrupt saved map must cost the user their custom
    bindings, never their ability to use the mode.
    """
    out = dict(DEFAULT_SWIPE_MAP)
    if not isinstance(m, dict):
        return out
    for gesture, action in m.items():
        if gesture not in DEFAULT_SWIPE_MAP:
            continue
        out[gesture] = (action if isinstance(action, str) and action in ACTIONS
                        else "none")
    return out

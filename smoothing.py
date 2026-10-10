"""One-Euro filter — adaptive smoothing for pointer control.

Fixed smoothing (a plain EMA) forces a bad trade-off: enough smoothing to kill
jitter makes fast motion laggy; enough responsiveness to feel snappy lets the
cursor shake at rest. The One-Euro filter adapts the cutoff to the movement
speed — heavy smoothing when the hand is (nearly) still, light smoothing when it
moves fast — so hand-tracked cursors feel steady *and* responsive.

Reference: Casiez, Roussel, Vogel — "1€ Filter" (CHI 2012).

There is a second reason to prefer it, which is about honesty rather than feel.
A fixed-alpha EMA has no timestamp: ``y += alpha * (x - y)`` smooths by a time
constant proportional to the frame interval, so the same alpha is a different
filter at a different frame rate. The head loop was measured producing anywhere
from 7.78 to 10.40 Hz across sessions on one machine, which means a preset's
smoothing already varies between sittings. One-Euro recovers the rate from the
timestamps it is given, so its numbers mean the same thing whatever the loop
achieves — and that is what makes raising the sample rate possible at all.
"""

import math


# The rate to translate a fixed-alpha EMA against: the median the head loop was
# actually measured delivering (7.78-10.40 Hz observed, median 10.39), not the
# 12.5 Hz MP_INTERVAL nominally allows and not the 30 fps the camera is asked
# for. Translating against a rate nobody runs at would hand every preset a
# different amount of smoothing than it has today, which is the one thing this
# mapping exists to avoid.
EMA_REFERENCE_HZ = 10.4


def ema_equivalent_cutoff(alpha, reference_hz=EMA_REFERENCE_HZ):
    """The One-Euro at-rest cutoff that smooths like this EMA alpha does today.

    A fixed-alpha EMA is a first-order low-pass with time constant
    ``tau = -dt / ln(1 - alpha)``; a first-order low-pass with cutoff ``fc`` has
    ``tau = 1 / (2*pi*fc)``. Equating them gives the cutoff below.

    This is what lets One-Euro be offered without re-tuning 21 presets by hand:
    a preset that was authored as "heavy smoothing for a tremor" keeps the
    smoothing it was authored with, and only its rate-dependence goes away.

    Returns the Default preset's own equivalent for a nonsensical alpha rather
    than raising — this is read on the path that moves somebody's only pointer.
    """
    try:
        a = float(alpha)
        hz = float(reference_hz)
    except (TypeError, ValueError):
        return 0.37
    if not (0.0 < a < 1.0) or hz <= 0.0:
        return 0.37
    return -hz * math.log(1.0 - a) / (2.0 * math.pi)


class _LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        self.y = x if self.y is None else alpha * x + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    def __init__(self, freq=20.0, mincutoff=1.0, beta=0.02, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)   # ↓ = steadier at rest
        self.beta = float(beta)             # ↑ = more responsive when moving
        self.dcutoff = float(dcutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t = None

    def __call__(self, x, t):
        if self._last_t is not None:
            dt = t - self._last_t
            if dt > 0:
                self.freq = 1.0 / dt
        self._last_t = t
        prev = self._x.y
        dx = 0.0 if prev is None else (x - prev) * self.freq
        edx = self._dx(dx, self._alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self._x(x, self._alpha(cutoff))


class OneEuro2D:
    """A One-Euro filter for a 2D point (independent x/y)."""

    def __init__(self, **kw):
        self.fx = OneEuroFilter(**kw)
        self.fy = OneEuroFilter(**kw)

    def tune(self, mincutoff=None, beta=None):
        for f in (self.fx, self.fy):
            if mincutoff is not None:
                f.mincutoff = float(mincutoff)
            if beta is not None:
                f.beta = float(beta)

    def reset(self):
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x, y, t):
        return self.fx(x, t), self.fy(y, t)

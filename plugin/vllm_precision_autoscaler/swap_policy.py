"""When should the swap format change precision? Pure Python, unit-testable.

Signal: tokens per engine step, smoothed with a time-based exponential moving
average (EMA), so one long prompt doesn't trigger a switch.
Hysteresis: go to FP8 above `up`, back to 16-bit below `down` (down < up),
and never change the target more often than every `dwell_s` seconds.
"""

from __future__ import annotations

import math


class HysteresisPolicy:
    def __init__(self, up: float, down: float, dwell_s: float, ema_s: float,
                 start: str = "high") -> None:
        if down >= up:
            raise ValueError(f"down ({down}) must be < up ({up})")
        self.up, self.down, self.dwell_s, self.ema_s = up, down, dwell_s, ema_s
        self.target = start
        # Start at 0, not at the first step's value: the first real step is
        # often a big prefill (run 4 switched to FP8 at startup because of it).
        self.ema: float | None = 0.0
        self._last_t: float | None = None
        self._last_change: float = -math.inf

    def update(self, now: float, num_tokens: int) -> str:
        """Feed one engine step; returns the (possibly new) target precision."""
        if self.ema is None:
            self.ema = float(num_tokens)
        else:
            dt = max(now - (self._last_t or now), 0.0)
            alpha = 1.0 - math.exp(-dt / self.ema_s) if self.ema_s > 0 else 1.0
            self.ema += alpha * (num_tokens - self.ema)
        self._last_t = now
        if now - self._last_change >= self.dwell_s:
            if self.target == "high" and self.ema > self.up:
                self.target, self._last_change = "fp8", now
            elif self.target == "fp8" and self.ema < self.down:
                self.target, self._last_change = "high", now
        return self.target

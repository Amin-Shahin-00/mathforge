"""Adaptive curriculum: keep every engine working at the edge of its ability.

Learning is fastest when problems are neither trivial nor hopeless (the
"zone of proximal development"). We track an exponential moving average of
the success rate per difficulty level and move the level up or down to keep it
inside ``target +- band``. Sampling is a mixture centred on the current level
so earlier skills keep being rehearsed (avoids catastrophic forgetting).
"""

from __future__ import annotations

import random


class Curriculum:
    def __init__(self, target: float = 0.5, band: float = 0.15, step: int = 1,
                 min_level: int = 1, max_level: int = 50, ema: float = 0.3):
        self.target, self.band, self.step = target, band, step
        self.min_level, self.max_level = min_level, max_level
        self.alpha = ema
        self.level = min_level
        self.rate: float | None = None
        self.history: list[tuple[int, float]] = []

    def sample_level(self, rng: random.Random) -> int:
        r = rng.random()
        if r < 0.70:
            lvl = self.level
        elif r < 0.85:
            lvl = self.level + 1          # stretch
        else:
            lvl = rng.randint(self.min_level, self.level)  # rehearse
        return max(self.min_level, min(self.max_level, lvl))

    def update(self, successes: int, attempts: int) -> int:
        if attempts <= 0:
            return self.level
        r = successes / attempts
        self.rate = r if self.rate is None else (1 - self.alpha) * self.rate + self.alpha * r
        self.history.append((self.level, r))
        if self.rate > self.target + self.band and self.level < self.max_level:
            self.level += self.step
            self.rate = self.target          # reset the estimate at the new level
        elif self.rate < self.target - self.band and self.level > self.min_level:
            self.level -= self.step
            self.rate = self.target
        return self.level

    def state_dict(self) -> dict:
        return {"level": self.level, "rate": self.rate, "history": self.history[-500:]}

    def load_state_dict(self, d: dict | None) -> None:
        if not d:
            return
        self.level = int(d.get("level", self.level))
        self.rate = d.get("rate")
        self.history = [tuple(x) for x in d.get("history", [])]

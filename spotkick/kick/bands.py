"""The kick's arithmetic: listener state, distance scale, bands, measured selection, dose, and the verdict.

All distances are cosine distances in the ruler's space. Bands are relative to the listener's *own* recent
spread, because CLAP distances are compressed and a fixed threshold would mean different things to different
listeners: rel = (d − typical step) / (far − typical step); 0 is an ordinary next song, 1 the far edge of what
they have been playing lately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STRENGTHS = ("tap", "kick", "boot")
TARGET_REL = {"tap": 0.25, "kick": 0.75, "boot": 1.3}   # where each strength should land on the listener's scale
DOSE = {"tap": 1, "kick": 3, "boot": 5}                # songs we force: one is absorbed, five bend (phase 3d)
ACCEPT_NEAR, ACCEPT_FAR = 0.75, 0.63                   # play-past-half probability across distance (Yambda response curve)
DEFAULT_SCALE = (0.10, 0.40)                           # typical step, far — before we have 3 plays
MIN_GAP = 0.10                                         # far − step never smaller than this


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def strength_for(magnitude: float) -> str:
    m = max(0.0, min(1.0, float(magnitude)))
    return "tap" if m < 0.33 else "kick" if m < 0.66 else "boot"


def target_for(magnitude: float) -> float:
    """Continuous: the wind-up angle maps to a target rel, piecewise-linear through the band centers."""
    m = max(0.0, min(1.0, float(magnitude)))
    xs, ys = (0.0, 0.165, 0.5, 0.83, 1.0), (0.0, TARGET_REL["tap"], TARGET_REL["kick"], TARGET_REL["boot"], 1.6)
    return float(np.interp(m, xs, ys))


def dose_for(magnitude: float) -> int:
    return DOSE[strength_for(magnitude)]


def acceptance(rel: float) -> float:
    return ACCEPT_NEAR + (ACCEPT_FAR - ACCEPT_NEAR) * float(np.clip(rel, 0, 1))


@dataclass
class ListenerState:
    """EWMA of the songs played, on the unit sphere: x_t = normalize(α x_{t-1} + (1−α) e_t). A skip counts less."""
    alpha: float = 0.7
    vector: np.ndarray | None = None
    history: list[np.ndarray] = field(default_factory=list)

    def update(self, e: np.ndarray, weight: float = 1.0) -> None:
        e = np.asarray(e, dtype=np.float32)
        if self.vector is None:
            self.vector = normalize(e.copy())
        else:
            a = 1.0 - (1.0 - self.alpha) * weight
            self.vector = normalize(a * self.vector + (1.0 - a) * e)
        self.history.append(e)
        del self.history[:-40]

    def distance(self, e: np.ndarray) -> float:
        return float(1.0 - self.vector @ e) if self.vector is not None else 0.0

    def scale(self) -> tuple[float, float]:
        """(typical step, far) = median and 95th percentile of pairwise distances among the last 20 plays."""
        if len(self.history) < 3:
            return DEFAULT_SCALE
        H = np.stack(self.history[-20:])
        d = 1.0 - H @ H.T
        vals = d[np.triu_indices(len(H), 1)]
        step, far = float(np.quantile(vals, 0.5)), float(np.quantile(vals, 0.95))
        return step, max(far, step + MIN_GAP)   # a few near-identical plays must not make every song look far

    def rel(self, distance: float) -> float:
        step, far = self.scale()
        return (distance - step) / (far - step)

    def band_for(self, distance: float) -> str:
        r = self.rel(distance)
        return "tap" if r < 0.5 else "kick" if r < 1.0 else "boot"


@dataclass
class Measured:
    index: int
    distance: float
    rel: float
    band: str
    acceptance: float


def measure(state: ListenerState, embeddings: list[np.ndarray]) -> list[Measured]:
    out = []
    for i, e in enumerate(embeddings):
        d = state.distance(e); r = state.rel(d)
        out.append(Measured(i, d, r, state.band_for(d), acceptance(r)))
    return out


def choose(measured: list[Measured], target_rel: float) -> Measured | None:
    """The candidate whose measured rel is nearest the target. The leg tells the truth because this happens *after*
    measuring; a near-labelled song that measured far is simply a far song."""
    if not measured:
        return None
    return min(measured, key=lambda m: abs(m.rel - target_rel))


def followed(pre: np.ndarray, kick: np.ndarray, now: np.ndarray) -> float:
    """How far the listener state has moved along the kick, as a fraction of the kick's own displacement:
    ((now − pre) · (kick − pre)) / |kick − pre|². 0 = back where it was, 1 = sitting on the kick, negative = recoiled."""
    d = kick - pre
    n = float(d @ d)
    return float((now - pre) @ d / n) if n > 1e-9 else 0.0


def verdict(f: float, n_since: int, *, min_since: int = 2) -> str:
    if n_since < min_since:
        return "listening"
    return "returned" if f < 0.25 else "bent" if f < 0.6 else "followed"

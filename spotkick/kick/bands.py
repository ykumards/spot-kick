"""The kick arithmetic: listener state, distance scale, bands, selection and verdict.

All distances are cosine distances between CLAP embeddings. Bands are relative to the listener's own recent
spread, because CLAP distances are compressed and a fixed threshold would mean different things for different
listeners: rel = (d − typical step) / (far − typical step), where 0 is an ordinary next song and 1 the far edge
of recent listening.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STRENGTHS = ("tap", "kick", "boot")
TARGET_REL = {"tap": 0.25, "kick": 0.75, "boot": 1.3}   # where each strength should land on the listener's scale
DEFAULT_SCALE = (0.10, 0.40)                            # typical step, far — before we have 3 plays
MIN_GAP = 0.10                                          # far − step never smaller than this

# strength_for: the wind-up magnitude below which each strength applies, checked in order
STRENGTH_CEILINGS = (("tap", 0.33), ("kick", 0.66))
STRENGTH_ABOVE_CEILINGS = "boot"

# The wind-up magnitude at the centre of each strength: what `spotkick kick tap|kick|boot` means.
STRENGTH_MAGNITUDE = {"tap": 0.165, "kick": 0.5, "boot": 0.83}

# target_for: the wind-up magnitude maps to a target rel, piecewise-linear through the band centers
TARGET_MAGNITUDES = (0.0, *STRENGTH_MAGNITUDE.values(), 1.0)
TARGET_RELS = (0.0, *TARGET_REL.values(), 1.6)

# band_for: the rel below which each band applies, checked in order. The middle band is the widest: most picks
# land near the middle of the scale, and a kick aimed at 0.75 should own a fair margin either side of it.
BAND_CEILINGS = (("tap", 0.35), ("kick", 1.15))
BAND_ABOVE_CEILINGS = "boot"

# verdict: the followed fraction below which each verdict applies, checked in order
VERDICT_CEILINGS = (("returned", 0.25), ("bent", 0.6))
VERDICT_ABOVE_CEILINGS = "followed"

SONGS_TO_JUDGE = 2         # Spotify-chosen songs after a kick that decide its verdict; later ones don't move it
HISTORY_KEEP = 40          # plays remembered in the state
SCALE_WINDOW = 20          # plays the distance scale is computed over
SCALE_MIN_PLAYS = 3        # below this the default scale is used
DEGENERATE_KICK = 1e-9     # |kick − pre|² at or below this means the kick didn't move anywhere


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector


def clamp_unit(magnitude: float) -> float:
    return max(0.0, min(1.0, float(magnitude)))


def first_below(value: float, ceilings: tuple[tuple[str, float], ...], above: str) -> str:
    """Return the label of the first ceiling ``value`` is below, or ``above`` when it is below none."""
    for label, ceiling in ceilings:
        if value < ceiling:
            return label
    return above


def strength_for(magnitude: float) -> str:
    return first_below(clamp_unit(magnitude), STRENGTH_CEILINGS, STRENGTH_ABOVE_CEILINGS)


def target_for(magnitude: float) -> float:
    """Map a wind-up magnitude to a target rel, piecewise-linear through the band centres."""
    return float(np.interp(clamp_unit(magnitude), TARGET_MAGNITUDES, TARGET_RELS))


@dataclass
class ListenerState:
    """Exponentially weighted mean of the songs played, on the unit sphere: x_t = normalize(α x_{t-1} + (1−α) e_t).

    ``history`` feeds the distance scale and therefore holds only songs the recommender chose: a kicked song moves
    the state but must not define the typical step.
    """
    alpha: float = 0.7
    vector: np.ndarray | None = None
    history: list[np.ndarray] = field(default_factory=list)

    def update(self, embedding: np.ndarray, *, counts_for_scale: bool = True) -> None:
        embedding = np.asarray(embedding, dtype=np.float32)
        if self.vector is None:
            self.vector = normalize(embedding.copy())
        else:
            self.vector = normalize(self.alpha * self.vector + (1.0 - self.alpha) * embedding)
        if counts_for_scale:
            self.history.append(embedding)
            del self.history[:-HISTORY_KEEP]

    def distance(self, embedding: np.ndarray) -> float:
        if self.vector is None:
            return 0.0
        return float(1.0 - self.vector @ embedding)

    def scale(self) -> tuple[float, float]:
        """Return (typical step, far): the median and 95th percentile of the distances from the mean of the last
        SCALE_WINDOW recommender-chosen plays to each of them.

        Distance to a mean is what every candidate is measured with; pairwise song-to-song distances run about
        twice as large and would make every candidate read as closer than a typical step.
        """
        if len(self.history) < SCALE_MIN_PLAYS:
            return DEFAULT_SCALE
        recent = np.stack(self.history[-SCALE_WINDOW:])
        centre = normalize(recent.mean(axis=0))
        distances = 1.0 - recent @ centre
        step = float(np.quantile(distances, 0.5))
        far = float(np.quantile(distances, 0.95))
        # a few near-identical plays must not make every song look far
        return step, max(far, step + MIN_GAP)

    def rel(self, distance: float) -> float:
        step, far = self.scale()
        return (distance - step) / (far - step)

    def band_for(self, distance: float) -> str:
        return first_below(self.rel(distance), BAND_CEILINGS, BAND_ABOVE_CEILINGS)


@dataclass
class Measured:
    index: int
    distance: float
    rel: float
    band: str


def measure(state: ListenerState, embeddings: list[np.ndarray]) -> list[Measured]:
    measured = []
    for index, embedding in enumerate(embeddings):
        distance = state.distance(embedding)
        rel = state.rel(distance)
        measured.append(Measured(index, distance, rel, state.band_for(distance)))
    return measured


def choose(measured: list[Measured], target_rel: float) -> Measured | None:
    """Return the candidate whose measured rel is nearest ``target_rel``, or None for an empty list.

    Selection uses the measurement, not the brain's reach label.
    """
    if not measured:
        return None
    return min(measured, key=lambda candidate: abs(candidate.rel - target_rel))


def followed(pre: np.ndarray, kick: np.ndarray, now: np.ndarray) -> float:
    """Return how far the state moved along the kick, as a fraction of the kick's displacement.

    Computed as ((now − pre) · (kick − pre)) / |kick − pre|²: 0 is no movement, 1 is the kick itself, negative is
    movement away.
    """
    displacement = kick - pre
    squared_length = float(displacement @ displacement)
    if squared_length <= DEGENERATE_KICK:
        return 0.0
    return float((now - pre) @ displacement / squared_length)


def verdict(followed_fraction: float, n_since: int, *, min_since: int = SONGS_TO_JUDGE) -> str:
    if n_since < min_since:
        return "listening"
    return first_below(followed_fraction, VERDICT_CEILINGS, VERDICT_ABOVE_CEILINGS)

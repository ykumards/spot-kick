"""Mini-Me's first job: P(you keep this song), learned from your own log.

A logistic regression on the CLAP embedding, trained on how much of each song you let play: a song kept to the
end is a 1, one you kicked away from at 2% is a 0, a love is a 1 with extra weight. It is tiny by design — a few
hundred labelled plays is all a personal log ever holds — and it refits in milliseconds, so it is never stale.
The kick uses it to choose *which* bench song to send on among those that land near the target; it never
overrides the distance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_LABELS = 20            # below this the model stays off and the kick falls back to nearest-to-target
MIN_CLASS_MASS = 3.0       # ... and each side (kept / not kept) needs at least this much label mass
L2 = 1.0                   # ridge strength: 512 dimensions, a few hundred rows — regularise hard
LEARNING_RATE = 0.5
EPOCHS = 300
SKIP_BY_KICK_WEIGHT = 1.5  # kicking away from a song is an active rejection; weigh it more than a natural skip
LOVE_WEIGHT = 2.0
LOVE_LABEL = 1.0
NEUTRAL_ABOVE = 0.3        # left after this fraction: neither kept nor rejected, so not a training example
KEPT_FROM = 0.8            # played to here or further: kept


@dataclass
class Example:
    embedding: np.ndarray
    label: float               # 0 rejected … 1 kept
    weight: float = 1.0


@dataclass
class TasteModel:
    """P(kept | embedding) = sigmoid(w · e + b), fit by weighted, L2-penalised gradient descent."""

    weights: np.ndarray | None = None
    bias: float = 0.0
    n_examples: int = 0
    history: list[float] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.weights is not None

    def fit(self, examples: list[Example]) -> None:
        """Refit from scratch on the examples; stays (or goes) off when there are too few or they are one-sided."""
        self.weights = None
        self.n_examples = len(examples)
        if len(examples) < MIN_LABELS:
            return
        labels = np.array([example.label for example in examples], dtype=np.float64)
        sample_weights = np.array([example.weight for example in examples], dtype=np.float64)
        kept_mass = float((sample_weights * labels).sum())
        rejected_mass = float((sample_weights * (1.0 - labels)).sum())
        if kept_mass < MIN_CLASS_MASS or rejected_mass < MIN_CLASS_MASS:
            return
        features = np.stack([example.embedding for example in examples]).astype(np.float64)
        self.weights, self.bias = fit_logistic(features, labels, sample_weights)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """P(kept) for each row of embeddings; 0.5 everywhere when the model is off."""
        if self.weights is None:
            return np.full(len(embeddings), 0.5)
        logits = np.asarray(embeddings, dtype=np.float64) @ self.weights + self.bias
        return sigmoid(logits)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def fit_logistic(features: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Full-batch gradient descent on the weighted cross-entropy with an L2 penalty on the weights (not the bias).

    The problem is convex and tiny, so a fixed number of epochs at a fixed rate converges well enough; nothing here
    is tuned to the data."""
    n_rows, n_features = features.shape
    weights = np.zeros(n_features)
    bias = 0.0
    normalised = sample_weights / sample_weights.sum()
    for _ in range(EPOCHS):
        probabilities = sigmoid(features @ weights + bias)
        residual = (probabilities - labels) * normalised
        gradient_weights = features.T @ residual + (L2 / n_rows) * weights
        gradient_bias = float(residual.sum())
        weights -= LEARNING_RATE * gradient_weights
        bias -= LEARNING_RATE * gradient_bias
    return weights, bias


def label_for_play(completion: float | None, *, loved: bool, left_by_kick: bool) -> Example | None:
    """How a logged play becomes a training example, or None when it says nothing about taste.

    Completion is the fraction of the song that played before the listener moved on. Kept to KEPT_FROM or beyond
    is a 1; left before NEUTRAL_ABOVE is a 0 (weighted harder when the listener kicked away); the band in between
    is silence — wanting a change is not the same as disliking the song. A love is a 1 regardless of completion."""
    embedding_placeholder = np.zeros(0)
    if loved:
        return Example(embedding_placeholder, LOVE_LABEL, LOVE_WEIGHT)
    if completion is None:
        return None
    if completion >= KEPT_FROM:
        return Example(embedding_placeholder, 1.0, 1.0)
    if completion < NEUTRAL_ABOVE:
        return Example(embedding_placeholder, 0.0, SKIP_BY_KICK_WEIGHT if left_by_kick else 1.0)
    return None

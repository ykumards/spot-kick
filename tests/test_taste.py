"""Mini-Me's taste model: learns two clusters, stays off on too little or one-sided data."""
import numpy as np

from spotkick.kick import taste


def cluster(seed: int, centre: np.ndarray, n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    points = centre + 0.2 * rng.standard_normal((n, centre.size))
    return [point / np.linalg.norm(point) for point in points]


def test_model_learns_kept_from_rejected_clusters():
    rng = np.random.default_rng(0)
    liked_centre = rng.standard_normal(32)
    disliked_centre = rng.standard_normal(32)
    examples = [taste.Example(vector, 1.0) for vector in cluster(1, liked_centre, 20)]
    examples += [taste.Example(vector, 0.0) for vector in cluster(2, disliked_centre, 20)]
    model = taste.TasteModel()
    model.fit(examples)
    assert model.ready
    probe = np.stack(cluster(3, liked_centre, 5) + cluster(4, disliked_centre, 5))
    kept = model.predict(probe)
    assert kept[:5].min() > 0.6 and kept[5:].max() < 0.4


def test_model_stays_off_on_few_or_one_sided_examples():
    vectors = cluster(5, np.ones(8), taste.MIN_LABELS + 5)
    model = taste.TasteModel()
    model.fit([taste.Example(vector, 1.0) for vector in vectors[:5]])
    assert not model.ready and model.n_examples == 5
    model.fit([taste.Example(vector, 1.0) for vector in vectors])      # enough rows, but nothing rejected
    assert not model.ready
    assert np.all(model.predict(np.stack(vectors[:3])) == 0.5)


def labelled(completion: float | None, *, loved: bool = False, left_by_kick: bool = False) -> taste.Example:
    example = taste.label_for_play(completion, loved=loved, left_by_kick=left_by_kick)
    assert example is not None
    return example


def test_labels_from_completion():
    assert labelled(0.95).label == 1.0
    rejected = labelled(0.02, left_by_kick=True)
    assert rejected.label == 0.0 and rejected.weight == taste.SKIP_BY_KICK_WEIGHT
    assert taste.label_for_play(0.5, loved=False, left_by_kick=False) is None    # wanted a change, not a dislike
    loved = labelled(0.1, loved=True)
    assert loved.label == 1.0 and loved.weight == taste.LOVE_WEIGHT
    assert taste.label_for_play(None, loved=False, left_by_kick=False) is None

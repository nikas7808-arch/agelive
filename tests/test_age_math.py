import numpy as np
from app.age import distribution_stats, softmax, AGES


def test_mean_of_peaked_distribution():
    p = np.exp(-0.5 * ((AGES - 25) / 2) ** 2)
    r = distribution_stats(p)
    assert abs(r.age - 25) < 0.5
    assert r.low <= r.age <= r.high
    assert r.confidence > 0.9


def test_p_over_threshold():
    p = np.exp(-0.5 * ((AGES - 30) / 3) ** 2)
    r = distribution_stats(p)
    assert r.p_over(18) > 0.99
    assert r.p_over(40) < 0.01


def test_softmax_sums_to_one():
    assert abs(softmax(np.random.randn(101)).sum() - 1) < 1e-5

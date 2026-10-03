import math

from src.evaluate import adjacent_accuracy, scores


def test_perfect():
    m = scores([0, 1, 2], [0, 1, 2])
    assert math.isclose(m["macro_f1"], 1.0) and math.isclose(m["accuracy"], 1.0)


def test_macro_punishes_ignoring_rare_class():
    y = [0] * 9 + [1]
    always_major = scores(y, [0] * 10)
    assert math.isclose(always_major["accuracy"], 0.9)
    assert always_major["macro_f1"] < 0.5


def test_adjacent_accuracy_forgives_neighbours():
    assert math.isclose(adjacent_accuracy([0, 1, 2, 3], [1, 1, 3, 3]), 1.0)
    assert math.isclose(adjacent_accuracy([0, 3], [3, 0]), 0.0)

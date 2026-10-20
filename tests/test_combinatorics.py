import numpy as np
import pytest

from generalized_rashomon_set import boundary_grid, feature_subsets, interaction_combos, sum_to_one_tuples
from generalized_rashomon_set.utils import find_all_sum_to_one_pairs


def test_boundary_grid():
    assert np.allclose(boundary_grid(0.1), np.linspace(0, 1, 11))
    assert np.allclose(boundary_grid(0.5), [0, 0.5, 1])
    with pytest.raises(ValueError):
        boundary_grid(0.3)


def test_sum_to_one_tuples_default_grid():
    pairs = sum_to_one_tuples(2, 0.1)
    assert pairs == [(a, 10 - a) for a in range(1, 10)]
    triples = sum_to_one_tuples(3, 0.1)
    assert len(triples) == 36 and all(sum(t) == 10 for t in triples)


def test_sum_to_one_tuples_other_delta_and_radius():
    assert sum_to_one_tuples(2, 0.5) == [(1, 1)]
    assert sum_to_one_tuples(2, 0.25) == [(1, 3), (2, 2), (3, 1)]
    assert sum_to_one_tuples(2, 0.1, radius=0.4) == [(1, 3), (2, 2), (3, 1)]
    with pytest.raises(ValueError):
        sum_to_one_tuples(3, 0.5)  # cannot split 1 into three parts of 0.5
    assert find_all_sum_to_one_pairs(2, 0.5) == [(1, 1)]


def test_interaction_combos_layout():
    combos = interaction_combos(2, 0.1, radii=(0.5, 1.0))
    assert len(combos) == (4 + 9) * 4
    r, a, s = combos[0]
    assert r == 0.5 and a == (1, 4) and s == (0, 0)
    # within a radius: tuple-major, sign-minor
    assert combos[4][1] == (2, 3) and combos[3][2] == (1, 1)


def test_feature_subsets():
    assert feature_subsets([0, 2, 5], 2) == [(0, 2), (0, 5), (2, 5)]

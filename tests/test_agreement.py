"""Agreement statistics against hand-computed values."""

import numpy as np
import pytest

from judgepanel import Panel, cohen_kappa, fleiss_kappa, pairwise_cohen_kappa
from judgepanel.simulate import SimJudge, simulate_panel


def test_cohen_kappa_hand_example() -> None:
    # 100 items: 45 agree on A, 15 agree on B, 25 rated A/B, 15 rated B/A.
    # po = 0.60; marginals 0.70/0.60 for A, 0.30/0.40 for B; pe = 0.54.
    a = ["A"] * 45 + ["B"] * 15 + ["A"] * 25 + ["B"] * 15
    b = ["A"] * 45 + ["B"] * 15 + ["B"] * 25 + ["A"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.06 / 0.46)


def test_cohen_kappa_edges() -> None:
    assert cohen_kappa(["A", "B", "A"], ["A", "B", "A"]) == pytest.approx(1.0)
    assert cohen_kappa(["A", "A"], ["A", "A"]) == pytest.approx(1.0)
    # One rater constant, the other split: observed equals chance, kappa 0.
    assert cohen_kappa(["A", "A"], ["A", "B"]) == pytest.approx(0.0)
    # Disjoint constants: po = 0 and pe = 0, so kappa is 0, not undefined.
    assert cohen_kappa(["A", "A"], ["B", "B"]) == pytest.approx(0.0)
    assert cohen_kappa(["A", "B"], ["B", "A"]) == pytest.approx(-1.0)


def test_cohen_kappa_symmetry_and_validation() -> None:
    a = ["A", "B", "A", "B", "A"]
    b = ["A", "A", "B", "B", "A"]
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa(b, a))
    with pytest.raises(ValueError, match="aligned"):
        cohen_kappa(["A"], ["A", "B"])
    with pytest.raises(ValueError, match="no labels"):
        cohen_kappa([], [])


def test_fleiss_kappa_hand_example() -> None:
    # Vote counts per item over 3 raters: (3,0), (0,3), (2,1), (1,2).
    # Mean pairwise agreement 2/3, chance 1/2, kappa 1/3.
    labels = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype=np.int64)
    panel = Panel(
        labels=labels,
        items=("i0", "i1", "i2", "i3"),
        judges=("a", "b", "c"),
        classes=("no", "yes"),
    )
    assert fleiss_kappa(panel) == pytest.approx(1.0 / 3.0)


def test_fleiss_kappa_skips_single_vote_items() -> None:
    labels = np.array([[0, 0], [1, -1]], dtype=np.int64)
    panel = Panel(labels=labels, items=("i0", "i1"), judges=("a", "b"), classes=("no", "yes"))
    # Item 1 has a single label and is skipped entirely; only item 0 remains,
    # where both raters agree, so the statistic degenerates to perfect agreement.
    assert fleiss_kappa(panel) == pytest.approx(1.0)
    all_single = Panel(
        labels=np.array([[0, -1], [-1, 1]], dtype=np.int64),
        items=("i0", "i1"),
        judges=("a", "b"),
        classes=("no", "yes"),
    )
    with pytest.raises(ValueError, match="2 or more labels"):
        fleiss_kappa(all_single)


def test_pairwise_matrix() -> None:
    labels = np.array(
        [[0, 0, -1], [1, 1, -1], [0, 1, -1], [1, 1, 0]],
        dtype=np.int64,
    )
    panel = Panel(
        labels=labels,
        items=("i0", "i1", "i2", "i3"),
        judges=("a", "b", "c"),
        classes=("no", "yes"),
    )
    matrix = pairwise_cohen_kappa(panel)
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 1.0)
    assert matrix[0, 1] == pytest.approx(matrix[1, 0])
    # Judge c shares only one item with each other judge: not enough to score.
    assert np.isnan(matrix[0, 2]) and np.isnan(matrix[1, 2])


def test_copycat_judges_agree_more() -> None:
    independent = [SimJudge("a", 0.8, 0.85), SimJudge("b", 0.8, 0.85), SimJudge("c", 0.8, 0.85)]
    copycat = [
        SimJudge("a", 0.8, 0.85),
        SimJudge("b", 0.8, 0.85, copies="a", copy_prob=0.8),
        SimJudge("c", 0.8, 0.85),
    ]
    panel_ind, _ = simulate_panel(independent, 4000, prevalence=0.4, seed=71)
    panel_cop, _ = simulate_panel(copycat, 4000, prevalence=0.4, seed=71)
    kappa_ind = pairwise_cohen_kappa(panel_ind)[0, 1]
    kappa_cop = pairwise_cohen_kappa(panel_cop)[0, 1]
    assert kappa_cop > kappa_ind + 0.15

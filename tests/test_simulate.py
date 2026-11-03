"""The simulators must produce what they promise."""

import numpy as np
import pytest

from judgepanel import MISSING, simulate_subtype_panel
from judgepanel.simulate import SimJudge, simulate_confusion_panel, simulate_panel


def test_marginals_match_specification() -> None:
    judge = SimJudge("j", 0.85, 0.90)
    panel, truth = simulate_panel([judge], 30000, prevalence=0.35, seed=3)
    labels = panel.labels[:, 0]
    sens = float(np.mean(labels[truth == 1] == 1))
    spec = float(np.mean(labels[truth == 0] == 0))
    assert abs(sens - 0.85) < 0.01
    assert abs(spec - 0.90) < 0.01
    assert abs(float(truth.mean()) - 0.35) < 0.01


def test_missing_rate() -> None:
    panel, _ = simulate_panel(
        [SimJudge("a", 0.8, 0.8), SimJudge("b", 0.8, 0.8)], 10000, missing=0.25, seed=9
    )
    missing_share = float(np.mean(panel.labels == MISSING))
    assert abs(missing_share - 0.25) < 0.02


def test_copycat_must_come_after_source() -> None:
    bad = [SimJudge("a", 0.8, 0.8, copies="b", copy_prob=0.5), SimJudge("b", 0.8, 0.8)]
    with pytest.raises(ValueError, match="listed earlier"):
        simulate_panel(bad, 100, seed=1)


def test_sim_judge_validation() -> None:
    with pytest.raises(ValueError, match="sensitivity"):
        SimJudge("a", 1.2, 0.5)
    with pytest.raises(ValueError, match="copy_prob"):
        SimJudge("a", 0.8, 0.8, copies="b", copy_prob=2.0)
    with pytest.raises(ValueError, match="no source judge"):
        SimJudge("a", 0.8, 0.8, copy_prob=0.5)
    with pytest.raises(ValueError, match="unique"):
        simulate_panel([SimJudge("a", 0.8, 0.8), SimJudge("a", 0.9, 0.9)], 10, seed=1)
    with pytest.raises(ValueError, match="prevalence"):
        simulate_panel([SimJudge("a", 0.8, 0.8)], 10, prevalence=1.5, seed=1)


def test_confusion_panel_matches_matrices() -> None:
    confusions = np.array(
        [
            [[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]],
            [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]],
        ]
    )
    prior = np.array([0.5, 0.3, 0.2])
    panel, truth = simulate_confusion_panel(confusions, prior, 30000, seed=17)
    for j in range(2):
        for k in range(3):
            rows = panel.labels[truth == k, j]
            for label in range(3):
                share = float(np.mean(rows == label))
                assert abs(share - confusions[j, k, label]) < 0.015


def test_confusion_panel_validation() -> None:
    bad_rows = np.array([[[0.5, 0.4], [0.5, 0.5]]])
    with pytest.raises(ValueError, match="sum to 1"):
        simulate_confusion_panel(bad_rows, np.array([0.5, 0.5]), 10, seed=1)
    good = np.array([[[0.9, 0.1], [0.2, 0.8]]])
    with pytest.raises(ValueError, match="prior length"):
        simulate_confusion_panel(good, np.array([0.5, 0.3, 0.2]), 10, seed=1)
    with pytest.raises(ValueError, match="names must match"):
        simulate_confusion_panel(good, np.array([0.5, 0.5]), 10, seed=1, judge_names=("a", "b"))


def test_subtype_simulator_shapes_and_rates() -> None:
    sens = np.array([[0.9, 0.1], [0.8, 0.8], [0.6, 0.6]])
    spec = np.array([0.95, 0.9, 0.85])
    panel, truth, subtype = simulate_subtype_panel(
        sens, spec, np.array([0.75, 0.25]), 100_000, prevalence=0.5, seed=8
    )
    assert panel.labels.shape == (100_000, 3)
    assert np.all(subtype[truth == 0] == -1)
    assert set(np.unique(subtype[truth == 1])) == {0, 1}
    assert np.mean(subtype[truth == 1] == 1) == pytest.approx(0.25, abs=0.01)
    hidden = (truth == 1) & (subtype == 1)
    assert np.mean(panel.labels[hidden, 0]) == pytest.approx(0.1, abs=0.01)
    assert np.mean(panel.labels[truth == 0, 2]) == pytest.approx(0.15, abs=0.01)


def test_subtype_simulator_validates_inputs() -> None:
    with pytest.raises(ValueError, match="probability vector"):
        simulate_subtype_panel(np.array([[0.9, 0.1]]), np.array([0.9]), np.array([0.5, 0.6]), 10)
    with pytest.raises(ValueError, match="need sensitivity"):
        simulate_subtype_panel(np.array([0.9]), np.array([0.9]), np.array([1.0]), 10)

"""Dawid-Skene estimation: recovery on known ground truth, and its guardrails."""

import numpy as np
import pytest

from judgepanel import MISSING, Panel, SimJudge, fit_dawid_skene, simulate_confusion_panel
from judgepanel.simulate import simulate_panel

FIVE_JUDGES = [
    SimJudge("a", 0.90, 0.95),
    SimJudge("b", 0.80, 0.85),
    SimJudge("c", 0.70, 0.90),
    SimJudge("d", 0.95, 0.60),
    SimJudge("e", 0.60, 0.80),
]


def test_recovers_known_judges() -> None:
    panel, truth = simulate_panel(FIVE_JUDGES, 4000, prevalence=0.3, seed=11)
    result = fit_dawid_skene(panel)
    assert result.converged
    sens = result.sensitivity("faulty")
    spec = result.specificity("faulty")
    for judge in FIVE_JUDGES:
        assert abs(sens[judge.name] - judge.sensitivity) < 0.04
        assert abs(spec[judge.name] - judge.specificity) < 0.04
    assert abs(result.prevalence("faulty") - float(truth.mean())) < 0.03


def test_perfect_judges_recover_truth_exactly() -> None:
    judges = [SimJudge(name, 1.0, 1.0) for name in ("a", "b", "c")]
    panel, truth = simulate_panel(judges, 500, prevalence=0.4, seed=3)
    result = fit_dawid_skene(panel)
    assert np.array_equal(result.hard_labels(), truth)
    for value in result.sensitivity("faulty").values():
        assert value > 0.995
    for value in result.specificity("faulty").values():
        assert value > 0.995


def test_missing_labels_still_recover() -> None:
    panel, truth = simulate_panel(FIVE_JUDGES, 4000, prevalence=0.3, missing=0.3, seed=17)
    result = fit_dawid_skene(panel)
    sens = result.sensitivity("faulty")
    spec = result.specificity("faulty")
    for judge in FIVE_JUDGES:
        assert abs(sens[judge.name] - judge.sensitivity) < 0.06
        assert abs(spec[judge.name] - judge.specificity) < 0.06
    assert abs(result.prevalence("faulty") - float(truth.mean())) < 0.03


def test_three_class_recovery() -> None:
    rng = np.random.default_rng(5)
    n_judges, n_classes = 4, 3
    confusions = np.full((n_judges, n_classes, n_classes), 0.1)
    for j in range(n_judges):
        np.fill_diagonal(confusions[j], 0.8)
    prior = np.array([0.5, 0.3, 0.2])
    panel, truth = simulate_confusion_panel(confusions, prior, 5000, seed=int(rng.integers(1e6)))
    result = fit_dawid_skene(panel)
    estimated_diag = result.confusion.diagonal(axis1=1, axis2=2)
    assert np.max(np.abs(estimated_diag - 0.8)) < 0.05
    assert np.max(np.abs(result.class_prior - prior)) < 0.04
    accuracy = float(np.mean(result.hard_labels() == truth))
    assert accuracy > 0.9


def test_fit_is_deterministic() -> None:
    panel, _ = simulate_panel(FIVE_JUDGES, 800, prevalence=0.3, seed=23)
    first = fit_dawid_skene(panel)
    second = fit_dawid_skene(panel)
    assert np.array_equal(first.posterior, second.posterior)
    assert np.array_equal(first.confusion, second.confusion)
    assert first.log_likelihood == second.log_likelihood


def test_log_likelihood_increases_with_iterations() -> None:
    panel, _ = simulate_panel(FIVE_JUDGES, 600, prevalence=0.4, seed=29)
    previous = -np.inf
    for max_iter in range(1, 9):
        result = fit_dawid_skene(panel, max_iter=max_iter, smoothing=0.001)
        assert result.log_likelihood >= previous - 1e-6
        previous = result.log_likelihood


def test_fewer_than_three_judges_warns() -> None:
    panel, _ = simulate_panel(FIVE_JUDGES[:2], 200, seed=7)
    with pytest.warns(UserWarning, match="weakly identified"):
        fit_dawid_skene(panel)


def test_no_observed_labels_raises() -> None:
    labels = np.full((4, 3), MISSING, dtype=np.int64)
    panel = Panel(
        labels=labels,
        items=("i0", "i1", "i2", "i3"),
        judges=("a", "b", "c"),
        classes=("neg", "pos"),
    )
    with pytest.raises(ValueError, match="no observed labels"):
        fit_dawid_skene(panel)


def test_parameter_validation() -> None:
    panel, _ = simulate_panel(FIVE_JUDGES[:3], 50, seed=1)
    with pytest.raises(ValueError, match="max_iter"):
        fit_dawid_skene(panel, max_iter=0)
    with pytest.raises(ValueError, match="smoothing"):
        fit_dawid_skene(panel, smoothing=0.0)


def test_accuracy_and_binary_helpers() -> None:
    panel, _ = simulate_panel(FIVE_JUDGES[:3], 400, prevalence=0.5, seed=13)
    result = fit_dawid_skene(panel)
    accuracy = result.accuracy("a")
    assert set(accuracy) == {"clean", "faulty"}
    youden = result.youden("faulty")
    sens = result.sensitivity("faulty")
    spec = result.specificity("faulty")
    for name in ("a", "b", "c"):
        assert youden[name] == pytest.approx(sens[name] + spec[name] - 1.0)


def test_binary_helpers_reject_multiclass() -> None:
    confusions = np.full((3, 3, 3), 0.1)
    for j in range(3):
        np.fill_diagonal(confusions[j], 0.8)
    panel, _ = simulate_confusion_panel(confusions, np.array([0.4, 0.3, 0.3]), 300, seed=2)
    result = fit_dawid_skene(panel)
    with pytest.raises(ValueError, match="binary"):
        result.sensitivity("class-0")

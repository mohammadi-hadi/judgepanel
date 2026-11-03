"""Gold-free diagnostics: hand-computed statistics, calibration, and detection."""

import numpy as np
import pytest

from judgepanel import (
    MISSING,
    DawidSkeneResult,
    Panel,
    SimJudge,
    bivariate_residuals,
    dependence_check,
    drop_one_shifts,
    fit_dawid_skene,
    pattern_g2,
    simulate_from_fit,
    simulate_panel,
    tetrads,
)

FIVE = [
    SimJudge("a", 0.90, 0.95),
    SimJudge("b", 0.80, 0.85),
    SimJudge("c", 0.70, 0.90),
    SimJudge("d", 0.85, 0.75),
    SimJudge("e", 0.75, 0.95),
]


def _fixed_fit(labels: np.ndarray, prior: list[float], confusion: np.ndarray) -> tuple:
    n, m = labels.shape
    panel = Panel(
        labels=labels.astype(np.int64),
        items=tuple(f"i{i}" for i in range(n)),
        judges=tuple(f"j{j}" for j in range(m)),
        classes=("no", "yes"),
    )
    fit = DawidSkeneResult(
        classes=panel.classes,
        judges=panel.judges,
        items=panel.items,
        class_prior=np.asarray(prior, dtype=np.float64),
        confusion=np.asarray(confusion, dtype=np.float64),
        posterior=np.full((n, 2), 0.5),
        log_likelihood=0.0,
        n_iter=1,
        converged=True,
    )
    return panel, fit


def test_bivariate_residual_matches_hand_computation() -> None:
    # Two judges, prior (0.5, 0.5), both with accuracy 0.8 on each class.
    # Model table: P(0,0) = P(1,1) = 0.5*0.64 + 0.5*0.04 = 0.34, P(0,1) = P(1,0) = 0.16.
    labels = np.array([[0, 0]] * 4 + [[1, 1]] * 4 + [[0, 1]] * 1 + [[1, 0]] * 1)
    conf = np.array([[[0.8, 0.2], [0.2, 0.8]]] * 2)
    panel, fit = _fixed_fit(labels, [0.5, 0.5], conf)
    expected = 10 * np.array([[0.34, 0.16], [0.16, 0.34]])
    observed = np.array([[4.0, 1.0], [1.0, 4.0]])
    hand = float(np.sum((observed - expected) ** 2 / expected))
    bvr = bivariate_residuals(panel, fit)
    assert bvr[0, 1] == pytest.approx(hand)
    assert bvr[1, 0] == pytest.approx(hand)
    assert np.isnan(bvr[0, 0])


def test_bivariate_residual_uses_co_observed_items_only() -> None:
    labels = np.array([[0, 0], [1, 1], [0, MISSING], [MISSING, 1]])
    conf = np.array([[[0.9, 0.1], [0.1, 0.9]]] * 2)
    panel, fit = _fixed_fit(labels, [0.5, 0.5], conf)
    # Only the first two items count: observed (1, 0, 0, 1) against expected 2 * P.
    p_same = 0.5 * 0.81 + 0.5 * 0.01
    p_diff = 0.5 * 0.09 + 0.5 * 0.09
    expected = 2 * np.array([[p_same, p_diff], [p_diff, p_same]])
    observed = np.array([[1.0, 0.0], [0.0, 1.0]])
    hand = float(np.sum((observed - expected) ** 2 / expected))
    assert bivariate_residuals(panel, fit)[0, 1] == pytest.approx(hand)


def test_pattern_g2_is_zero_when_frequencies_match_the_model() -> None:
    # Three perfect judges: only the patterns 000 and 111 have mass, half each.
    labels = np.array([[0, 0, 0]] * 5 + [[1, 1, 1]] * 5)
    eps = 1e-12
    conf = np.array([[[1 - eps, eps], [eps, 1 - eps]]] * 3)
    panel, fit = _fixed_fit(labels, [0.5, 0.5], conf)
    assert pattern_g2(panel, fit) == pytest.approx(0.0, abs=1e-8)


def test_pattern_g2_matches_hand_computation() -> None:
    labels = np.array([[0, 0, 0]] * 6 + [[1, 1, 1]] * 2 + [[0, 1, 0]] * 2)
    conf = np.array([[[0.9, 0.1], [0.2, 0.8]]] * 3)
    panel, fit = _fixed_fit(labels, [0.6, 0.4], conf)

    def prob(pattern: tuple[int, int, int]) -> float:
        total = 0.0
        for k, pk in enumerate((0.6, 0.4)):
            term = pk
            for value in pattern:
                term *= conf[0, k, value]
            total += term
        return total

    counts = {(0, 0, 0): 6, (1, 1, 1): 2, (0, 1, 0): 2}
    hand = 2 * sum(c * np.log(c / (10 * prob(p))) for p, c in counts.items())
    assert pattern_g2(panel, fit) == pytest.approx(hand)


def test_simulate_from_fit_keeps_shape_missingness_and_seed() -> None:
    panel, _ = simulate_panel(FIVE, 300, prevalence=0.4, missing=0.2, seed=1)
    fit = fit_dawid_skene(panel)
    sim = simulate_from_fit(fit, observed=panel.observed(), seed=5)
    assert sim.labels.shape == panel.labels.shape
    assert np.array_equal(sim.observed(), panel.observed())
    again = simulate_from_fit(fit, observed=panel.observed(), seed=5)
    assert np.array_equal(sim.labels, again.labels)


def test_simulate_from_fit_reproduces_the_fitted_rates() -> None:
    panel, _ = simulate_panel(FIVE, 400, prevalence=0.3, seed=2)
    fit = fit_dawid_skene(panel)
    sim = simulate_from_fit(fit, n_items=200_000, seed=9)
    flag_rate = (sim.labels == 1).mean(axis=0)
    implied = fit.class_prior @ fit.confusion[:, :, 1].T
    assert np.allclose(flag_rate, implied, atol=0.005)


def test_drop_one_shifts_are_small_for_an_independent_panel() -> None:
    panel, _ = simulate_panel(FIVE, 2000, prevalence=0.4, seed=3)
    shifts = drop_one_shifts(panel)
    assert shifts.shape == (5,)
    assert float(shifts.max()) < 0.03


def test_drop_one_shifts_need_four_judges() -> None:
    panel, _ = simulate_panel(FIVE[:3], 100, seed=1)
    with pytest.raises(ValueError, match="at least 4"):
        drop_one_shifts(panel)


def test_tetrads_vanish_for_an_independent_panel_and_skip_constant_judges() -> None:
    judges = [*FIVE, SimJudge("constant", 1.0, 0.0)]
    panel, _ = simulate_panel(judges, 20_000, prevalence=0.4, seed=4)
    result = tetrads(panel)
    assert 5 not in result.judges
    assert result.values.shape == (5, 3)
    assert result.max_abs < 0.01


def test_tetrads_break_for_a_copying_pair() -> None:
    judges = [*FIVE, SimJudge("copy", 0.90, 0.95, copies="a", copy_prob=1.0)]
    panel, _ = simulate_panel(judges, 20_000, prevalence=0.4, seed=4)
    assert tetrads(panel).max_abs > 0.03


def test_tetrads_need_a_binary_panel() -> None:
    labels = np.array([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]], dtype=np.int64)
    panel = Panel(
        labels=labels, items=("a", "b", "c"), judges=tuple("wxyz"), classes=("p", "q", "r")
    )
    with pytest.raises(ValueError, match="binary"):
        tetrads(panel)


def test_dependence_check_is_quiet_on_an_independent_panel() -> None:
    panel, _ = simulate_panel(FIVE, 400, prevalence=0.4, seed=21)
    check = dependence_check(panel, n_boot=39, seed=7)
    assert check.n_boot == 39
    assert check.drop_one_p.shape == (5,)
    for p in (check.g2_p, check.bvr_max_p, check.drop_one_max_p, check.tetrad_p):
        assert p is not None
        assert 1 / 40 <= p <= 1.0
        assert p > 0.05


def test_dependence_check_flags_a_perfect_copy() -> None:
    judges = [*FIVE, SimJudge("copy", 0.90, 0.95, copies="a", copy_prob=1.0)]
    panel, _ = simulate_panel(judges, 400, prevalence=0.4, seed=4)
    check = dependence_check(panel, n_boot=39, seed=3)
    assert check.drop_one_max_p <= 0.05
    assert check.tetrad_p is not None and check.tetrad_p <= 0.05


def test_dependence_check_can_skip_drop_one() -> None:
    panel, _ = simulate_panel(FIVE, 200, seed=2)
    check = dependence_check(panel, n_boot=9, drop_one=False)
    assert np.isnan(check.drop_one_max_p)
    assert np.all(np.isnan(check.drop_one_p))

"""Bootstrap confidence intervals for judge parameters."""

import numpy as np
import pytest

from judgepanel import bootstrap_judges
from judgepanel.simulate import SimJudge, simulate_panel

JUDGES = [
    SimJudge("a", 0.90, 0.95),
    SimJudge("b", 0.80, 0.85),
    SimJudge("c", 0.70, 0.90),
    SimJudge("d", 0.95, 0.60),
    SimJudge("e", 0.60, 0.80),
]


def test_intervals_cover_the_panels_empirical_rates() -> None:
    panel, truth = simulate_panel(JUDGES, 1500, prevalence=0.3, seed=101)
    result = bootstrap_judges(panel, n_boot=100, seed=7, positive="faulty")
    assert result.sensitivity is not None and result.specificity is not None
    covered = 0
    for j, judge in enumerate(JUDGES):
        labels = panel.labels[:, j]
        empirical_sens = float(np.mean(labels[truth == 1] == 1))
        empirical_spec = float(np.mean(labels[truth == 0] == 0))
        lo, hi = result.sensitivity[judge.name]
        covered += int(lo <= empirical_sens <= hi)
        lo, hi = result.specificity[judge.name]
        covered += int(lo <= empirical_spec <= hi)
    # The bootstrap distribution centres on what this panel realised, so the
    # empirical rates should sit inside nearly every interval; the population
    # parameters may drift a sampling error away and are checked loosely below.
    assert covered >= 9
    population_covered = 0
    for judge in JUDGES:
        lo, hi = result.sensitivity[judge.name]
        population_covered += int(lo <= judge.sensitivity <= hi)
        lo, hi = result.specificity[judge.name]
        population_covered += int(lo <= judge.specificity <= hi)
    assert population_covered >= 7


def test_intervals_shrink_with_more_items() -> None:
    small, _ = simulate_panel(JUDGES, 200, prevalence=0.3, seed=5)
    large, _ = simulate_panel(JUDGES, 1600, prevalence=0.3, seed=5)
    ci_small = bootstrap_judges(small, n_boot=80, seed=9, positive="faulty")
    ci_large = bootstrap_judges(large, n_boot=80, seed=9, positive="faulty")
    assert ci_small.sensitivity is not None and ci_large.sensitivity is not None

    def mean_width(intervals: dict[str, tuple[float, float]]) -> float:
        return float(np.mean([hi - lo for lo, hi in intervals.values()]))

    assert mean_width(ci_large.sensitivity) < mean_width(ci_small.sensitivity)


def test_bootstrap_is_deterministic() -> None:
    panel, _ = simulate_panel(JUDGES[:3], 300, seed=31)
    first = bootstrap_judges(panel, n_boot=50, seed=13, positive="faulty")
    second = bootstrap_judges(panel, n_boot=50, seed=13, positive="faulty")
    assert first == second


def test_validation() -> None:
    panel, _ = simulate_panel(JUDGES[:3], 100, seed=1)
    with pytest.raises(ValueError, match="level"):
        bootstrap_judges(panel, level=0.2)
    with pytest.raises(ValueError, match="not one of the panel classes"):
        bootstrap_judges(panel, positive="nope")

"""Gold anchors: the Wilson interval and the audit of a gold-free fit."""

import numpy as np
import pytest

from judgepanel import (
    SimJudge,
    anchor_audit,
    clopper_pearson_interval,
    fit_dawid_skene,
    simulate_panel,
    wilson_interval,
)


def test_wilson_interval_known_values() -> None:
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_validates() -> None:
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
    with pytest.raises(ValueError):
        wilson_interval(1, 2, level=1.5)


def test_clopper_pearson_interval_known_values() -> None:
    # Reference values from the beta-quantile form of the exact interval.
    lo, hi = clopper_pearson_interval(5, 10)
    assert lo == pytest.approx(0.18709, abs=1e-4)
    assert hi == pytest.approx(0.81291, abs=1e-4)
    assert clopper_pearson_interval(0, 10) == (0.0, pytest.approx(0.30850, abs=1e-4))
    assert clopper_pearson_interval(10, 10) == (pytest.approx(0.69150, abs=1e-4), 1.0)
    assert clopper_pearson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        clopper_pearson_interval(3, 2)


def test_clopper_pearson_covers_where_wilson_does_not() -> None:
    # Exact coverage at p = 0.98, n = 20, level 0.995: a Wilson interval
    # undercovers near the boundary, the exact one never does.
    from math import comb

    p, n, level = 0.98, 20, 0.995
    weights = [comb(n, x) * p**x * (1 - p) ** (n - x) for x in range(n + 1)]

    def coverage(interval) -> float:  # type: ignore[no-untyped-def]
        covered = [interval(x, n, level)[0] <= p <= interval(x, n, level)[1] for x in range(n + 1)]
        return sum(w for w, ok in zip(weights, covered, strict=True) if ok)

    assert coverage(clopper_pearson_interval) >= level
    assert coverage(wilson_interval) < level


def _panel_and_gold() -> tuple:
    judges = [SimJudge("a", 0.9, 0.95), SimJudge("b", 0.8, 0.85), SimJudge("c", 0.7, 0.9)]
    panel, truth = simulate_panel(judges, 600, prevalence=0.4, seed=12)
    gold = {item: panel.classes[int(t)] for item, t in zip(panel.items, truth, strict=True)}
    return panel, gold


def test_anchor_audit_counts_and_passes_a_correct_fit() -> None:
    panel, gold = _panel_and_gold()
    fit = fit_dawid_skene(panel)
    audit = anchor_audit(panel, fit, gold)
    assert audit.n_anchor == 600
    assert len(audit.checks) == 3 * 2
    assert audit.n_flagged == 0
    for check in audit.checks:
        assert check.n > 0
        assert check.anchor is not None
        assert check.interval[0] <= check.anchor <= check.interval[1]


def test_anchor_audit_flags_a_wrong_estimate() -> None:
    panel, gold = _panel_and_gold()
    fit = fit_dawid_skene(panel)
    confusion = fit.confusion.copy()
    confusion[0] = [[0.5, 0.5], [0.5, 0.5]]  # judge "a" claimed to be a coin flip
    wrong = type(fit)(**{**fit.__dict__, "confusion": confusion})
    flagged = {(c.judge, c.true_class) for c in anchor_audit(panel, wrong, gold).flagged()}
    assert flagged == {("a", "clean"), ("a", "faulty")}


def test_anchor_audit_ignores_unknown_items_and_rejects_unknown_labels() -> None:
    panel, gold = _panel_and_gold()
    fit = fit_dawid_skene(panel)
    some = {item: gold[item] for item in panel.items[:40]}
    some["not-in-panel"] = "clean"
    assert anchor_audit(panel, fit, some).n_anchor == 40
    with pytest.raises(ValueError, match="not panel classes"):
        anchor_audit(panel, fit, {panel.items[0]: "maybe"})


def test_anchor_audit_with_no_gold_of_a_class_does_not_flag_it() -> None:
    panel, gold = _panel_and_gold()
    fit = fit_dawid_skene(panel)
    clean_only = {item: label for item, label in gold.items() if label == "clean"}
    audit = anchor_audit(panel, fit, clean_only)
    faulty_checks = [c for c in audit.checks if c.true_class == "faulty"]
    assert all(c.n == 0 and c.anchor is None and not c.flagged for c in faulty_checks)
    assert np.isclose(sum(c.n for c in audit.checks if c.judge == "a"), len(clean_only))


def test_rectified_estimates_equal_gold_when_every_item_is_an_anchor() -> None:
    from judgepanel import rectified_estimates

    panel, gold = _panel_and_gold()
    fit = fit_dawid_skene(panel)
    est = rectified_estimates(panel, fit, gold, positive="faulty")
    truth = np.array([gold[item] == "faulty" for item in panel.items])
    flags = panel.labels == panel.classes.index("faulty")
    assert est.prevalence == pytest.approx(truth.mean())
    for j, judge in enumerate(panel.judges):
        assert est.sensitivity[judge] == pytest.approx(flags[truth, j].mean())
        assert est.specificity[judge] == pytest.approx((~flags[~truth, j]).mean())


def test_rectified_estimates_repair_a_wrong_fit_on_average() -> None:
    from judgepanel import rectified_estimates

    panel, gold = _panel_and_gold()
    fit = fit_dawid_skene(panel)
    # Push every posterior towards "clean": the gold-free prevalence is badly off.
    posterior = fit.posterior.copy()
    posterior[:, 1] *= 0.5
    posterior[:, 0] = 1 - posterior[:, 1]
    wrong = type(fit)(**{**fit.__dict__, "posterior": posterior})
    truth = np.array([gold[item] == "faulty" for item in panel.items])
    rng = np.random.default_rng(0)
    prevalences = []
    for _ in range(300):
        chosen = rng.choice(panel.n_items, size=60, replace=False)
        sample = {panel.items[i]: gold[panel.items[i]] for i in chosen}
        prevalences.append(rectified_estimates(panel, wrong, sample, positive="faulty").prevalence)
    assert abs(float(posterior[:, 1].mean()) - truth.mean()) > 0.15
    assert float(np.mean(prevalences)) == pytest.approx(truth.mean(), abs=0.01)


def test_rectified_estimates_validate() -> None:
    from judgepanel import rectified_estimates

    panel, gold = _panel_and_gold()
    fit = fit_dawid_skene(panel)
    with pytest.raises(ValueError, match="positive"):
        rectified_estimates(panel, fit, gold, positive="maybe")
    with pytest.raises(ValueError, match="no gold"):
        rectified_estimates(panel, fit, {"nope": "clean"}, positive="faulty")

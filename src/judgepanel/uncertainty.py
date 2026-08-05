"""Bootstrap uncertainty for Dawid-Skene judge estimates."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from judgepanel.em import fit_dawid_skene
from judgepanel.panel import Panel

Interval = tuple[float, float]


@dataclass(frozen=True)
class BootstrapResult:
    """Percentile confidence intervals from an item-level bootstrap."""

    level: float
    n_boot: int
    positive: str | None
    accuracy: dict[str, dict[str, Interval]]
    prevalence: dict[str, Interval]
    sensitivity: dict[str, Interval] | None
    specificity: dict[str, Interval] | None


def bootstrap_judges(
    panel: Panel,
    *,
    n_boot: int = 200,
    level: float = 0.95,
    seed: int = 20260905,
    positive: str | None = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
    smoothing: float = 0.01,
) -> BootstrapResult:
    """Resample items with replacement and refit Dawid-Skene each time.

    Intervals are percentile intervals over the bootstrap fits: per-judge
    per-class accuracy (the confusion diagonal), class prevalence, and - for
    binary panels with ``positive`` given - sensitivity and specificity.
    """
    if not 0.5 < level < 1.0:
        raise ValueError("level must be between 0.5 and 1")
    if positive is not None and positive not in panel.classes:
        raise ValueError(f"positive class {positive!r} is not one of the panel classes")
    if positive is not None and panel.n_classes != 2:
        raise ValueError("positive is only meaningful for binary panels")
    rng = np.random.default_rng(seed)
    n_judges, n_classes = panel.n_judges, panel.n_classes
    diagonals = np.empty((n_boot, n_judges, n_classes), dtype=np.float64)
    priors = np.empty((n_boot, n_classes), dtype=np.float64)
    for b in range(n_boot):
        rows = rng.integers(0, panel.n_items, size=panel.n_items)
        resampled = Panel(
            labels=panel.labels[rows],
            items=tuple(panel.items[i] for i in rows),
            judges=panel.judges,
            classes=panel.classes,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = fit_dawid_skene(resampled, max_iter=max_iter, tol=tol, smoothing=smoothing)
        diagonals[b] = fit.confusion.diagonal(axis1=1, axis2=2)
        priors[b] = fit.class_prior

    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0

    def interval(samples: np.ndarray) -> Interval:
        return (float(np.quantile(samples, lo_q)), float(np.quantile(samples, hi_q)))

    accuracy = {
        judge: {c: interval(diagonals[:, j, k]) for k, c in enumerate(panel.classes)}
        for j, judge in enumerate(panel.judges)
    }
    prevalence = {c: interval(priors[:, k]) for k, c in enumerate(panel.classes)}
    sensitivity: dict[str, Interval] | None = None
    specificity: dict[str, Interval] | None = None
    if positive is not None:
        pos = panel.classes.index(positive)
        neg_name = panel.classes[1 - pos]
        sensitivity = {judge: accuracy[judge][positive] for judge in panel.judges}
        specificity = {judge: accuracy[judge][neg_name] for judge in panel.judges}
    return BootstrapResult(
        level=level,
        n_boot=n_boot,
        positive=positive,
        accuracy=accuracy,
        prevalence=prevalence,
        sensitivity=sensitivity,
        specificity=specificity,
    )

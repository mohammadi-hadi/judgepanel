"""Audit gold-free estimates against a small set of gold-labeled items.

A gold-free fit is a screening tool: it is cheap and usually close, but a
group of judges that errs together can bias it without any warning from the
fit itself. A few dozen randomly chosen items with known labels are enough to
check it. For every judge and class, the anchor gives a direct estimate of the
judge's accuracy with a Wilson interval; a gold-free estimate outside that
interval is flagged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from judgepanel.em import DawidSkeneResult
from judgepanel.panel import MISSING, Panel


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 1) with no trials."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("need 0 <= successes <= trials")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    if trials == 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


@dataclass(frozen=True)
class AnchorCheck:
    """One judge, one class: gold-free estimate against the anchor."""

    judge: str
    true_class: str
    gold_free: float
    anchor: float | None
    interval: tuple[float, float]
    n: int
    flagged: bool


@dataclass(frozen=True)
class AnchorAudit:
    """Per-judge, per-class comparison of a gold-free fit with an anchor set."""

    level: float
    n_anchor: int
    checks: tuple[AnchorCheck, ...]

    @property
    def n_flagged(self) -> int:
        return sum(check.flagged for check in self.checks)

    def flagged(self) -> list[AnchorCheck]:
        return [check for check in self.checks if check.flagged]


def anchor_audit(
    panel: Panel,
    fit: DawidSkeneResult,
    gold: Mapping[str, str],
    *,
    level: float = 0.95,
) -> AnchorAudit:
    """Compare each judge's gold-free accuracy with its accuracy on gold items.

    ``gold`` maps item names to their true class names; items not in the
    panel are ignored. For judge ``j`` and true class ``k`` the anchor estimate
    is the share of gold items of class ``k`` that ``j`` labeled ``k``, and
    the check is flagged when the fitted ``confusion[j, k, k]`` lies outside
    the Wilson interval. With ``J`` judges and ``K`` classes there are
    ``J * K`` checks, so expect about ``(1 - level) * J * K`` false flags when
    the fit is right; the ``level`` can be raised to control that.
    """
    if panel.judges != fit.judges or panel.classes != fit.classes:
        raise ValueError("the fit was not made on a panel with these judges and classes")
    class_index = {c: k for k, c in enumerate(panel.classes)}
    unknown = {label for label in gold.values() if label not in class_index}
    if unknown:
        raise ValueError(f"gold labels {sorted(unknown)} are not panel classes")
    item_index = {item: i for i, item in enumerate(panel.items)}
    rows = [item_index[item] for item in gold if item in item_index]
    truth = np.array([class_index[gold[panel.items[i]]] for i in rows], dtype=np.int64)
    checks: list[AnchorCheck] = []
    for j, judge in enumerate(panel.judges):
        labels = panel.labels[rows, j] if rows else np.empty(0, dtype=np.int64)
        for k, name in enumerate(panel.classes):
            sel = (truth == k) & (labels != MISSING)
            n = int(sel.sum())
            hits = int(np.sum(labels[sel] == k))
            interval = wilson_interval(hits, n, level)
            estimate = float(fit.confusion[j, k, k])
            checks.append(
                AnchorCheck(
                    judge=judge,
                    true_class=name,
                    gold_free=estimate,
                    anchor=hits / n if n else None,
                    interval=interval,
                    n=n,
                    flagged=n > 0 and not (interval[0] <= estimate <= interval[1]),
                )
            )
    return AnchorAudit(level=level, n_anchor=len(rows), checks=tuple(checks))


@dataclass(frozen=True)
class RectifiedEstimates:
    """Prediction-powered estimates: the gold-free fit, corrected by the anchor.

    Each quantity is written as an expectation over all items, estimated with
    the model's posterior on every item and then corrected by the average
    error of that posterior on the anchor items (Angelopoulos et al., 2023).
    The correction is unbiased whenever the anchor is a random sample, however
    wrong the gold-free fit is; the fit only has to be good for the estimate
    to be precise.
    """

    positive: str
    n_anchor: int
    prevalence: float
    sensitivity: dict[str, float]
    specificity: dict[str, float]


def rectified_estimates(
    panel: Panel,
    fit: DawidSkeneResult,
    gold: Mapping[str, str],
    *,
    positive: str,
) -> RectifiedEstimates:
    """Prevalence, sensitivity and specificity rectified with a random gold anchor.

    For a binary panel, with ``p_i`` the posterior probability that item ``i``
    is ``positive``, ``z_i`` its gold label on the anchor ``A`` and ``y_ij``
    whether judge ``j`` called it positive:

    - prevalence = mean_all(p) + mean_A(z - p)
    - sensitivity_j = [mean(y_j p) + mean_A(y_j (z - p))] / prevalence
    - specificity_j = [mean((1 - y_j)(1 - p)) + mean_A((1 - y_j)(p - z))] / (1 - prevalence)

    with means over the items judge ``j`` labeled. The anchor must be a
    simple random sample of the panel's items for the correction to be unbiased.
    """
    if panel.n_classes != 2:
        raise ValueError("rectified estimates are defined for binary panels")
    if positive not in panel.classes:
        raise ValueError(f"positive class {positive!r} is not one of the panel classes")
    if panel.judges != fit.judges or panel.classes != fit.classes:
        raise ValueError("the fit was not made on a panel with these judges and classes")
    pos = panel.classes.index(positive)
    item_index = {item: i for i, item in enumerate(panel.items)}
    rows = np.array([item_index[item] for item in gold if item in item_index], dtype=np.int64)
    if rows.size == 0:
        raise ValueError("no gold item is in the panel")
    unknown = {gold[panel.items[i]] for i in rows} - set(panel.classes)
    if unknown:
        raise ValueError(f"gold labels {sorted(unknown)} are not panel classes")
    z = np.array([gold[panel.items[i]] == positive for i in rows], dtype=np.float64)
    p = fit.posterior[:, pos]
    prevalence = float(p.mean() + np.mean(z - p[rows]))
    prevalence = min(max(prevalence, 1e-9), 1.0 - 1e-9)
    sens: dict[str, float] = {}
    spec: dict[str, float] = {}
    for j, judge in enumerate(panel.judges):
        seen = panel.labels[:, j] != MISSING
        seen_anchor = seen[rows]
        y = (panel.labels[:, j] == pos).astype(np.float64)
        ya = y[rows][seen_anchor]
        za = z[seen_anchor]
        pa = p[rows][seen_anchor]
        correction_pos = float(np.mean(ya * (za - pa))) if ya.size else 0.0
        correction_neg = float(np.mean((1 - ya) * (pa - za))) if ya.size else 0.0
        joint_pos = float(np.mean(y[seen] * p[seen])) + correction_pos
        joint_neg = float(np.mean((1 - y[seen]) * (1 - p[seen]))) + correction_neg
        sens[judge] = float(np.clip(joint_pos / prevalence, 0.0, 1.0))
        spec[judge] = float(np.clip(joint_neg / (1.0 - prevalence), 0.0, 1.0))
    return RectifiedEstimates(
        positive=positive,
        n_anchor=int(rows.size),
        prevalence=prevalence,
        sensitivity=sens,
        specificity=spec,
    )

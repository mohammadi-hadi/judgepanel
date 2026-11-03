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

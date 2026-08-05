"""Baseline label aggregation by majority vote."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from judgepanel.panel import Panel

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class MajorityResult:
    """Majority-vote labels, with ties made explicit instead of hidden."""

    classes: tuple[str, ...]
    indices: IntArray
    ties: BoolArray

    def labels(self) -> list[str]:
        return [self.classes[k] for k in self.indices]


def majority_vote(panel: Panel, *, tie_break: str = "prevalence") -> MajorityResult:
    """Aggregate a panel by per-item majority, deterministically.

    ``tie_break`` decides tied items (including items with no votes at all):
    ``"prevalence"`` picks the class with the most votes across the whole
    panel, ``"first"`` picks the earliest class in ``panel.classes``, and any
    class name picks that class whenever it is among the tied leaders.
    """
    counts = panel.counts()
    top = counts.max(axis=1)
    is_leader = counts == top[:, None]
    ties = is_leader.sum(axis=1) > 1

    if tie_break == "prevalence":
        totals = counts.sum(axis=0)
        # Stable preference order: global vote count, then class order.
        preference = np.argsort(-totals, kind="stable")
    elif tie_break == "first":
        preference = np.arange(panel.n_classes)
    elif tie_break in panel.classes:
        chosen = panel.classes.index(tie_break)
        preference = np.concatenate(
            (np.array([chosen]), np.delete(np.arange(panel.n_classes), chosen))
        )
    else:
        raise ValueError(f"unknown tie_break {tie_break!r}")

    rank = np.empty(panel.n_classes, dtype=np.int64)
    rank[preference] = np.arange(panel.n_classes)
    # Among tied leaders, take the class ranked highest by the tie-break order.
    ranked = np.where(is_leader, rank[None, :], panel.n_classes)
    indices = np.asarray(np.argmin(ranked, axis=1), dtype=np.int64)
    return MajorityResult(classes=panel.classes, indices=indices, ties=np.asarray(ties))

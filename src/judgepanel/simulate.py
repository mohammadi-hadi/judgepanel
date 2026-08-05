"""Synthetic judge panels with known ground truth, for validation and planning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from judgepanel.panel import MISSING, Panel

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SimJudge:
    """A simulated binary judge, optionally copying another judge's labels.

    ``copies``/``copy_prob`` model correlated judges: with probability
    ``copy_prob`` the judge repeats the named earlier judge's label instead of
    judging independently, which breaks the conditional-independence
    assumption Dawid-Skene relies on.
    """

    name: str
    sensitivity: float
    specificity: float
    copies: str | None = None
    copy_prob: float = 0.0

    def __post_init__(self) -> None:
        for value, what in ((self.sensitivity, "sensitivity"), (self.specificity, "specificity")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{what} must be in [0, 1]")
        if not 0.0 <= self.copy_prob <= 1.0:
            raise ValueError("copy_prob must be in [0, 1]")
        if self.copy_prob > 0.0 and self.copies is None:
            raise ValueError("copy_prob set but no source judge named in copies")


def simulate_panel(
    judges: Sequence[SimJudge],
    n_items: int,
    *,
    prevalence: float = 0.5,
    missing: float = 0.0,
    seed: int | None = None,
    classes: tuple[str, str] = ("clean", "faulty"),
) -> tuple[Panel, IntArray]:
    """Simulate a binary panel; returns the panel and the true class indices.

    Truth is Bernoulli(``prevalence``) on ``classes[1]``. Each judge answers
    correctly with probability ``sensitivity`` on positives and
    ``specificity`` on negatives. Cells then go missing independently with
    probability ``missing``. Copycat judges must name an earlier judge.
    """
    if not judges:
        raise ValueError("at least one judge is required")
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError("prevalence must be in [0, 1]")
    if not 0.0 <= missing < 1.0:
        raise ValueError("missing must be in [0, 1)")
    names = [judge.name for judge in judges]
    if len(set(names)) != len(names):
        raise ValueError("judge names must be unique")
    rng = np.random.default_rng(seed)
    truth = np.asarray(rng.random(n_items) < prevalence, dtype=np.int64)
    labels = np.empty((n_items, len(judges)), dtype=np.int64)
    for j, judge in enumerate(judges):
        correct = np.where(
            truth == 1,
            rng.random(n_items) < judge.sensitivity,
            rng.random(n_items) < judge.specificity,
        )
        own = np.where(correct, truth, 1 - truth)
        if judge.copies is not None:
            if judge.copies not in names[:j]:
                raise ValueError(
                    f"judge {judge.name!r} copies {judge.copies!r}, "
                    "which must be listed earlier in the panel"
                )
            source = labels[:, names.index(judge.copies)]
            copied = rng.random(n_items) < judge.copy_prob
            labels[:, j] = np.where(copied, source, own)
        else:
            labels[:, j] = own
    if missing > 0.0:
        labels[rng.random(labels.shape) < missing] = MISSING
    panel = Panel(
        labels=labels,
        items=tuple(f"item-{i:05d}" for i in range(n_items)),
        judges=tuple(names),
        classes=classes,
    )
    return panel, truth


def simulate_confusion_panel(
    confusions: FloatArray,
    prior: FloatArray,
    n_items: int,
    *,
    missing: float = 0.0,
    seed: int | None = None,
    classes: Sequence[str] | None = None,
    judge_names: Sequence[str] | None = None,
) -> tuple[Panel, IntArray]:
    """Simulate a categorical panel from explicit confusion matrices.

    ``confusions`` has shape (judges, classes, classes); rows are true classes
    and must sum to 1. Returns the panel and the true class indices.
    """
    confusions = np.asarray(confusions, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if confusions.ndim != 3 or confusions.shape[1] != confusions.shape[2]:
        raise ValueError("confusions must have shape (judges, classes, classes)")
    n_judges, n_classes, _ = confusions.shape
    if prior.shape != (n_classes,):
        raise ValueError("prior length must match the number of classes")
    if not np.allclose(confusions.sum(axis=2), 1.0) or not np.isclose(prior.sum(), 1.0):
        raise ValueError("confusion rows and the prior must sum to 1")
    if not 0.0 <= missing < 1.0:
        raise ValueError("missing must be in [0, 1)")
    class_names = (
        tuple(classes) if classes is not None else tuple(f"class-{k}" for k in range(n_classes))
    )
    names = (
        tuple(judge_names)
        if judge_names is not None
        else tuple(f"judge-{j}" for j in range(n_judges))
    )
    if len(class_names) != n_classes or len(names) != n_judges:
        raise ValueError("names must match the confusion dimensions")
    rng = np.random.default_rng(seed)
    truth = rng.choice(n_classes, size=n_items, p=prior).astype(np.int64)
    labels = np.empty((n_items, n_judges), dtype=np.int64)
    for j in range(n_judges):
        cumulative = confusions[j].cumsum(axis=1)
        draws = rng.random(n_items)
        labels[:, j] = (draws[:, None] > cumulative[truth]).sum(axis=1)
    if missing > 0.0:
        labels[rng.random(labels.shape) < missing] = MISSING
    panel = Panel(
        labels=labels,
        items=tuple(f"item-{i:05d}" for i in range(n_items)),
        judges=names,
        classes=class_names,
    )
    return panel, truth

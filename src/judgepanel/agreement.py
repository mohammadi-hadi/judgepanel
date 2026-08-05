"""Chance-corrected agreement between judges."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from judgepanel.panel import MISSING, Panel

FloatArray = npt.NDArray[np.float64]

_EPS = 1e-12


def cohen_kappa(a: Sequence[object], b: Sequence[object]) -> float:
    """Cohen's kappa between two aligned label sequences.

    Returns 1.0 when both raters agree everywhere with no variation, and NaN
    when chance agreement is 1 but observed agreement is not (kappa undefined).
    """
    if len(a) != len(b):
        raise ValueError("sequences must be aligned")
    if not a:
        raise ValueError("no labels")
    labels_a = list(a)
    labels_b = list(b)
    classes = sorted({*labels_a, *labels_b}, key=repr)
    index = {c: k for k, c in enumerate(classes)}
    n = len(labels_a)
    table = np.zeros((len(classes), len(classes)), dtype=np.float64)
    for x, y in zip(labels_a, labels_b, strict=True):
        table[index[x], index[y]] += 1.0
    table /= n
    observed = float(np.trace(table))
    expected = float(table.sum(axis=1) @ table.sum(axis=0))
    if 1.0 - expected < _EPS:
        return 1.0 if 1.0 - observed < _EPS else float("nan")
    return (observed - expected) / (1.0 - expected)


def pairwise_cohen_kappa(panel: Panel) -> FloatArray:
    """Cohen's kappa for every judge pair, on their co-observed items.

    The diagonal is 1. A pair with fewer than 2 co-observed items is NaN.
    """
    out = np.full((panel.n_judges, panel.n_judges), np.nan, dtype=np.float64)
    np.fill_diagonal(out, 1.0)
    labels = panel.labels
    for j in range(panel.n_judges):
        for m in range(j + 1, panel.n_judges):
            both = (labels[:, j] != MISSING) & (labels[:, m] != MISSING)
            if int(both.sum()) < 2:
                continue
            kappa = cohen_kappa(labels[both, j].tolist(), labels[both, m].tolist())
            out[j, m] = out[m, j] = kappa
    return out


def fleiss_kappa(panel: Panel) -> float:
    """Fleiss' kappa over the panel, generalised to a variable rater count.

    Items with fewer than 2 observed labels carry no pairwise-agreement
    information and are skipped.
    """
    counts = panel.counts().astype(np.float64)
    per_item = counts.sum(axis=1)
    usable = per_item >= 2
    if not bool(usable.any()):
        raise ValueError("no item has 2 or more labels")
    counts = counts[usable]
    per_item = per_item[usable]
    agree = ((counts * (counts - 1)).sum(axis=1)) / (per_item * (per_item - 1))
    observed = float(agree.mean())
    shares = counts.sum(axis=0) / counts.sum()
    expected = float(shares @ shares)
    if 1.0 - expected < _EPS:
        return 1.0 if 1.0 - observed < _EPS else float("nan")
    return (observed - expected) / (1.0 - expected)

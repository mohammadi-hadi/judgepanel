"""Gold-free checks on a fitted Dawid-Skene model.

The fit itself cannot tell you whether its central assumption holds: when a
group of judges errs together, maximum likelihood can explain their agreement
by making them accurate, and the residuals of that group then look perfect.
The checks here compare the fit against pieces of itself instead:

- ``bivariate_residuals``: the classical latent-class local-dependence check,
  observed versus model-implied two-way tables for every judge pair.
- ``pattern_g2``: the likelihood-ratio statistic over full response patterns.
- ``drop_one_shifts``: refit without each judge in turn; under the model the
  remaining estimates should barely move.
- ``tetrads``: vanishing-tetrad differences of the label covariance matrix,
  which is rank one off the diagonal for a binary panel that fits the model.

``dependence_check`` puts all four on a common footing with a parametric
bootstrap from the fitted model, so each statistic gets a p-value under the
hypothesis that the fitted model is the truth.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from judgepanel.em import DawidSkeneResult, fit_dawid_skene
from judgepanel.panel import MISSING, Panel

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def _quiet_fit(panel: Panel) -> DawidSkeneResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_dawid_skene(panel)


def simulate_from_fit(
    fit: DawidSkeneResult,
    *,
    n_items: int | None = None,
    observed: BoolArray | None = None,
    seed: int | None = None,
) -> Panel:
    """Draw a new panel from a fitted model (the parametric bootstrap).

    True classes come from the fitted prevalence and each judge answers from
    its fitted confusion matrix, independently given the class. ``observed``
    (items x judges) copies a missingness pattern; without it every cell is
    observed. ``n_items`` defaults to the size of the fitted panel.
    """
    n = len(fit.items) if n_items is None else n_items
    if observed is not None and observed.shape != (n, len(fit.judges)):
        raise ValueError("observed must have shape (n_items, n_judges)")
    rng = np.random.default_rng(seed)
    n_classes = len(fit.classes)
    truth = rng.choice(n_classes, size=n, p=fit.class_prior)
    labels = np.empty((n, len(fit.judges)), dtype=np.int64)
    for j in range(len(fit.judges)):
        cumulative = fit.confusion[j].cumsum(axis=1)
        draws = rng.random(n)
        labels[:, j] = np.minimum((draws[:, None] > cumulative[truth]).sum(axis=1), n_classes - 1)
    if observed is not None:
        labels[~observed] = MISSING
    return Panel(
        labels=labels,
        items=tuple(f"item-{i:05d}" for i in range(n)),
        judges=fit.judges,
        classes=fit.classes,
    )


def bivariate_residuals(panel: Panel, fit: DawidSkeneResult) -> FloatArray:
    """Pearson chi-square of observed vs model-implied tables, per judge pair.

    For judges ``a`` and ``b`` the model implies
    ``P(a=l, b=m) = sum_k prior_k * conf[a,k,l] * conf[b,k,m]``; the residual
    compares that with the observed counts on the items both judges labeled.
    Large values flag local dependence. The diagonal is NaN.
    """
    _check_compatible(panel, fit)
    m = panel.n_judges
    out = np.full((m, m), np.nan, dtype=np.float64)
    labels = panel.labels
    n_classes = panel.n_classes
    for a, b in itertools.combinations(range(m), 2):
        both = (labels[:, a] != MISSING) & (labels[:, b] != MISSING)
        n_both = int(both.sum())
        if n_both == 0:
            continue
        observed = np.zeros((n_classes, n_classes), dtype=np.float64)
        np.add.at(observed, (labels[both, a], labels[both, b]), 1.0)
        expected = (
            np.einsum("k,kl,km->lm", fit.class_prior, fit.confusion[a], fit.confusion[b]) * n_both
        )
        out[a, b] = out[b, a] = float(np.sum((observed - expected) ** 2 / expected))
    return out


def pattern_g2(panel: Panel, fit: DawidSkeneResult) -> float:
    """Likelihood-ratio statistic G2 over the complete response patterns.

    Only items labeled by every judge enter. With many judges most patterns
    are empty and the chi-square reference is useless; calibrate it with
    :func:`dependence_check` instead.
    """
    _check_compatible(panel, fit)
    complete = np.all(panel.labels != MISSING, axis=1)
    if not bool(complete.any()):
        raise ValueError("no item is labeled by every judge")
    rows = panel.labels[complete]
    patterns, counts = np.unique(rows, axis=0, return_counts=True)
    log_prob = np.tile(np.log(fit.class_prior), (patterns.shape[0], 1))
    for j in range(panel.n_judges):
        log_prob += np.log(fit.confusion[j][:, patterns[:, j]]).T
    peak = log_prob.max(axis=1)
    model = np.exp(peak + np.log(np.exp(log_prob - peak[:, None]).sum(axis=1)))
    total = float(counts.sum())
    return float(2.0 * np.sum(counts * np.log(counts / (total * model))))


def drop_one_shifts(panel: Panel, fit: DawidSkeneResult | None = None) -> FloatArray:
    """Largest change in any shared estimate when each judge is left out.

    For each judge ``d`` the panel is refitted without it, and the statistic
    is the maximum absolute change, against the full fit, over the class
    prevalences and every remaining judge's per-class accuracy (the confusion
    diagonal; sensitivity and specificity in a binary panel). Under the model
    both fits estimate the same quantities, so the shift is sampling noise;
    a judge whose removal moves the answer a lot is one the solution hinges on.
    Needs at least four judges, so that every refit keeps three.
    """
    m = panel.n_judges
    if m < 4:
        raise ValueError("drop-one checks need at least 4 judges")
    full = _quiet_fit(panel) if fit is None else fit
    _check_compatible(panel, full)
    full_diag = full.confusion.diagonal(axis1=1, axis2=2)
    out = np.empty(m, dtype=np.float64)
    for d in range(m):
        keep = [j for j in range(m) if j != d]
        reduced = _quiet_fit(
            Panel(
                labels=panel.labels[:, keep],
                items=panel.items,
                judges=tuple(panel.judges[j] for j in keep),
                classes=panel.classes,
            )
        )
        diag = reduced.confusion.diagonal(axis1=1, axis2=2)
        shift = max(
            float(np.max(np.abs(reduced.class_prior - full.class_prior))),
            float(np.max(np.abs(diag - full_diag[keep]))),
        )
        out[d] = shift
    return out


@dataclass(frozen=True)
class Tetrads:
    """Vanishing-tetrad differences for every quadruple of informative judges.

    For ``quads[q] = (i, j, k, l)``, ``values[q]`` holds the three tetrads
    ``c_ij c_kl - c_ik c_jl``, ``c_ij c_kl - c_il c_jk`` and
    ``c_ik c_jl - c_il c_jk`` (``c`` the label covariance). ``judges`` are the
    judge indices that entered (judges whose labels barely vary are left out).
    """

    judges: tuple[int, ...]
    quads: tuple[tuple[int, int, int, int], ...]
    values: FloatArray

    @property
    def max_abs(self) -> float:
        return float(np.max(np.abs(self.values))) if self.values.size else 0.0


def tetrads(panel: Panel, *, min_variance: float = 0.05) -> Tetrads:
    """Tetrad differences of a binary panel's label covariance matrix.

    With labels coded +1/-1, a binary panel that satisfies the model has a
    covariance matrix whose off-diagonal part is rank one (Parisi et al.,
    2014), so every tetrad difference is zero up to sampling noise. A pair of
    judges that errs together inflates its own covariance and breaks the
    tetrads it appears in. Judges whose label variance is below
    ``min_variance`` (near-constant labelers) carry no covariance signal and
    are excluded. Uses items labeled by every judge.
    """
    if panel.n_classes != 2:
        raise ValueError("tetrads are defined for binary panels")
    complete = np.all(panel.labels != MISSING, axis=1)
    signed = 2.0 * panel.labels[complete].astype(np.float64) - 1.0
    if signed.shape[0] < 3:
        raise ValueError("need at least 3 items labeled by every judge")
    cov = np.cov(signed.T)
    informative = tuple(int(j) for j in np.where(np.diag(cov) >= min_variance)[0])
    quads: list[tuple[int, int, int, int]] = []
    values: list[list[float]] = []
    for i, j, k, h in itertools.combinations(informative, 4):
        # The three ways of pairing four judges give three tetrads, two of them
        # independent (t3 = t2 - t1); all three are kept so the max is symmetric.
        t1 = cov[i, j] * cov[k, h] - cov[i, k] * cov[j, h]
        t2 = cov[i, j] * cov[k, h] - cov[i, h] * cov[j, k]
        t3 = cov[i, k] * cov[j, h] - cov[i, h] * cov[j, k]
        quads.append((i, j, k, h))
        values.append([float(t1), float(t2), float(t3)])
    array = np.asarray(values, dtype=np.float64).reshape(len(quads), 3)
    return Tetrads(judges=informative, quads=tuple(quads), values=array)


@dataclass(frozen=True)
class DependenceCheck:
    """Observed diagnostics and their parametric-bootstrap p-values.

    Each p-value is the share of panels simulated from the fitted model whose
    statistic is at least as large as the observed one (with the usual +1
    correction, so the smallest possible value is ``1 / (n_boot + 1)``).
    ``drop_one_p`` holds one p-value per left-out judge and ``drop_one_max_p``
    controls for having looked at all of them.
    """

    judges: tuple[str, ...]
    n_boot: int
    g2: float
    g2_p: float
    bvr: FloatArray
    bvr_max_p: float
    drop_one: FloatArray
    drop_one_p: FloatArray
    drop_one_max_p: float
    tetrad_max: float | None
    tetrad_p: float | None


def _p_value(null: FloatArray, observed: float) -> float:
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + null.size))


def dependence_check(
    panel: Panel,
    *,
    n_boot: int = 200,
    seed: int = 20260905,
    drop_one: bool = True,
    min_variance: float = 0.05,
) -> DependenceCheck:
    """Run every check and calibrate it against the fitted model.

    Each bootstrap replicate draws a panel from the fitted model (keeping the
    observed missingness pattern), refits it, and recomputes every statistic,
    so the reference distributions include the estimation noise of the fit.
    ``drop_one=False`` skips the drop-one refits, which cost one extra fit per
    judge per replicate. Tetrads are computed for binary panels only.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    fit = _quiet_fit(panel)
    observed_mask = panel.observed()
    binary = panel.n_classes == 2

    def stats(p: Panel, f: DawidSkeneResult) -> tuple[float, float, FloatArray, float | None]:
        bvr = bivariate_residuals(p, f)
        shifts = drop_one_shifts(p, f) if drop_one else np.full(p.n_judges, np.nan)
        tet = tetrads(p, min_variance=min_variance).max_abs if binary else None
        return pattern_g2(p, f), float(np.nanmax(bvr)), shifts, tet

    g2, bvr_max, shifts, tet = stats(panel, fit)
    bvr = bivariate_residuals(panel, fit)
    g2_null = np.empty(n_boot)
    bvr_null = np.empty(n_boot)
    shift_null = np.empty((n_boot, panel.n_judges))
    tet_null = np.empty(n_boot)
    for b in range(n_boot):
        sim = simulate_from_fit(fit, observed=observed_mask, seed=seed + b)
        sim_fit = _quiet_fit(sim)
        g2_null[b], bvr_null[b], shift_null[b], t = stats(sim, sim_fit)
        tet_null[b] = np.nan if t is None else t
    if drop_one:
        drop_p = np.array(
            [_p_value(shift_null[:, d], float(shifts[d])) for d in range(panel.n_judges)]
        )
        drop_max_p = _p_value(shift_null.max(axis=1), float(np.max(shifts)))
    else:
        drop_p = np.full(panel.n_judges, np.nan)
        drop_max_p = float("nan")
    return DependenceCheck(
        judges=panel.judges,
        n_boot=n_boot,
        g2=g2,
        g2_p=_p_value(g2_null, g2),
        bvr=bvr,
        bvr_max_p=_p_value(bvr_null, bvr_max),
        drop_one=shifts,
        drop_one_p=drop_p,
        drop_one_max_p=drop_max_p,
        tetrad_max=tet,
        tetrad_p=None if tet is None else _p_value(tet_null, tet),
    )


def _check_compatible(panel: Panel, fit: DawidSkeneResult) -> None:
    if panel.judges != fit.judges or panel.classes != fit.classes:
        raise ValueError("the fit was not made on a panel with these judges and classes")

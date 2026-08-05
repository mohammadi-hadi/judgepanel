"""Dawid-Skene estimation of judge accuracy without gold labels.

Each item has a latent true class; each judge has a confusion matrix giving the
probability of emitting each label conditional on the true class; labels are
conditionally independent across judges given the true class. EM alternates
between a posterior over each item's true class and re-estimating the
confusion matrices from those posteriors (Dawid & Skene, 1979).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from judgepanel.panel import Panel

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class DawidSkeneResult:
    """Fitted Dawid-Skene model for a judge panel.

    ``confusion[j, k, l]`` is the estimated probability that judge ``j`` emits
    label ``l`` when the true class is ``k``. ``posterior[i, k]`` is the
    probability that item ``i`` truly belongs to class ``k``.
    """

    classes: tuple[str, ...]
    judges: tuple[str, ...]
    items: tuple[str, ...]
    class_prior: FloatArray
    confusion: FloatArray
    posterior: FloatArray
    log_likelihood: float
    n_iter: int
    converged: bool

    def hard_labels(self) -> IntArray:
        return np.asarray(np.argmax(self.posterior, axis=1), dtype=np.int64)

    def labels(self) -> list[str]:
        return [self.classes[k] for k in self.hard_labels()]

    def accuracy(self, judge: str) -> dict[str, float]:
        """P(judge labels correctly | true class), per class, for one judge."""
        j = self.judges.index(judge)
        diagonal = np.diag(self.confusion[j])
        return {c: float(diagonal[k]) for k, c in enumerate(self.classes)}

    def _positive_index(self, positive: str) -> int:
        if len(self.classes) != 2:
            raise ValueError("sensitivity/specificity are defined for binary panels only")
        return self.classes.index(positive)

    def sensitivity(self, positive: str) -> dict[str, float]:
        """P(judge says positive | truly positive), per judge (binary panels)."""
        pos = self._positive_index(positive)
        return {name: float(self.confusion[j, pos, pos]) for j, name in enumerate(self.judges)}

    def specificity(self, positive: str) -> dict[str, float]:
        """P(judge says negative | truly negative), per judge (binary panels)."""
        neg = 1 - self._positive_index(positive)
        return {name: float(self.confusion[j, neg, neg]) for j, name in enumerate(self.judges)}

    def youden(self, positive: str) -> dict[str, float]:
        """Youden's J = sensitivity + specificity - 1, per judge (binary panels)."""
        sens = self.sensitivity(positive)
        spec = self.specificity(positive)
        return {name: sens[name] + spec[name] - 1.0 for name in self.judges}

    def prevalence(self, positive: str) -> float:
        return float(self.class_prior[self.classes.index(positive)])


def _logsumexp_rows(a: FloatArray) -> FloatArray:
    m = a.max(axis=1)
    return np.asarray(m + np.log(np.exp(a - m[:, None]).sum(axis=1)), dtype=np.float64)


def _m_step(
    labels: IntArray,
    observed: BoolArray,
    posterior: FloatArray,
    n_classes: int,
    smoothing: float,
) -> tuple[FloatArray, FloatArray]:
    n_items, n_judges = labels.shape
    prior = (posterior.sum(axis=0) + smoothing) / (n_items + n_classes * smoothing)
    confusion = np.empty((n_judges, n_classes, n_classes), dtype=np.float64)
    for j in range(n_judges):
        obs = observed[:, j]
        emitted = labels[obs, j]
        onehot = np.zeros((emitted.shape[0], n_classes), dtype=np.float64)
        onehot[np.arange(emitted.shape[0]), emitted] = 1.0
        counts = posterior[obs].T @ onehot + smoothing
        confusion[j] = counts / counts.sum(axis=1, keepdims=True)
    return np.asarray(prior, dtype=np.float64), confusion


def _e_step(
    labels: IntArray,
    observed: BoolArray,
    prior: FloatArray,
    confusion: FloatArray,
) -> tuple[FloatArray, float]:
    n_items, n_judges = labels.shape
    log_post = np.tile(np.log(prior), (n_items, 1))
    for j in range(n_judges):
        obs = observed[:, j]
        log_conf = np.log(confusion[j])
        log_post[obs] += log_conf[:, labels[obs, j]].T
    norm = _logsumexp_rows(log_post)
    posterior = np.exp(log_post - norm[:, None])
    return posterior, float(norm.sum())


def fit_dawid_skene(
    panel: Panel,
    *,
    max_iter: int = 1000,
    tol: float = 1e-7,
    smoothing: float = 0.01,
) -> DawidSkeneResult:
    """Fit the Dawid-Skene model to a panel by EM.

    The posterior is initialised from soft majority votes, so the latent
    classes start anchored to the observed label names. ``smoothing`` adds
    Laplace pseudo-counts to every confusion cell and the prior, which keeps
    probabilities strictly positive. The fit is deterministic.

    ``tol`` is the per-item change in log-likelihood below which EM stops.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive to keep probabilities finite")
    labels = panel.labels
    n_items, n_judges = labels.shape
    n_classes = panel.n_classes
    if n_judges < 3:
        warnings.warn(
            "Dawid-Skene with fewer than 3 judges is weakly identified; "
            "treat the estimates as rough at best",
            stacklevel=2,
        )
    observed = panel.observed()
    if not bool(observed.any()):
        raise ValueError("panel has no observed labels")

    posterior = np.zeros((n_items, n_classes), dtype=np.float64)
    for k in range(n_classes):
        posterior[:, k] = (labels == k).sum(axis=1)
    votes = posterior.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        posterior = np.where(votes > 0, posterior / votes, 1.0 / n_classes)

    log_likelihood = -np.inf
    converged = False
    n_iter = 0
    prior = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    confusion = np.zeros((n_judges, n_classes, n_classes), dtype=np.float64)
    while n_iter < max_iter:
        n_iter += 1
        prior, confusion = _m_step(labels, observed, posterior, n_classes, smoothing)
        posterior, new_ll = _e_step(labels, observed, prior, confusion)
        if np.isfinite(log_likelihood) and abs(new_ll - log_likelihood) < tol * n_items:
            log_likelihood = new_ll
            converged = True
            break
        log_likelihood = new_ll

    if n_classes == 2:
        mean_diagonal = float(confusion.diagonal(axis1=1, axis2=2).mean())
        if mean_diagonal < 0.5:
            # The latent classes converged onto the mirror solution: relabel them
            # so class k means "the class judges call k more often than not".
            prior = prior[::-1].copy()
            posterior = posterior[:, ::-1].copy()
            confusion = confusion[:, ::-1, :].copy()
            warnings.warn(
                "the fitted panel looked worse than chance on average, which is the "
                "label-swapped solution; classes were flipped back",
                stacklevel=2,
            )

    return DawidSkeneResult(
        classes=panel.classes,
        judges=panel.judges,
        items=panel.items,
        class_prior=prior,
        confusion=confusion,
        posterior=posterior,
        log_likelihood=log_likelihood,
        n_iter=n_iter,
        converged=converged,
    )

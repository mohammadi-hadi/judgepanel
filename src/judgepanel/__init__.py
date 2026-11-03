"""Estimate LLM-judge accuracy without gold labels and aggregate judge panels.

The core is the Dawid-Skene model: several imperfect judges label the same
items, and EM jointly recovers each judge's confusion matrix, the class
prevalence, and a posterior over every item's true label - no gold labels
required. Around it: majority-vote baselines, chance-corrected agreement,
bootstrap confidence intervals, and simulators for planning panels and
studying failure modes such as correlated judges.
"""

from judgepanel.agreement import cohen_kappa, fleiss_kappa, pairwise_cohen_kappa
from judgepanel.anchor import (
    AnchorAudit,
    AnchorCheck,
    RectifiedEstimates,
    anchor_audit,
    rectified_estimates,
    wilson_interval,
)
from judgepanel.diagnostics import (
    DependenceCheck,
    Tetrads,
    bivariate_residuals,
    dependence_check,
    drop_one_shifts,
    pattern_g2,
    simulate_from_fit,
    tetrads,
)
from judgepanel.em import DawidSkeneResult, LocalOptimum, fit_dawid_skene, multistart_dawid_skene
from judgepanel.io import read_panel_jsonl, write_labels_jsonl
from judgepanel.panel import MISSING, Panel
from judgepanel.simulate import (
    SimJudge,
    simulate_confusion_panel,
    simulate_panel,
    simulate_subtype_panel,
)
from judgepanel.uncertainty import BootstrapResult, bootstrap_judges
from judgepanel.vote import MajorityResult, majority_vote

__version__ = "0.1.1"

__all__ = [
    "MISSING",
    "AnchorAudit",
    "AnchorCheck",
    "BootstrapResult",
    "DawidSkeneResult",
    "DependenceCheck",
    "LocalOptimum",
    "MajorityResult",
    "Panel",
    "RectifiedEstimates",
    "SimJudge",
    "Tetrads",
    "__version__",
    "anchor_audit",
    "bivariate_residuals",
    "bootstrap_judges",
    "cohen_kappa",
    "dependence_check",
    "drop_one_shifts",
    "fit_dawid_skene",
    "fleiss_kappa",
    "majority_vote",
    "multistart_dawid_skene",
    "pairwise_cohen_kappa",
    "pattern_g2",
    "read_panel_jsonl",
    "rectified_estimates",
    "simulate_confusion_panel",
    "simulate_from_fit",
    "simulate_panel",
    "simulate_subtype_panel",
    "tetrads",
    "wilson_interval",
    "write_labels_jsonl",
]

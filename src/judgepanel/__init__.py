"""Estimate LLM-judge accuracy without gold labels and aggregate judge panels.

The core is the Dawid-Skene model: several imperfect judges label the same
items, and EM jointly recovers each judge's confusion matrix, the class
prevalence, and a posterior over every item's true label - no gold labels
required. Around it: majority-vote baselines, chance-corrected agreement,
bootstrap confidence intervals, and simulators for planning panels and
studying failure modes such as correlated judges.
"""

from judgepanel.agreement import cohen_kappa, fleiss_kappa, pairwise_cohen_kappa
from judgepanel.em import DawidSkeneResult, fit_dawid_skene
from judgepanel.io import read_panel_jsonl, write_labels_jsonl
from judgepanel.panel import MISSING, Panel
from judgepanel.simulate import SimJudge, simulate_confusion_panel, simulate_panel
from judgepanel.uncertainty import BootstrapResult, bootstrap_judges
from judgepanel.vote import MajorityResult, majority_vote

__version__ = "0.1.0"

__all__ = [
    "MISSING",
    "BootstrapResult",
    "DawidSkeneResult",
    "MajorityResult",
    "Panel",
    "SimJudge",
    "__version__",
    "bootstrap_judges",
    "cohen_kappa",
    "fit_dawid_skene",
    "fleiss_kappa",
    "majority_vote",
    "pairwise_cohen_kappa",
    "read_panel_jsonl",
    "simulate_confusion_panel",
    "simulate_panel",
    "write_labels_jsonl",
]

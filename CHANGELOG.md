# Changelog

## 0.1.0 - 2026-08-05

First release.

- Dawid-Skene EM for categorical judge panels: per-judge confusion matrices,
  class prevalence, item posteriors; missing labels supported; deterministic;
  binary helpers for sensitivity, specificity and Youden's J.
- Majority vote with explicit tie handling.
- Cohen's kappa, Fleiss' kappa (variable rater count), pairwise kappa matrix.
- Item-level bootstrap confidence intervals for judge parameters.
- Panel simulators, including copycat judges for correlated-panel studies.
- Long-format JSONL IO and a `judgepanel` CLI (fit / agree / simulate).
- Validated against a real five-judge panel with construction-time gold labels
  (see README).

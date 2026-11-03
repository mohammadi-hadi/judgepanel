<div align="center">

# judgepanel

[![CI](https://github.com/mohammadi-hadi/judgepanel/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammadi-hadi/judgepanel/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/judgepanel)](https://pypi.org/project/judgepanel/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21809473.svg)](https://doi.org/10.5281/zenodo.21809473)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mohammadi-hadi/judgepanel/blob/main/LICENSE)

*Estimate how accurate your LLM judges are without gold labels. On a real panel of 400 agent trajectories, posterior aggregation ties majority vote on accuracy (0.895 vs 0.897, one item apart).*

</div>

When several imperfect judges label the same items, the disagreement pattern
itself reveals who is reliable. judgepanel fits the classic Dawid–Skene model
(1979) to a judge panel and recovers, jointly:

- each judge's **confusion matrix** (sensitivity/specificity in the binary case),
- the **class prevalence**,
- a **posterior over every item's true label**, weighted by estimated judge
  quality instead of one-judge-one-vote.

Around the core: majority-vote baselines with explicit tie handling,
chance-corrected agreement (Cohen's and Fleiss' kappa), bootstrap confidence
intervals, and simulators for planning panels and studying failure modes.
Pure NumPy; no model calls; every fit deterministic.

## Install

```bash
pip install judgepanel
```

## Sixty seconds

Your data is one JSON object per (item, judge) verdict:

```json
{"item": "case-0017", "judge": "gpt-strict", "label": "faulty"}
```

```bash
judgepanel fit examples/data/demo_panel.jsonl --positive faulty --bootstrap 200
```

```
panel: 300 items, 5 judges, 2 classes (clean, faulty)
converged after 55 iterations
estimated prevalence: clean=0.577, faulty=0.423

judge            sens    spec  youden
gpt-strict      0.906   0.931   0.836
gpt-lenient     0.715   0.999   0.714
small-model     0.692   0.710   0.402
rules           0.562   0.996   0.559
flag-happy      0.953   0.394   0.347
```

The demo panel is synthetic with known truth (`examples/data/demo_gold.jsonl`):
the true parameters are (0.92, 0.94), (0.70, 0.97), (0.66, 0.72), (0.55, 0.99)
and (0.97, 0.35) — every estimate above lands within 0.04 without ever seeing a
gold label, and the flag-everything judge is exposed immediately.

Same thing in Python:

```python
from judgepanel import fit_dawid_skene, read_panel_jsonl

panel = read_panel_jsonl("panel.jsonl")
result = fit_dawid_skene(panel)
result.sensitivity("faulty")   # {"gpt-strict": 0.906, ...}
result.labels()                # posterior-aggregated label per item
```

Keys are configurable, so existing verdict files work directly
(`--item-key trajectory_id --judge-key judge_id --label-key faulty`; boolean
labels become `"true"`/`"false"`).

## Validation on a real judge panel

The [trajectory-judge](https://github.com/mohammadi-hadi/trajectory-judge)
study had five LLM judges label the same 400 agent trajectories as faulty or
clean, with gold labels known by construction. Fitting judgepanel on the
verdicts alone (gold hidden), then unblinding
(`python examples/case_study.py --raw .../results/raw`):

| judge | est sens | true sens | est spec | true spec |
|---|---|---|---|---|
| programmatic rules | 0.771 | 0.667 | 0.993 | 1.000 |
| outcome-only (14B) | 0.639 | 0.613 | 0.633 | 0.670 |
| step-rubric (14B) | 0.995 | 0.857 | 1.000 | 1.000 |
| step-rubric (8B) | 0.988 | 0.990 | 0.000 | 0.000 |
| self-consistency k=3 (14B) | 0.980 | 0.843 | 0.994 | 0.990 |

Three honest findings:

1. **The spammer is exposed exactly.** The 8B judge flags nearly everything;
   its estimated specificity is 0.000 — matching the truth — with no gold
   label in sight. The judge quality *ranking* is right across the board, and
   the production-default outcome judge is estimated within 0.04.
2. **A coalition captures the latent class.** The step judge and its
   self-consistency ensemble (kappa 0.96) are both overestimated by up to
   +0.14, and estimated prevalence drops to 0.645 against a true 0.750. The
   step judge is fitted as nearly perfect, and the fitted prevalence is its
   positive rate. Copying is not the main cause: one fault type (a promise
   invented in the reply) is missed by the rule checker and the 14B pair
   alike, and resampling the judges independently *within fault types*
   reproduces the whole bias. The library prints the kappa matrix, and the
   checks below, so you look before believing the fit.
3. **No aggregation recovers what every judge misses.** Faults that leave the
   final answer correct are invisible to the whole panel, so posterior
   aggregation (0.895 accuracy) cannot beat majority (0.897) here. Weighted
   aggregation wins when judge quality is *uneven*, not when errors are shared.

## When not to trust it

- **Fewer than 3 judges**: weakly identified; the fit warns.
- **Highly correlated judges** (same base model, ensembles of one judge,
  shared blind spots): estimates bias toward the correlated cluster — check
  `judgepanel agree` first, and prefer diverse judge families.
- **A panel that is wrong on average**: if most judges are worse than chance,
  the labels-and-mirror solution is indistinguishable without gold.
- **Tiny or extremely imbalanced panels**: run `judgepanel simulate` with your
  expected operating points to see the estimation error you should expect
  before you rely on the numbers.

## Checking a fit

A fitted judge that looks nearly perfect is a warning sign, not good news:
its residuals with every other judge are zero by construction, so the
classical bivariate-residual check cannot see the problem. Two more things
help.

```python
from judgepanel import anchor_audit, dependence_check, rectified_estimates

check = dependence_check(panel, n_boot=200)   # parametric bootstrap
check.g2_p, check.tetrad_p, check.drop_one_max_p, check.bvr_max_p

gold = {"case-0017": "faulty", ...}           # 20 to 50 random items with known labels
audit = anchor_audit(panel, result, gold, level=1 - 0.05 / (2 * panel.n_judges))
audit.flagged()                                # judge/class estimates the anchor rejects
rectified_estimates(panel, result, gold, positive="faulty")
```

`dependence_check` calibrates four statistics (largest bivariate residual,
pattern G², vanishing tetrads, and the shift when one judge is left out and
the panel refitted) against panels drawn from the fitted model. A rejection
means the panel holds coalitions that disagree about the latent class;
silence does not mean the estimates are right, because a blind spot shared by
the judges that win leaves no trace in the verdicts. A small random gold
anchor does see it, and `rectified_estimates` uses the anchor to correct the
fit (prediction-powered inference).

## API

| call | what it does |
|---|---|
| `read_panel_jsonl(path, ...)` | long-format JSONL → `Panel` |
| `fit_dawid_skene(panel)` | EM fit → confusions, prevalence, posteriors |
| `bootstrap_judges(panel, n_boot=200)` | percentile CIs for judge parameters |
| `majority_vote(panel)` | baseline aggregation, ties made explicit |
| `cohen_kappa`, `fleiss_kappa`, `pairwise_cohen_kappa` | agreement diagnostics |
| `multistart_dawid_skene(panel, n_starts=50)` | EM from random starts, local optima grouped |
| `dependence_check(panel, n_boot=200)` | bootstrap-calibrated checks of conditional independence |
| `bivariate_residuals`, `pattern_g2`, `tetrads`, `drop_one_shifts` | the statistics behind it |
| `anchor_audit(panel, fit, gold)` | Wilson-interval audit of a fit against gold items |
| `rectified_estimates(panel, fit, gold, positive=...)` | prevalence, sensitivity, specificity corrected by a gold anchor |
| `simulate_panel`, `simulate_confusion_panel`, `simulate_subtype_panel` | synthetic panels with known truth |
| `simulate_from_fit(fit)` | panels drawn from a fitted model |

CLI: `judgepanel fit`, `judgepanel agree`, `judgepanel simulate`,
`judgepanel version` (each supports `--json`).

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # 83 tests: hand-computed values, seeded recovery, CLI
.venv/bin/ruff check src tests && .venv/bin/mypy src
```

The suite checks statistical behaviour, not just plumbing: kappa against
hand-computed tables, EM recovery on seeded synthetic panels, log-likelihood
monotonicity, bootstrap coverage, and determinism.

## Part of evalstack

[evalstack](https://github.com/mohammadi-hadi/evalstack) is the map of these
eleven evaluation tools: what each one measures, what it found on real data,
and the two chains that run end to end.

## Citing

The accompanying paper, *Judging Without Gold: Gold-Free Estimation of LLM-Judge Accuracy and Its Failure Modes*, is a working paper (2026).

Releases are archived on Zenodo. Cite the concept DOI
[10.5281/zenodo.21809473](https://doi.org/10.5281/zenodo.21809473), which always
resolves to the latest version; structured metadata is in
[CITATION.cff](https://github.com/mohammadi-hadi/judgepanel/blob/main/CITATION.cff).

## License

MIT — see [LICENSE](https://github.com/mohammadi-hadi/judgepanel/blob/main/LICENSE).

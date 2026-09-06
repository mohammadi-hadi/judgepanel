<div align="center">

# judgepanel

[![CI](https://github.com/mohammadi-hadi/judgepanel/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammadi-hadi/judgepanel/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/judgepanel)](https://pypi.org/project/judgepanel/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21809473.svg)](https://doi.org/10.5281/zenodo.21809473)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Estimate how accurate your LLM judges are — without gold labels — and aggregate their verdicts better than majority vote.*

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
2. **Correlated judges inflate their own estimates.** Self-consistency is the
   step judge sampled three times, and the two agree at kappa 0.96. Dawid–Skene
   assumes conditionally independent judges, so this pair drags the latent
   truth toward its shared verdicts: both get overestimated by up to +0.14,
   and estimated prevalence drops to 0.645 against a true 0.750. The library
   prints the kappa matrix precisely so you check this before believing the fit.
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

## API

| call | what it does |
|---|---|
| `read_panel_jsonl(path, ...)` | long-format JSONL → `Panel` |
| `fit_dawid_skene(panel)` | EM fit → confusions, prevalence, posteriors |
| `bootstrap_judges(panel, n_boot=200)` | percentile CIs for judge parameters |
| `majority_vote(panel)` | baseline aggregation, ties made explicit |
| `cohen_kappa`, `fleiss_kappa`, `pairwise_cohen_kappa` | agreement diagnostics |
| `simulate_panel`, `simulate_confusion_panel` | synthetic panels with known truth |

CLI: `judgepanel fit`, `judgepanel agree`, `judgepanel simulate`,
`judgepanel version` (each supports `--json`).

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # 52 tests: hand-computed values, seeded recovery, CLI
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

Releases are archived on Zenodo. Cite the concept DOI
[10.5281/zenodo.21809473](https://doi.org/10.5281/zenodo.21809473), which always
resolves to the latest version; structured metadata is in
[CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).

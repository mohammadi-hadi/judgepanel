"""Synthetic experiments: recovery grid, correlated-judge bias, label-budget tradeoff.

Deterministic: every simulate/fit pair is seeded from the cell index. Rerunning
this script must produce a byte-identical results/sims.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from judgepanel import SimJudge, fit_dawid_skene, majority_vote, simulate_panel

BASE_SEED = 20260905
R_REPS = 300

ARCHETYPES = [
    (0.90, 0.95),
    (0.80, 0.85),
    (0.70, 0.90),
    (0.95, 0.60),
    (0.60, 0.80),
    (0.85, 0.75),
    (0.75, 0.95),
]


def judges_for(m: int) -> list[SimJudge]:
    points = [ARCHETYPES[i % len(ARCHETYPES)] for i in range(m)]
    return [SimJudge(f"j{i}", s, t) for i, (s, t) in enumerate(points)]


def fit_errors(judges: list[SimJudge], n: int, prevalence: float, seed: int) -> dict[str, float]:
    panel, truth = simulate_panel(judges, n, prevalence=prevalence, seed=seed)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_dawid_skene(panel)
    sens = result.sensitivity("faulty")
    spec = result.specificity("faulty")
    errors = []
    for judge in judges:
        errors.append(abs(sens[judge.name] - judge.sensitivity))
        errors.append(abs(spec[judge.name] - judge.specificity))
    positive = panel.classes.index("faulty")
    posterior_acc = float(np.mean((result.hard_labels() == positive) == (truth == 1)))
    majority_acc = float(np.mean((majority_vote(panel).indices == positive) == (truth == 1)))
    return {
        "mae": float(np.mean(errors)),
        "prev_err": abs(result.prevalence("faulty") - float(truth.mean())),
        "acc_gain": posterior_acc - majority_acc,
    }


def e1_recovery_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cell = 0
    for m in (3, 5, 7):
        judges = judges_for(m)
        for n in (100, 200, 400, 800, 1600):
            for prevalence in (0.1, 0.3, 0.5):
                cell += 1
                stats = [
                    fit_errors(judges, n, prevalence, seed=BASE_SEED + 1000 * cell + r)
                    for r in range(R_REPS)
                ]
                maes = np.array([s["mae"] for s in stats])
                rows.append(
                    {
                        "m": m,
                        "n": n,
                        "prevalence": prevalence,
                        "mae_mean": float(maes.mean()),
                        "mae_p90": float(np.quantile(maes, 0.9)),
                        "prev_err_mean": float(np.mean([s["prev_err"] for s in stats])),
                        "acc_gain_mean": float(np.mean([s["acc_gain"] for s in stats])),
                    }
                )
    return rows


def e2_copycat_bias() -> list[dict[str, object]]:
    """A sixth judge copies the strongest judge with increasing probability."""
    rows: list[dict[str, object]] = []
    source = ARCHETYPES[0]
    for step, copy_prob in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        judges = [SimJudge(f"j{i}", s, t) for i, (s, t) in enumerate(ARCHETYPES[:5])]
        judges.append(
            SimJudge("clone", source[0], source[1], copies="j0", copy_prob=copy_prob)
        )
        source_bias, clone_bias, others_bias, prev_bias = [], [], [], []
        for r in range(R_REPS):
            panel, truth = simulate_panel(
                judges, 400, prevalence=0.75, seed=BASE_SEED + 90_000 + 1000 * step + r
            )
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = fit_dawid_skene(panel)
            sens = result.sensitivity("faulty")
            source_bias.append(sens["j0"] - source[0])
            clone_bias.append(sens["clone"] - source[0])
            others_bias.append(
                float(
                    np.mean(
                        [sens[f"j{i}"] - ARCHETYPES[i][0] for i in range(1, 5)]
                    )
                )
            )
            prev_bias.append(result.prevalence("faulty") - float(truth.mean()))
        rows.append(
            {
                "copy_prob": copy_prob,
                "source_sens_bias": float(np.mean(source_bias)),
                "clone_sens_bias": float(np.mean(clone_bias)),
                "others_sens_bias": float(np.mean(others_bias)),
                "prevalence_bias": float(np.mean(prev_bias)),
            }
        )
    return rows


def e5_label_budget() -> list[dict[str, object]]:
    """Fixed budget of 2,000 labels: many judges on few items, or few on many?"""
    rows: list[dict[str, object]] = []
    for step, m in enumerate((3, 4, 5, 8, 10)):
        judges = judges_for(m)
        n = 2000 // m
        maes = [
            fit_errors(judges, n, 0.3, seed=BASE_SEED + 200_000 + 1000 * step + r)["mae"]
            for r in range(R_REPS)
        ]
        rows.append(
            {
                "m": m,
                "n": n,
                "mae_mean": float(np.mean(maes)),
                "mae_p90": float(np.quantile(np.array(maes), 0.9)),
            }
        )
    return rows


def main() -> None:
    out = Path(__file__).parent / "results" / "sims.json"
    payload = {
        "base_seed": BASE_SEED,
        "reps": R_REPS,
        "archetypes": ARCHETYPES,
        "e1_recovery": e1_recovery_grid(),
        "e2_copycat": e2_copycat_bias(),
        "e5_budget": e5_label_budget(),
    }
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

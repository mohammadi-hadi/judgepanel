"""Real-panel analysis: gold-free fit, unblinded scoring, and the independence counterfactual.

Reads data/verdicts.jsonl (five judges x 400 trajectories) and data/gold.jsonl
(construction-time labels, never shown to the fit). Deterministic; rerunning
must produce a byte-identical results/real.json.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from judgepanel import (
    SimJudge,
    bootstrap_judges,
    fit_dawid_skene,
    majority_vote,
    pairwise_cohen_kappa,
    read_panel_jsonl,
    simulate_panel,
)

SEED = 20260905
BOOT = 2000
COUNTERFACTUAL_REPS = 500
POSITIVE = "true"

DATA = Path(__file__).parent.parent / "data"


def scores(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((predicted == 1) & (truth == 1)))
    fp = int(np.sum((predicted == 1) & (truth == 0)))
    fn = int(np.sum((predicted == 0) & (truth == 1)))
    return {
        "accuracy": float(np.mean(predicted == truth)),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
    }


def main() -> None:
    panel = read_panel_jsonl(
        DATA / "verdicts.jsonl",
        item_key="trajectory_id",
        judge_key="judge_id",
        label_key="faulty",
    )
    gold_rows = [json.loads(line) for line in (DATA / "gold.jsonl").open()]
    faulty_by_item = {r["trajectory_id"]: bool(r["faulty"]) for r in gold_rows}
    silent_by_item = {
        r["trajectory_id"]: bool(r["faulty"]) and bool(r["outcome_correct"]) for r in gold_rows
    }
    truth = np.array([int(faulty_by_item[item]) for item in panel.items], dtype=np.int64)
    silent = np.array([int(silent_by_item[item]) for item in panel.items], dtype=np.int64)
    positive = panel.classes.index(POSITIVE)

    result = fit_dawid_skene(panel)
    boot = bootstrap_judges(panel, n_boot=BOOT, seed=SEED, positive=POSITIVE)
    assert boot.sensitivity is not None and boot.specificity is not None

    judges = []
    all_errors = []
    for j, name in enumerate(panel.judges):
        labels = panel.labels[:, j]
        true_sens = float(np.mean(labels[truth == 1] == positive))
        true_spec = float(np.mean(labels[truth == 0] != positive))
        est_sens = result.sensitivity(POSITIVE)[name]
        est_spec = result.specificity(POSITIVE)[name]
        all_errors += [abs(est_sens - true_sens), abs(est_spec - true_spec)]
        predicted = (labels == positive).astype(np.int64)
        judges.append(
            {
                "judge": name,
                "est_sens": est_sens,
                "true_sens": true_sens,
                "sens_ci": list(boot.sensitivity[name]),
                "est_spec": est_spec,
                "true_spec": true_spec,
                "spec_ci": list(boot.specificity[name]),
                **scores(predicted, truth),
            }
        )

    posterior = (result.hard_labels() == positive).astype(np.int64)
    majority = (majority_vote(panel).indices == positive).astype(np.int64)

    # What the panel collectively misses. LLM detections count all five judges.
    detections = (panel.labels == positive).sum(axis=1)
    faulty_mask = truth == 1
    missed_by_posterior = faulty_mask & (posterior == 0)
    kappa = pairwise_cohen_kappa(panel)
    off_diagonal = kappa[np.triu_indices(panel.n_judges, k=1)]
    max_pair = np.unravel_index(np.nanargmax(kappa - np.eye(panel.n_judges) * 2), kappa.shape)

    # Independence counterfactual: same marginal operating points, no correlation.
    operating = [
        SimJudge(row["judge"], row["true_sens"], row["true_spec"]) for row in judges
    ]
    counterfactual_maes = []
    for r in range(COUNTERFACTUAL_REPS):
        sim_panel, sim_truth = simulate_panel(
            operating, panel.n_items, prevalence=float(truth.mean()), seed=SEED + 500_000 + r
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim_fit = fit_dawid_skene(sim_panel)
        sens = sim_fit.sensitivity("faulty")
        spec = sim_fit.specificity("faulty")
        errors = []
        for judge in operating:
            errors.append(abs(sens[judge.name] - judge.sensitivity))
            errors.append(abs(spec[judge.name] - judge.specificity))
        counterfactual_maes.append(float(np.mean(errors)))
    counterfactual = np.array(counterfactual_maes)
    real_mae = float(np.mean(all_errors))

    payload = {
        "seed": SEED,
        "n_boot": BOOT,
        "n_items": panel.n_items,
        "judges": judges,
        "prevalence_est": result.prevalence(POSITIVE),
        "prevalence_true": float(truth.mean()),
        "aggregation": {
            "posterior": scores(posterior, truth),
            "majority": scores(majority, truth),
        },
        "kappa": {
            "judges": list(panel.judges),
            "matrix": [[float(x) for x in row] for row in kappa],
            "max_pair": [panel.judges[max_pair[0]], panel.judges[max_pair[1]]],
            "max_value": float(np.nanmax(off_diagonal)),
        },
        "blind_spots": {
            "n_faulty": int(faulty_mask.sum()),
            "n_silent": int(silent.sum()),
            "faulty_with_zero_detections": int(np.sum(faulty_mask & (detections == 0))),
            "faulty_missed_by_posterior": int(missed_by_posterior.sum()),
            "silent_share_of_posterior_misses": float(
                silent[missed_by_posterior].mean()
            )
            if missed_by_posterior.any()
            else 0.0,
        },
        "errors": {
            "real_mae": real_mae,
            "real_max": float(np.max(all_errors)),
            "counterfactual_mae_mean": float(counterfactual.mean()),
            "counterfactual_mae_p95": float(np.quantile(counterfactual, 0.95)),
            "counterfactual_mae_max": float(counterfactual.max()),
            "real_exceeds_share": float(np.mean(counterfactual < real_mae)),
            "counterfactual_maes": [round(float(x), 6) for x in counterfactual],
        },
    }
    out = Path(__file__).parent / "results" / "real.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(f"real MAE {real_mae:.3f} vs independent counterfactual p95 "
          f"{payload['errors']['counterfactual_mae_p95']:.3f}")


if __name__ == "__main__":
    main()

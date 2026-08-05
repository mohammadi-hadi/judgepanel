"""Validate gold-free judge estimation on a real 5-judge panel.

The panel comes from the trajectory-judge study (github.com/mohammadi-hadi/
trajectory-judge): five LLM judges each labelled the same 400 agent
trajectories as faulty or clean, and the injected-fault design provides gold
labels. This script fits Dawid-Skene on the judges' verdicts alone - gold
hidden - then scores the estimates against the gold-derived truth.

Usage:
    python examples/case_study.py --raw /path/to/trajectory-judge/results/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from judgepanel import (
    bootstrap_judges,
    fit_dawid_skene,
    majority_vote,
    pairwise_cohen_kappa,
    read_panel_jsonl,
)

POSITIVE = "true"  # the "faulty" verdict


def load_gold(trajectories: Path) -> dict[str, bool]:
    gold: dict[str, bool] = {}
    with trajectories.open() as handle:
        for line in handle:
            row = json.loads(line)
            gold[row["trajectory_id"]] = bool(row["label"]["faulty"])
    return gold


def scores(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    accuracy = float(np.mean(predicted == truth))
    tp = int(np.sum((predicted == 1) & (truth == 1)))
    fp = int(np.sum((predicted == 1) & (truth == 0)))
    fn = int(np.sum((predicted == 0) & (truth == 1)))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return accuracy, f1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="trajectory-judge results/raw directory (verdicts.jsonl + trajectories.jsonl)",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    panel = read_panel_jsonl(
        args.raw / "verdicts.jsonl",
        item_key="trajectory_id",
        judge_key="judge_id",
        label_key="faulty",
    )
    gold_by_item = load_gold(args.raw / "trajectories.jsonl")
    truth = np.array([int(gold_by_item[item]) for item in panel.items], dtype=np.int64)
    positive_index = panel.classes.index(POSITIVE)

    result = fit_dawid_skene(panel)
    boot = bootstrap_judges(panel, n_boot=args.bootstrap, seed=args.seed, positive=POSITIVE)
    assert boot.sensitivity is not None and boot.specificity is not None

    print(f"panel: {panel.n_items} items x {panel.n_judges} judges (gold hidden from the fit)")
    print(
        f"prevalence: estimated {result.prevalence(POSITIVE):.3f}, "
        f"true {float(truth.mean()):.3f}"
    )
    print()

    estimated_sens = result.sensitivity(POSITIVE)
    estimated_spec = result.specificity(POSITIVE)
    width = max(len(name) for name in panel.judges) + 2
    print(
        f"{'judge':<{width}}{'est sens':>9}{'true sens':>10}{'est spec':>9}{'true spec':>10}"
        f"   {'sens 95% CI':>16}"
    )
    max_error = 0.0
    for j, name in enumerate(panel.judges):
        labels = panel.labels[:, j]
        true_sens = float(np.mean(labels[truth == 1] == positive_index))
        true_spec = float(np.mean(labels[truth == 0] != positive_index))
        max_error = max(
            max_error,
            abs(estimated_sens[name] - true_sens),
            abs(estimated_spec[name] - true_spec),
        )
        lo, hi = boot.sensitivity[name]
        print(
            f"{name:<{width}}{estimated_sens[name]:>9.3f}{true_sens:>10.3f}"
            f"{estimated_spec[name]:>9.3f}{true_spec:>10.3f}   [{lo:.3f}, {hi:.3f}]"
        )
    print(f"\nlargest |estimate - truth| across all sens/spec: {max_error:.3f}")
    print()

    posterior = (result.hard_labels() == positive_index).astype(np.int64)
    majority = (majority_vote(panel).indices == positive_index).astype(np.int64)
    print(f"{'aggregation':<{width}}{'accuracy':>9}{'F1':>8}")
    for label, predicted in (("posterior", posterior), ("majority", majority)):
        accuracy, f1 = scores(predicted, truth)
        print(f"{label:<{width}}{accuracy:>9.3f}{f1:>8.3f}")
    for j, name in enumerate(panel.judges):
        predicted = (panel.labels[:, j] == positive_index).astype(np.int64)
        accuracy, f1 = scores(predicted, truth)
        print(f"{name:<{width}}{accuracy:>9.3f}{f1:>8.3f}")
    print()

    print("pairwise Cohen's kappa (correlated judges violate the model's assumption):")
    matrix = pairwise_cohen_kappa(panel)
    print(" " * width + "".join(f"{name.split(':')[0][:9]:>10}" for name in panel.judges))
    for j, name in enumerate(panel.judges):
        row = "".join(f"{matrix[j, m]:>10.3f}" for m in range(panel.n_judges))
        print(f"{name:<{width}}{row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""False alarms and power of a random gold anchor, exact against Wilson intervals.

For every condition of run_diag_sims.py (same panels, same seeds), fit the
panel and audit the fit with random anchors of a few sizes at the Bonferroni
level 1 - 0.05 / (2m). A flag on an independent panel is a false alarm; on a
dependent one it is a detection. Only fits are needed, so this runs in a
minute or two.
"""

from __future__ import annotations

import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from judgepanel import anchor_audit, fit_dawid_skene
from run_diag_sims import ALPHA, BASE_SEED, N_ITEMS, REPS, _conditions, _simulate

SIZES = (10, 20, 50, 100)
WORKERS = 8


def _replicate(task: tuple[int, int]) -> dict[str, dict[str, bool]]:
    ci, r = task
    seed = BASE_SEED + 1_000_000 * (ci + 1) + 1000 * r
    panel, truth, _ = _simulate(_conditions()[ci], seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_dawid_skene(panel)
    level = 1.0 - ALPHA / (2 * panel.n_judges)
    gold = {item: panel.classes[int(t)] for item, t in zip(panel.items, truth, strict=True)}
    rng = np.random.default_rng(seed + 17)
    out: dict[str, dict[str, bool]] = {}
    for k in SIZES:
        chosen = rng.choice(panel.n_items, size=k, replace=False)
        sample = {panel.items[i]: gold[panel.items[i]] for i in chosen}
        out[str(k)] = {
            method: bool(anchor_audit(panel, fit, sample, level=level, method=method).flagged())
            for method in ("exact", "wilson")
        }
    return out


def main() -> None:
    conds = _conditions()
    tasks = [(ci, r) for ci in range(len(conds)) for r in range(REPS)]
    with Pool(WORKERS) as pool:
        results = pool.map(_replicate, tasks, chunksize=8)
    summary = []
    for ci, cond in enumerate(conds):
        reps = results[ci * REPS : (ci + 1) * REPS]
        summary.append({
            "name": cond["name"],
            "flag": {
                method: {str(k): float(np.mean([r[str(k)][method] for r in reps])) for k in SIZES}
                for method in ("exact", "wilson")
            },
        })
    payload = {"seed": BASE_SEED, "reps": REPS, "n_items": N_ITEMS, "alpha": ALPHA, "conditions": summary}
    out = Path(__file__).parent / "results" / "anchor_sims.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")
    for row in summary:
        print(f"{row['name']:30s} exact {row['flag']['exact']} wilson {row['flag']['wilson']}")


if __name__ == "__main__":
    main()

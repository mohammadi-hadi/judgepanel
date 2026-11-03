"""Size and power of the gold-free dependence checks, and the anchor's false-flag rate.

Every replicate simulates a panel, fits it, runs the four bootstrap-calibrated
checks of judgepanel.dependence_check, and audits the fit with random gold
anchors of a few sizes. Conditions cover independent panels (size), a judge
that copies another (copy probability c), a hidden fault subtype that a
coalition of judges misses (share h), and three resampling schemes built from
the real panel. Deterministic: seeds depend only on condition and replicate
index, so the process count does not change the output. The anchor fields here
use Wilson intervals; run_anchor_sims.py compares them with exact intervals.
"""

from __future__ import annotations

import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from judgepanel import (
    Panel,
    SimJudge,
    anchor_audit,
    bivariate_residuals,
    dependence_check,
    fit_dawid_skene,
    read_panel_jsonl,
    simulate_panel,
    simulate_subtype_panel,
)

BASE_SEED = 20260905
REPS = 300
BOOT = 199
N_ITEMS = 400
ANCHOR_SIZES = (20, 50, 100)
ALPHA = 0.05
WORKERS = 14

FIVE = [(0.90, 0.95), (0.80, 0.85), (0.70, 0.90), (0.95, 0.60), (0.60, 0.80)]
DATA = Path(__file__).parent.parent / "data"


def _conditions() -> list[dict[str, object]]:
    conds: list[dict[str, object]] = [
        {"name": "independent_p30", "kind": "independent", "prevalence": 0.30},
        {"name": "independent_p75", "kind": "independent", "prevalence": 0.75},
        {"name": "independent_constant_judge", "kind": "constant", "prevalence": 0.75},
    ]
    for c in (0.25, 0.5, 0.75, 1.0):
        conds.append({"name": f"copy_c{int(c * 100):03d}", "kind": "copy", "c": c, "prevalence": 0.75})
    for h in (0.1, 0.2, 0.3):
        conds.append({"name": f"blind_q2_h{int(h * 100):02d}", "kind": "blind", "h": h, "q": 2, "prevalence": 0.75})
    conds.append({"name": "blind_q3_h20", "kind": "blind", "h": 0.2, "q": 3, "prevalence": 0.75})
    for scheme in ("indep_given_class", "indep_given_type", "pair_joint_given_class"):
        conds.append({"name": f"real_{scheme}", "kind": "real", "scheme": scheme})
    # The regime axis: how many of the five judges share the blind spot (h = 0.2).
    # Appended last so earlier conditions keep their seeds.
    for q in (1, 4, 5):
        conds.append({"name": f"blind_q{q}_h20", "kind": "blind", "h": 0.2, "q": q, "prevalence": 0.75})
    return conds


_REAL: dict[str, object] = {}


def _load_real() -> dict[str, object]:
    if not _REAL:
        panel = read_panel_jsonl(
            DATA / "verdicts.jsonl", item_key="trajectory_id", judge_key="judge_id", label_key="faulty"
        )
        gold = {json.loads(line)["trajectory_id"]: json.loads(line) for line in (DATA / "gold.jsonl").open()}
        truth = np.array([int(gold[i]["faulty"]) for i in panel.items], dtype=np.int64)
        ftype = np.array([gold[i]["failure_type"] or "clean" for i in panel.items])
        # Recode so that 1 means "flagged faulty", matching the synthetic panels.
        pos = panel.classes.index("true")
        labels = (panel.labels == pos).astype(np.int64)
        pair = [panel.judges.index("step:qwen2.5:14b"), panel.judges.index("selfcons3:qwen2.5:14b")]
        _REAL.update(labels=labels, truth=truth, ftype=ftype, pair=pair, judges=panel.judges)
    return _REAL


def _simulate(cond: dict[str, object], seed: int) -> tuple[Panel, np.ndarray, list[int]]:
    """Return a panel (class 1 = positive), its truth, and the dependent judges."""
    kind = cond["kind"]
    prevalence = float(cond.get("prevalence", 0.5))  # type: ignore[arg-type]
    if kind in ("independent", "constant", "copy"):
        points = list(FIVE)
        if kind == "constant":
            points[3] = (0.99, 0.02)
        judges = [SimJudge(f"j{i}", s, t) for i, (s, t) in enumerate(points)]
        dependent: list[int] = []
        if kind == "copy":
            judges.append(SimJudge("copy", 0.90, 0.95, copies="j0", copy_prob=float(cond["c"])))  # type: ignore[arg-type]
            dependent = [0, 5]
        panel, truth = simulate_panel(judges, N_ITEMS, prevalence=prevalence, seed=seed)
        return panel, truth, dependent
    if kind == "blind":
        h = float(cond["h"])  # type: ignore[arg-type]
        q = int(cond["q"])  # type: ignore[call-overload]
        base_sens = np.array([s for s, _ in FIVE])
        spec = np.array([t for _, t in FIVE])
        # Coalition: the q judges with the best Youden index miss the hidden subtype.
        youden = base_sens + spec - 1
        coalition = sorted(int(j) for j in np.argsort(-youden)[:q])
        sens = np.column_stack([base_sens, base_sens])
        sens[coalition, 1] = 0.05
        panel, truth, _ = simulate_subtype_panel(
            sens, spec, np.array([1 - h, h]), N_ITEMS, prevalence=prevalence, seed=seed
        )
        return panel, truth, coalition
    real = _load_real()
    labels = np.asarray(real["labels"])
    truth = np.asarray(real["truth"])
    ftype = np.asarray(real["ftype"])
    pair = list(real["pair"])  # type: ignore[call-overload]
    m = labels.shape[1]
    others = [j for j in range(m) if j not in pair]
    scheme = cond["scheme"]
    if scheme == "indep_given_class":
        strata, blocks, dependent = truth, [[j] for j in range(m)], []
    elif scheme == "indep_given_type":
        strata, blocks, dependent = ftype, [[j] for j in range(m)], pair
    else:
        strata, blocks, dependent = truth, [[j] for j in others] + [pair], pair
    rng = np.random.default_rng(seed)
    out = np.empty_like(labels)
    for s in np.unique(strata):
        rows = np.where(strata == s)[0]
        for cols in blocks:
            donors = rng.choice(rows, size=rows.size, replace=True)
            out[np.ix_(rows, cols)] = labels[np.ix_(donors, cols)]
    panel = Panel(
        labels=out,
        items=tuple(f"item-{i:05d}" for i in range(out.shape[0])),
        judges=tuple(str(j) for j in real["judges"]),  # type: ignore[attr-defined]
        classes=("clean", "faulty"),
    )
    return panel, truth, dependent


def _replicate(task: tuple[int, int]) -> dict[str, object]:
    ci, r = task
    cond = _conditions()[ci]
    seed = BASE_SEED + 1_000_000 * (ci + 1) + 1000 * r
    panel, truth, dependent = _simulate(cond, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_dawid_skene(panel)
    sens_true = (panel.labels[truth == 1] == 1).mean(axis=0)
    spec_true = (panel.labels[truth == 0] == 0).mean(axis=0)
    errors = np.r_[np.abs(fit.confusion[:, 1, 1] - sens_true), np.abs(fit.confusion[:, 0, 0] - spec_true)]
    check = dependence_check(panel, n_boot=BOOT, seed=seed + 7)
    bvr = bivariate_residuals(panel, fit)
    bvr_pair = sorted(int(x) for x in np.unravel_index(np.nanargmax(bvr), bvr.shape))
    top_drop = int(np.argmax(check.drop_one))
    rejected = [int(j) for j in np.where(check.drop_one_p <= ALPHA)[0]]
    rng = np.random.default_rng(seed + 11)
    m = panel.n_judges
    level_fw = 1.0 - ALPHA / (2 * m)
    gold_all = {item: panel.classes[int(t)] for item, t in zip(panel.items, truth, strict=True)}
    anchors = {}
    for k in ANCHOR_SIZES:
        chosen = rng.choice(panel.n_items, size=k, replace=False)
        sample = {panel.items[i]: gold_all[panel.items[i]] for i in chosen}
        anchors[str(k)] = {
            "any": anchor_audit(panel, fit, sample, method="wilson").n_flagged > 0,
            "any_fw": anchor_audit(panel, fit, sample, level=level_fw, method="wilson").n_flagged > 0,
        }
    return {
        "mae": float(errors.mean()),
        "max_err": float(errors.max()),
        "g2_p": check.g2_p,
        "bvr_max_p": check.bvr_max_p,
        "drop_one_max_p": check.drop_one_max_p,
        "tetrad_p": check.tetrad_p,
        "bvr_pair_hits_dependent": bool(dependent) and bvr_pair == sorted(dependent),
        "drop_top_in_dependent": bool(dependent) and top_drop in dependent,
        "drop_rejects_only_dependent": bool(dependent)
        and bool(rejected)
        and all(j in dependent for j in rejected),
        "anchors": anchors,
    }


def _summarise(cond: dict[str, object], reps: list[dict[str, object]]) -> dict[str, object]:
    def rate(key: str) -> float:
        values = [r[key] for r in reps]
        return float(np.mean([v is not None and v <= ALPHA for v in values]))  # type: ignore[operator]

    out: dict[str, object] = {
        **cond,
        "reps": len(reps),
        "mae": float(np.mean([r["mae"] for r in reps])),
        "max_err": float(np.mean([r["max_err"] for r in reps])),
        "reject": {
            "g2": rate("g2_p"),
            "bvr_max": rate("bvr_max_p"),
            "drop_one": rate("drop_one_max_p"),
            "tetrad": rate("tetrad_p"),
        },
        "localise": {
            "bvr_pair_hits_dependent": float(np.mean([r["bvr_pair_hits_dependent"] for r in reps])),
            "drop_top_in_dependent": float(np.mean([r["drop_top_in_dependent"] for r in reps])),
            "drop_rejects_only_dependent": float(np.mean([r["drop_rejects_only_dependent"] for r in reps])),
        },
        "anchor_any_flag": {
            k: float(np.mean([r["anchors"][k]["any"] for r in reps]))  # type: ignore[index]
            for k in map(str, ANCHOR_SIZES)
        },
        "anchor_any_flag_fw": {
            k: float(np.mean([r["anchors"][k]["any_fw"] for r in reps]))  # type: ignore[index]
            for k in map(str, ANCHOR_SIZES)
        },
    }
    return out


def main() -> None:
    conds = _conditions()
    out = Path(__file__).parent / "results" / "diag_sims.json"
    # Conditions already computed with the same settings are kept, not rerun;
    # seeds depend only on the condition index, so this does not change them.
    cached: dict[str, dict[str, object]] = {}
    if out.exists():
        old = json.loads(out.read_text())
        if (old["reps"], old["n_boot"], old["n_items"]) == (REPS, BOOT, N_ITEMS):
            cached = {c["name"]: c for c in old["conditions"]}
    todo = [ci for ci, cond in enumerate(conds) if cond["name"] not in cached]
    tasks = [(ci, r) for ci in todo for r in range(REPS)]
    results: list[dict[str, object]] = []
    if tasks:
        with Pool(WORKERS) as pool:
            results = pool.map(_replicate, tasks, chunksize=4)
    fresh = {
        conds[ci]["name"]: _summarise(conds[ci], results[t * REPS : (t + 1) * REPS])
        for t, ci in enumerate(todo)
    }
    summary = [cached.get(str(c["name"])) or fresh[str(c["name"])] for c in conds]
    payload = {"seed": BASE_SEED, "reps": REPS, "n_boot": BOOT, "n_items": N_ITEMS, "alpha": ALPHA, "conditions": summary}
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")
    for row in summary:
        print(f"{row['name']:32s} mae {row['mae']:.3f} reject {row['reject']} loc {row['localise']}")


if __name__ == "__main__":
    main()

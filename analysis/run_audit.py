"""Real-panel audit: where the gold-free estimates go wrong, why, and what catches it.

Reads the same committed data as run_real.py. Gold labels and fault types are
used only to score and to stratify resampling, never by a gold-free fit.
Deterministic; rerunning must produce a byte-identical results/audit.json.

Sections of the output:
- by_type: each judge's true sensitivity per injected fault type.
- precision: estimated vs true precision per judge (prevalence-dependent).
- multistart: distinct EM optima from random starts.
- drop_one: the full refit without each judge, scored against truth.
- diagnostics: bootstrap-calibrated G2, bivariate residuals, drop-one, tetrads.
- attribution: estimation error under stratified resampling schemes that keep
  or break specific parts of the panel's dependence.
- estimators: other ways to estimate judge accuracy from the same verdicts.
- anchors: how a small random gold sample audits the gold-free fit.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

import itertools

from judgepanel import (
    Panel,
    anchor_audit,
    rectified_estimates,
    bivariate_residuals,
    dependence_check,
    fit_dawid_skene,
    majority_vote,
    multistart_dawid_skene,
    read_panel_jsonl,
    tetrads,
)

SEED = 20260905
DIAG_BOOT = 999
ATTRIBUTION_REPS = 500
ANCHOR_SIZES = (10, 20, 30, 50, 75, 100)
ANCHOR_DRAWS = 2000
LCA_STARTS = 200
NULL_REPS = 300
POSITIVE = "true"
PAIR = ("step:qwen2.5:14b", "selfcons3:qwen2.5:14b")

DATA = Path(__file__).parent.parent / "data"


def quiet_fit(panel: Panel):  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_dawid_skene(panel)


def subpanel(panel: Panel, keep: list[int], labels: np.ndarray | None = None) -> Panel:
    source = panel.labels if labels is None else labels
    return Panel(
        labels=source[:, keep],
        items=panel.items,
        judges=tuple(panel.judges[j] for j in keep),
        classes=panel.classes,
    )


def quiet_fit_init(panel: Panel, init: np.ndarray):  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_dawid_skene(panel, init=init)


def true_rates(labels: np.ndarray, truth: np.ndarray, pos: int) -> tuple[np.ndarray, np.ndarray]:
    sens = (labels[truth == 1] == pos).mean(axis=0)
    spec = (labels[truth == 0] != pos).mean(axis=0)
    return sens, spec


def fit_rates(fit, pos: int) -> tuple[np.ndarray, np.ndarray, float]:  # type: ignore[no-untyped-def]
    neg = 1 - pos
    return fit.confusion[:, pos, pos], fit.confusion[:, neg, neg], float(fit.class_prior[pos])


def mae(est_sens, est_spec, sens, spec) -> float:  # type: ignore[no-untyped-def]
    return float(np.mean(np.r_[np.abs(est_sens - sens), np.abs(est_spec - spec)]))


def precision(sens: np.ndarray, spec: np.ndarray, prev: float) -> np.ndarray:
    flagged = sens * prev + (1 - spec) * (1 - prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(flagged > 0, sens * prev / flagged, np.nan)


def lca_binary(y: np.ndarray, n_classes: int, n_starts: int, seed: int) -> dict[str, object]:
    """Latent class model with ``n_classes`` classes for binary labels (best of many starts)."""
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    best: dict[str, object] | None = None
    for _ in range(n_starts):
        post = rng.dirichlet(np.ones(n_classes), size=n)
        old = -np.inf
        for _ in range(5000):
            prior = (post.sum(0) + 0.01) / (n + n_classes * 0.01)
            flag = (post.T @ y + 0.01) / (post.sum(0)[:, None] + 0.02)
            log_post = np.log(prior)[None, :] + y @ np.log(flag).T + (1 - y) @ np.log(1 - flag).T
            peak = log_post.max(1, keepdims=True)
            ll = float((peak[:, 0] + np.log(np.exp(log_post - peak).sum(1))).sum())
            post = np.exp(log_post - peak)
            post /= post.sum(1, keepdims=True)
            if abs(ll - old) < 1e-10 * n:
                break
            old = ll
        if best is None or ll > float(best["ll"]):  # type: ignore[arg-type]
            best = {"ll": ll, "prior": prior, "flag": flag, "post": post}
    assert best is not None
    return best


def pair_joint_fit(y: np.ndarray, pair: list[int], n_starts: int, seed: int) -> dict[str, object]:
    """Two-class latent model in which the two judges of ``pair`` are one joint indicator.

    The pair's four verdict combinations get their own distribution in each
    class, so any dependence between the two given the class is modelled; the
    other judges stay conditionally independent. Best of ``n_starts`` random
    starts, with the same 0.01 pseudo-count as the library's EM.
    """
    rng = np.random.default_rng(seed)
    n, m = y.shape
    others = [j for j in range(m) if j not in pair]
    joint = (2 * y[:, pair[0]] + y[:, pair[1]]).astype(np.int64)
    onehot = np.eye(4)[joint]
    best: dict[str, object] | None = None
    for _ in range(n_starts):
        post = rng.dirichlet(np.ones(2), size=n)
        old = -np.inf
        for _ in range(5000):
            prior = (post.sum(0) + 0.01) / (n + 0.02)
            flag = (post.T @ y[:, others] + 0.01) / (post.sum(0)[:, None] + 0.02)
            cells = (post.T @ onehot + 0.01) / (post.sum(0)[:, None] + 0.04)
            log_post = (
                np.log(prior)[None, :]
                + y[:, others] @ np.log(flag).T
                + (1 - y[:, others]) @ np.log(1 - flag).T
                + onehot @ np.log(cells).T
            )
            peak = log_post.max(1, keepdims=True)
            ll = float((peak[:, 0] + np.log(np.exp(log_post - peak).sum(1))).sum())
            post = np.exp(log_post - peak)
            post /= post.sum(1, keepdims=True)
            if abs(ll - old) < 1e-10 * n:
                break
            old = ll
        if best is None or ll > float(best["ll"]):  # type: ignore[arg-type]
            best = {"ll": ll, "prior": prior, "flag": flag, "cells": cells}
    assert best is not None
    flag, cells, prior = np.asarray(best["flag"]), np.asarray(best["cells"]), np.asarray(best["prior"])
    # Per-judge flag rates in each class; the pair's from the joint cells (10 = first, 01 = second).
    rates = np.zeros((2, m))
    rates[:, others] = flag
    rates[:, pair[0]] = cells[:, 2] + cells[:, 3]
    rates[:, pair[1]] = cells[:, 1] + cells[:, 3]
    # Orient: the faulty class is the one the panel flags more on average.
    faulty = int(np.argmax(rates.mean(axis=1)))
    clean = 1 - faulty
    return {
        "sens": rates[faulty].tolist(),
        "spec": (1 - rates[clean]).tolist(),
        "prevalence": float(prior[faulty]),
        "log_likelihood": float(best["ll"]),
    }


def main() -> None:
    panel = read_panel_jsonl(
        DATA / "verdicts.jsonl",
        item_key="trajectory_id",
        judge_key="judge_id",
        label_key="faulty",
    )
    gold_rows = {json.loads(line)["trajectory_id"]: json.loads(line) for line in (DATA / "gold.jsonl").open()}
    truth = np.array([int(gold_rows[i]["faulty"]) for i in panel.items], dtype=np.int64)
    ftype = np.array([gold_rows[i]["failure_type"] or "clean" for i in panel.items])
    silent = np.array([int(gold_rows[i]["faulty"] and gold_rows[i]["outcome_correct"]) for i in panel.items])
    pos = panel.classes.index(POSITIVE)
    labels = panel.labels
    m = panel.n_judges
    judges = list(panel.judges)
    pair_idx = [judges.index(p) for p in PAIR]

    full = quiet_fit(panel)
    est_sens, est_spec, est_prev = fit_rates(full, pos)
    sens, spec = true_rates(labels, truth, pos)
    prev = float(truth.mean())

    # By fault type.
    types = sorted(set(ftype) - {"clean"})
    by_type = {
        "types": types,
        "n_per_type": {t: int(np.sum(ftype == t)) for t in types},
        "sensitivity": {
            judges[j]: {t: float(np.mean(labels[ftype == t, j] == pos)) for t in types} for j in range(m)
        },
        "silent": {judges[j]: float(np.mean(labels[silent == 1, j] == pos)) for j in range(m)},
        "loud": {
            judges[j]: float(np.mean(labels[(truth == 1) & (silent == 0), j] == pos)) for j in range(m)
        },
    }
    # Sensitivity on the faults outside the type the dominant pair misses.
    visible = (truth == 1) & (ftype != "unsupported_claim")
    by_type["sens_excluding_unsupported_claim"] = {
        judges[j]: float(np.mean(labels[visible, j] == pos)) for j in range(m)
    }
    posterior_faulty = full.hard_labels() == pos
    missed = (truth == 1) & ~posterior_faulty
    by_type["posterior_misses_by_type"] = {t: int(np.sum(missed & (ftype == t))) for t in types}

    # Precision.
    precision_block = {
        "estimated": dict(zip(judges, precision(est_sens, est_spec, est_prev).tolist(), strict=True)),
        "true": dict(zip(judges, precision(sens, spec, prev).tolist(), strict=True)),
    }

    # Multi-start EM.
    optima = multistart_dawid_skene(panel, n_starts=200, seed=SEED)
    multistart = [
        {"log_likelihood": o.fit.log_likelihood, "count": o.count, "prevalence": float(o.fit.class_prior[pos])}
        for o in optima
    ]
    # Starts far from the uninformative point: each judge's own labelling (the
    # captures that sub-panels reach), and random hard partitions of the items.
    best_ll = optima[0].fit.log_likelihood
    targeted = []
    for j in range(m):
        init = np.full((panel.n_items, panel.n_classes), 0.01)
        init[np.arange(panel.n_items), np.clip(labels[:, j], 0, panel.n_classes - 1)] = 0.99
        init /= init.sum(1, keepdims=True)
        targeted.append(quiet_fit_init(panel, init))
    start_rng = np.random.default_rng(SEED + 9)
    for _ in range(100):
        hard = start_rng.integers(0, panel.n_classes, panel.n_items)
        init = np.full((panel.n_items, panel.n_classes), 0.01)
        init[np.arange(panel.n_items), hard] = 0.99
        init /= init.sum(1, keepdims=True)
        targeted.append(quiet_fit_init(panel, init))
    multistart_targeted = {
        "n_starts": len(targeted),
        "n_same_optimum": int(sum(abs(f.log_likelihood - best_ll) < 1e-3 for f in targeted)),
        "best_other_ll_gap": float(min((best_ll - f.log_likelihood for f in targeted
                                        if abs(f.log_likelihood - best_ll) >= 1e-3), default=0.0)),
        "any_higher": bool(any(f.log_likelihood > best_ll + 1e-3 for f in targeted)),
    }

    # Drop-one refits scored against truth.
    drop_one = {}
    for d in range(m):
        keep = [j for j in range(m) if j != d]
        f = quiet_fit(subpanel(panel, keep))
        s_hat, p_hat, prev_hat = fit_rates(f, pos)
        drop_one[judges[d]] = {
            "prevalence": prev_hat,
            "mae": mae(s_hat, p_hat, sens[keep], spec[keep]),
            "sens": dict(zip([judges[j] for j in keep], s_hat.tolist(), strict=True)),
            "spec": dict(zip([judges[j] for j in keep], p_hat.tolist(), strict=True)),
        }

    # Bootstrap-calibrated diagnostics.
    check = dependence_check(panel, n_boot=DIAG_BOOT, seed=SEED + 1)
    tet = tetrads(panel)
    bvr = bivariate_residuals(panel, full)
    diagnostics = {
        "n_boot": DIAG_BOOT,
        "g2": check.g2,
        "g2_p": check.g2_p,
        "bvr": [[None if np.isnan(x) else float(x) for x in row] for row in bvr],
        "bvr_max": float(np.nanmax(bvr)),
        "bvr_max_pair": [judges[k] for k in np.unravel_index(np.nanargmax(bvr), bvr.shape)],
        "bvr_pair_of_interest": float(bvr[pair_idx[0], pair_idx[1]]),
        "bvr_max_p": check.bvr_max_p,
        "drop_one_shift": dict(zip(judges, check.drop_one.tolist(), strict=True)),
        "drop_one_p": dict(zip(judges, check.drop_one_p.tolist(), strict=True)),
        "drop_one_max_p": check.drop_one_max_p,
        "tetrad_judges": [judges[j] for j in tet.judges],
        "tetrads": tet.values.tolist(),
        "tetrad_max": check.tetrad_max,
        "tetrad_p": check.tetrad_p,
    }

    # Attribution by stratified resampling. Each scheme draws every column block
    # from a random donor row within the item's stratum: blocks drawn together
    # keep their joint behaviour, separate blocks are made independent.
    rng = np.random.default_rng(SEED + 2)
    others = [j for j in range(m) if j not in pair_idx]
    schemes = {
        "indep_given_class": (truth, [[j] for j in range(m)]),
        "pair_joint_given_class": (truth, [[j] for j in others] + [pair_idx]),
        "indep_given_type": (ftype, [[j] for j in range(m)]),
        "pair_joint_given_type": (ftype, [[j] for j in others] + [pair_idx]),
        "all_joint_given_type": (ftype, [list(range(m))]),
        "item_bootstrap": (truth, [list(range(m))]),
    }
    attribution = {}
    for name, (strata, blocks) in schemes.items():
        maes, pair_bias, rules_bias, prev_bias = [], [], [], []
        for _ in range(ATTRIBUTION_REPS):
            resampled = np.empty_like(labels)
            for s in np.unique(strata):
                rows = np.where(strata == s)[0]
                for cols in blocks:
                    donors = rng.choice(rows, size=rows.size, replace=True)
                    resampled[np.ix_(rows, cols)] = labels[np.ix_(donors, cols)]
            f = quiet_fit(subpanel(panel, list(range(m)), resampled))
            s_hat, p_hat, prev_hat = fit_rates(f, pos)
            s_true, p_true = true_rates(resampled, truth, pos)
            maes.append(mae(s_hat, p_hat, s_true, p_true))
            pair_bias.append(float(np.mean((s_hat - s_true)[pair_idx])))
            rules_bias.append(float(s_hat[0] - s_true[0]))
            prev_bias.append(prev_hat - prev)
        attribution[name] = {
            "mae_mean": float(np.mean(maes)),
            "mae_p05": float(np.quantile(maes, 0.05)),
            "mae_p95": float(np.quantile(maes, 0.95)),
            "pair_sens_bias": float(np.mean(pair_bias)),
            "rules_sens_bias": float(np.mean(rules_bias)),
            "prev_bias": float(np.mean(prev_bias)),
        }

    # Other estimators on the same verdicts.
    estimators = {}
    estimators["dawid_skene"] = {"sens": est_sens.tolist(), "spec": est_spec.tolist(), "prevalence": est_prev}
    majority = majority_vote(panel).indices == pos
    estimators["agreement_with_majority"] = {
        "sens": [float(np.mean(labels[majority, j] == pos)) for j in range(m)],
        "spec": [float(np.mean(labels[~majority, j] != pos)) for j in range(m)],
        "prevalence": float(majority.mean()),
    }
    for drop in PAIR:
        d = judges.index(drop)
        keep = [j for j in range(m) if j != d]
        f = quiet_fit(subpanel(panel, keep))
        s_hat, p_hat, prev_hat = fit_rates(f, pos)
        s_full = np.full(m, np.nan)
        p_full = np.full(m, np.nan)
        s_full[keep] = s_hat
        p_full[keep] = p_hat
        estimators[f"dawid_skene_without_{drop}"] = {
            "sens": [None if np.isnan(x) else float(x) for x in s_full],
            "spec": [None if np.isnan(x) else float(x) for x in p_full],
            "prevalence": prev_hat,
        }
    y = (labels == pos).astype(np.float64)
    estimators["pair_joint"] = pair_joint_fit(y, pair_idx, LCA_STARTS, SEED + 5)
    lca3 = lca_binary(y, 3, LCA_STARTS, SEED + 3)
    post3 = np.asarray(lca3["post"])
    flag3 = np.asarray(lca3["flag"])
    prior3 = np.asarray(lca3["prior"])
    # The class with the lowest mean flag rate among the informative judges is "clean".
    informative = [j for j in range(m) if np.var(y[:, j]) >= 0.05]
    clean_class = int(np.argmin(flag3[:, informative].mean(axis=1)))
    faulty_classes = [k for k in range(3) if k != clean_class]
    w = prior3[faulty_classes] / prior3[faulty_classes].sum()
    estimators["latent_class_3"] = {
        "sens": (w @ flag3[faulty_classes]).tolist(),
        "spec": (1 - flag3[clean_class]).tolist(),
        "prevalence": float(prior3[faulty_classes].sum()),
        "log_likelihood": float(lca3["ll"]),
        "class_contents": [
            {t: int(np.sum((post3.argmax(1) == k) & (ftype == t))) for t in ["clean", *types]}
            for k in range(3)
        ],
        "clean_class": clean_class,
    }
    for name, block in estimators.items():
        s_hat = np.array([np.nan if v is None else v for v in block["sens"]])
        p_hat = np.array([np.nan if v is None else v for v in block["spec"]])
        ok = ~np.isnan(s_hat)
        block["mae"] = float(np.mean(np.r_[np.abs(s_hat[ok] - sens[ok]), np.abs(p_hat[ok] - spec[ok])]))
        block["max_err"] = float(np.max(np.r_[np.abs(s_hat[ok] - sens[ok]), np.abs(p_hat[ok] - spec[ok])]))

    # Gold anchors: random items whose gold label is revealed.
    gold_map = {item: POSITIVE if gold_rows[item]["faulty"] else panel.classes[1 - pos] for item in panel.items}
    gold_free_err = np.r_[np.abs(est_sens - sens), np.abs(est_spec - spec)]
    # Family-wise level: Bonferroni over the 2 * m checks of one audit.
    level_fw = 1.0 - 0.05 / (2 * m)
    anchors = {"level_family_wise": level_fw}
    for k in ANCHOR_SIZES:
        any_flag, n_flags, direct_mae, pair_flag, any_fw, pair_fw = [], [], [], [], [], []
        any_fw_wilson, pair_fw_wilson = [], []
        for _ in range(ANCHOR_DRAWS):
            chosen = rng.choice(panel.n_items, size=k, replace=False)
            sample = {panel.items[i]: gold_map[panel.items[i]] for i in chosen}
            audit = anchor_audit(panel, full, sample)
            flagged = audit.flagged()
            any_flag.append(bool(flagged))
            n_flags.append(len(flagged))
            pair_flag.append(any(c.judge in PAIR and c.true_class == POSITIVE for c in flagged))
            flagged_fw = anchor_audit(panel, full, sample, level=level_fw).flagged()
            any_fw.append(bool(flagged_fw))
            pair_fw.append(any(c.judge in PAIR and c.true_class == POSITIVE for c in flagged_fw))
            flagged_w = anchor_audit(panel, full, sample, level=level_fw, method="wilson").flagged()
            any_fw_wilson.append(bool(flagged_w))
            pair_fw_wilson.append(any(c.judge in PAIR and c.true_class == POSITIVE for c in flagged_w))
            errs = []
            for c in audit.checks:
                if c.anchor is None:
                    continue
                j = judges.index(c.judge)
                truth_value = sens[j] if c.true_class == POSITIVE else spec[j]
                errs.append(abs(c.anchor - truth_value))
            direct_mae.append(float(np.mean(errs)))
        anchors[str(k)] = {
            "p_any_flag": float(np.mean(any_flag)),
            "p_pair_flag": float(np.mean(pair_flag)),
            "p_any_flag_fw": float(np.mean(any_fw)),
            "p_pair_flag_fw": float(np.mean(pair_fw)),
            "p_any_flag_fw_wilson": float(np.mean(any_fw_wilson)),
            "p_pair_flag_fw_wilson": float(np.mean(pair_fw_wilson)),
            "mean_flags": float(np.mean(n_flags)),
            "direct_mae": float(np.mean(direct_mae)),
        }

    # False alarms when the fit is right: panels drawn from the real fit itself,
    # refitted, and audited with random anchors (exact and Wilson intervals).
    null_rng = np.random.default_rng(SEED + 13)
    null_flags: dict[str, dict[str, list[bool]]] = {
        str(k): {"exact": [], "wilson": []} for k in (10, 20, 50, 100)
    }
    for _ in range(NULL_REPS):
        sim_truth = null_rng.choice(panel.n_classes, size=panel.n_items, p=full.class_prior)
        sim_labels = np.empty_like(labels)
        for j in range(m):
            cumulative = full.confusion[j].cumsum(axis=1)
            draws = null_rng.random(panel.n_items)
            sim_labels[:, j] = np.minimum((draws[:, None] > cumulative[sim_truth]).sum(axis=1), panel.n_classes - 1)
        sim = Panel(labels=sim_labels, items=panel.items, judges=panel.judges, classes=panel.classes)
        sim_fit = quiet_fit(sim)
        sim_gold = {item: panel.classes[int(c)] for item, c in zip(panel.items, sim_truth, strict=True)}
        for k, store in null_flags.items():
            chosen = null_rng.choice(panel.n_items, size=int(k), replace=False)
            sample = {panel.items[i]: sim_gold[panel.items[i]] for i in chosen}
            for method in ("exact", "wilson"):
                store[method].append(bool(anchor_audit(sim, sim_fit, sample, level=level_fw, method=method).flagged()))
    anchors_null_from_fit = {
        k: {method: float(np.mean(v)) for method, v in store.items()} for k, store in null_flags.items()
    }

    # Anchors scored as estimators: direct, and prediction-powered (rectified).
    ppi = {}
    for k in ANCHOR_SIZES:
        maes, prev_err = [], []
        for _ in range(ANCHOR_DRAWS // 4):
            chosen = rng.choice(panel.n_items, size=k, replace=False)
            sample = {panel.items[i]: gold_map[panel.items[i]] for i in chosen}
            r = rectified_estimates(panel, full, sample, positive=POSITIVE)
            s_hat = np.array([r.sensitivity[j] for j in judges])
            p_hat = np.array([r.specificity[j] for j in judges])
            maes.append(mae(s_hat, p_hat, sens, spec))
            prev_err.append(abs(r.prevalence - prev))
        ppi[str(k)] = {"mae": float(np.mean(maes)), "prev_abs_err": float(np.mean(prev_err))}

    # Anchors drawn from the items the posterior calls clean, where the misses live.
    post_clean = np.where(~posterior_faulty)[0]
    stratified = {
        "n_posterior_clean": int(post_clean.size),
        "faulty_among_posterior_clean": int(truth[post_clean].sum()),
        "expected_faulty_under_fit": float(full.posterior[post_clean, pos].sum()),
    }
    for k in (5, 10, 20):
        hits = [bool(truth[rng.choice(post_clean, size=k, replace=False)].any()) for _ in range(ANCHOR_DRAWS)]
        stratified[f"p_catch_{k}"] = float(np.mean(hits))
    # The stratum the fit cannot see: faulty items the posterior calls clean.
    hidden_share = float(np.mean((truth == 1) & ~posterior_faulty))
    type_share = float(np.mean(ftype == "unsupported_claim"))
    design = {str(k): 1.0 - (1.0 - hidden_share) ** k for k in ANCHOR_SIZES}

    # Ablation: drop all items of one fault type, refit, and rerun the checks.
    ablation = {}
    masks = {
        "full": np.ones(panel.n_items, dtype=bool),
        "without_unsupported_claim": ftype != "unsupported_claim",
        "without_wrong_tool": ftype != "wrong_tool",
        "without_both": (ftype != "unsupported_claim") & (ftype != "wrong_tool"),
    }
    for name, mask in masks.items():
        part = Panel(
            labels=labels[mask],
            items=tuple(np.asarray(panel.items)[mask]),
            judges=panel.judges,
            classes=panel.classes,
        )
        f = quiet_fit(part)
        s_hat, p_hat, prev_hat = fit_rates(f, pos)
        s_t, p_t = true_rates(labels[mask], truth[mask], pos)
        c = check if name == "full" else dependence_check(part, n_boot=DIAG_BOOT, seed=SEED + 4)
        ablation[name] = {
            "n": int(mask.sum()),
            "mae": mae(s_hat, p_hat, s_t, p_t),
            "max_err": float(np.max(np.r_[np.abs(s_hat - s_t), np.abs(p_hat - p_t)])),
            "prevalence": prev_hat,
            "prevalence_true": float(truth[mask].mean()),
            "drop_one_max": float(np.max(c.drop_one)),
            "drop_one_max_p": c.drop_one_max_p,
            "g2_p": c.g2_p,
            "bvr_max_p": c.bvr_max_p,
            "tetrad_p": c.tetrad_p,
        }

    # Coalition identity: in every sub-panel of three or more judges, the fitted
    # prevalence against the positive rate of the judge the fit ranks best.
    flags = labels == pos
    coalition = []
    for size in range(3, m + 1):
        for cols in itertools.combinations(range(m), size):
            f = quiet_fit(subpanel(panel, list(cols)))
            s_hat, p_hat, prev_hat = fit_rates(f, pos)
            youden = s_hat + p_hat - 1
            top = cols[int(np.argmax(youden))]
            coalition.append(
                {
                    "judges": [judges[c] for c in cols],
                    "prevalence": prev_hat,
                    "top_judge": judges[top],
                    "top_flag_rate": float(flags[:, top].mean()),
                    "top_youden_est": float(youden.max()),
                }
            )
    true_youden = sens + spec - 1

    # Boundary capture: the residuals of the near-perfect judge with every other judge.
    captured = int(np.argmax(est_sens + est_spec - 1))
    proposition = {
        "captured_judge": judges[captured],
        "captured_sens": float(est_sens[captured]),
        "captured_spec": float(est_spec[captured]),
        "captured_bvr_max": float(np.nanmax(np.delete(bvr[captured], captured))),
        "captured_flag_rate": float(flags[:, captured].mean()),
    }

    payload = {
        "seed": SEED,
        "ppi": ppi,
        "stratified_anchor": stratified,
        "anchor_design_formula": {"hidden_share": hidden_share, "type_share": type_share, "p_at_least_one": design},
        "ablation": ablation,
        "coalition": coalition,
        "coalition_max_gap": float(max(abs(r["prevalence"] - r["top_flag_rate"]) for r in coalition)),
        "coalition_min_top_youden": float(min(r["top_youden_est"] for r in coalition)),
        "true_max_youden": float(true_youden.max()),
        "proposition": proposition,
        "judges": judges,
        "pair": list(PAIR),
        "truth": {"sens": sens.tolist(), "spec": spec.tolist(), "prevalence": prev},
        "gold_free": {"sens": est_sens.tolist(), "spec": est_spec.tolist(), "prevalence": est_prev},
        "gold_free_mae": float(np.mean(gold_free_err)),
        "by_type": by_type,
        "precision": precision_block,
        "multistart": multistart,
        "drop_one": drop_one,
        "diagnostics": diagnostics,
        "attribution": attribution,
        "estimators": estimators,
        "anchors": anchors,
        "anchors_null_from_fit": anchors_null_from_fit,
        "multistart_targeted": multistart_targeted,
    }
    out = Path(__file__).parent / "results" / "audit.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

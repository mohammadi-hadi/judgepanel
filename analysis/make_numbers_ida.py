"""Emit ida/numbers.tex: every result number in the IDA paper as a LaTeX macro.

Asserts pin the claims the text makes in words (which judges, which pair,
which test fires), so a change in data or code fails here, not in prose.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = Path(__file__).parent / "results"
# Where the paper's LaTeX lives; override with IDA_DIR when the paper sits elsewhere.
IDA_DIR = Path(os.environ.get("IDA_DIR", ROOT / "ida"))

KEYS = {
    "programmatic": "Rules",
    "outcome:qwen2.5:14b": "Outcome",
    "step:qwen2.5:14b": "StepQ",
    "step:llama3.1:8b": "StepL",
    "selfcons3:qwen2.5:14b": "SelfCons",
}
PAIR = ("step:qwen2.5:14b", "selfcons3:qwen2.5:14b")


def prob(x: float) -> str:
    return f"{x:.3f}"


def two(x: float) -> str:
    return f"{x:.2f}"


def signed(x: float) -> str:
    return f"\\ensuremath{{{x:+.3f}}}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def pct_up(x: float) -> str:
    """Percent rounded up, for bounds stated as 'at most'."""
    return f"{math.ceil(100 * x - 1e-9):d}"


def pval(p: float, n_boot: int) -> str:
    floor = 1.0 / (n_boot + 1)
    return f"{p:.3f}" if p > floor + 1e-12 else f"{floor:.3f}"


def main() -> None:
    audit = json.loads((RESULTS / "audit.json").read_text())
    real = json.loads((RESULTS / "real.json").read_text())
    sims = json.loads((RESULTS / "sims.json").read_text())
    diag_path = RESULTS / "diag_sims.json"
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else None

    judges = audit["judges"]
    pair_idx = [judges.index(p) for p in PAIR]
    d = audit["diagnostics"]
    # --- claims made in words ---
    assert len(audit["multistart"]) == 1, "EM must have a single optimum on the real panel"
    assert set(d["bvr_max_pair"]) == {"programmatic", "outcome:qwen2.5:14b"}
    assert d["bvr_pair_of_interest"] < 0.01
    assert d["drop_one_p"][PAIR[0]] <= 0.005 and d["drop_one_p"][PAIR[1]] <= 0.005
    others = [j for j in judges if j not in PAIR]
    assert all(d["drop_one_p"][j] > 0.3 for j in others)
    assert d["g2_p"] <= 0.005 and d["tetrad_p"] <= 0.005
    misses = audit["by_type"]["posterior_misses_by_type"]
    assert misses["unsupported_claim"] == max(misses.values())
    assert audit["estimators"]["dawid_skene_without_selfcons3:qwen2.5:14b"]["mae"] > 2 * audit["gold_free_mae"]

    macros: dict[str, str] = {}
    truth, est = audit["truth"], audit["gold_free"]
    for j, name in enumerate(judges):
        key = KEYS[name]
        macros[f"jp{key}SensEst"] = prob(est["sens"][j])
        macros[f"jp{key}SensTrue"] = prob(truth["sens"][j])
        macros[f"jp{key}SpecEst"] = prob(est["spec"][j])
        macros[f"jp{key}SpecTrue"] = prob(truth["spec"][j])
        macros[f"jp{key}PrecEst"] = prob(audit["precision"]["estimated"][name])
        macros[f"jp{key}PrecTrue"] = prob(audit["precision"]["true"][name])
        macros[f"jp{key}SensVisible"] = prob(audit["by_type"]["sens_excluding_unsupported_claim"][name])
        macros[f"jp{key}SensUnsup"] = two(audit["by_type"]["sensitivity"][name]["unsupported_claim"])
        macros[f"jp{key}SensWrongTool"] = two(audit["by_type"]["sensitivity"][name]["wrong_tool"])
        macros[f"jp{key}DropShift"] = prob(d["drop_one_shift"][name])
        macros[f"jp{key}DropP"] = pval(d["drop_one_p"][name], d["n_boot"])
    for row in real["judges"]:
        key = KEYS[row["judge"]]
        macros[f"jp{key}SensLo"] = prob(row["sens_ci"][0])
        macros[f"jp{key}SensHi"] = prob(row["sens_ci"][1])
    macros["jpPrevEst"] = prob(est["prevalence"])
    macros["jpPrevTrue"] = prob(truth["prevalence"])
    macros["jpPrevBias"] = signed(est["prevalence"] - truth["prevalence"])
    macros["jpGoldFreeMae"] = prob(audit["gold_free_mae"])
    errs = [abs(est["sens"][j] - truth["sens"][j]) for j in range(len(judges))]
    errs += [abs(est["spec"][j] - truth["spec"][j]) for j in range(len(judges))]
    macros["jpGoldFreeMax"] = prob(max(errs))
    macros["jpKappaMax"] = prob(real["kappa"]["max_value"])
    macros["jpBootN"] = f"{real['n_boot']:,}".replace(",", "{,}")
    macros["jpNItems"] = str(real["n_items"])
    macros["jpNFaulty"] = str(real["blind_spots"]["n_faulty"])
    macros["jpNClean"] = str(real["n_items"] - real["blind_spots"]["n_faulty"])
    macros["jpNSilent"] = str(real["blind_spots"]["n_silent"])
    macros["jpMissedPost"] = str(real["blind_spots"]["faulty_missed_by_posterior"])
    macros["jpMissedUnsup"] = str(misses["unsupported_claim"])
    macros["jpPostAcc"] = prob(real["aggregation"]["posterior"]["accuracy"])
    macros["jpMajAcc"] = prob(real["aggregation"]["majority"]["accuracy"])

    macros["jpMultiStarts"] = str(audit["multistart"][0]["count"])
    tgt = audit["multistart_targeted"]
    assert tgt["n_same_optimum"] == tgt["n_starts"] and not tgt["any_higher"]
    macros["jpMultiTargeted"] = str(tgt["n_starts"])
    macros["jpMultiTargetedHard"] = str(tgt["n_starts"] - len(judges))

    drop = audit["drop_one"]
    for name in PAIR:
        key = KEYS[name]
        macros[f"jpDrop{key}Prev"] = prob(drop[name]["prevalence"])
        macros[f"jpDrop{key}Mae"] = prob(drop[name]["mae"])
        macros[f"jpDrop{key}RulesSens"] = prob(drop[name]["sens"]["programmatic"])
    macros["jpDropSelfConsStepSpec"] = prob(drop["selfcons3:qwen2.5:14b"]["spec"]["step:qwen2.5:14b"])

    macros["jpDiagBoot"] = f"{d['n_boot']}"
    macros["jpGTwo"] = f"{d['g2']:.1f}"
    macros["jpGTwoP"] = pval(d["g2_p"], d["n_boot"])
    macros["jpBvrMax"] = f"{d['bvr_max']:.2f}"
    macros["jpBvrMaxP"] = pval(d["bvr_max_p"], d["n_boot"])
    macros["jpBvrPair"] = f"{d['bvr_pair_of_interest']:.4f}"
    macros["jpDropMaxP"] = pval(d["drop_one_max_p"], d["n_boot"])
    macros["jpTetradMax"] = prob(d["tetrad_max"])
    macros["jpTetradP"] = pval(d["tetrad_p"], d["n_boot"])

    names = {
        "indep_given_class": "IndClass",
        "pair_joint_given_class": "PairClass",
        "indep_given_type": "IndType",
        "pair_joint_given_type": "PairType",
        "all_joint_given_type": "AllType",
        "item_bootstrap": "Boot",
    }
    for key, short in names.items():
        row = audit["attribution"][key]
        macros[f"jpAttr{short}Mae"] = prob(row["mae_mean"])
        macros[f"jpAttr{short}PairBias"] = signed(row["pair_sens_bias"])
        macros[f"jpAttr{short}RulesBias"] = signed(row["rules_sens_bias"])
        macros[f"jpAttr{short}PrevBias"] = signed(row["prev_bias"])

    est_names = {
        "dawid_skene": "DS",
        "agreement_with_majority": "Maj",
        "latent_class_3": "LCA",
        "dawid_skene_without_selfcons3:qwen2.5:14b": "NoSelfCons",
        "dawid_skene_without_step:qwen2.5:14b": "NoStepQ",
        "pair_joint": "PairJoint",
    }
    for key, short in est_names.items():
        row = audit["estimators"][key]
        macros[f"jpEst{short}Mae"] = prob(row["mae"])
        macros[f"jpEst{short}Max"] = prob(row["max_err"])
        macros[f"jpEst{short}Prev"] = prob(row["prevalence"])
    lca_contents = audit["estimators"]["latent_class_3"]["class_contents"]
    clean = audit["estimators"]["latent_class_3"]["clean_class"]
    macros["jpLcaCleanUnsup"] = str(lca_contents[clean]["unsupported_claim"])

    anchors = audit["anchors"]
    for k in ("10", "20", "30", "50", "100"):
        word = {"10": "Ten", "20": "Twenty", "30": "Thirty", "50": "Fifty", "100": "Hundred"}[k]
        macros[f"jpAnchor{word}Pair"] = pct(anchors[k]["p_pair_flag_fw"])
        macros[f"jpAnchor{word}Mae"] = prob(anchors[k]["direct_mae"])
    macros["jpAnchorLevel"] = f"{anchors['level_family_wise']:.3f}"
    for k, word in (("10", "Ten"), ("20", "Twenty"), ("50", "Fifty")):
        macros[f"jpAnchor{word}PairWilson"] = pct(anchors[k]["p_pair_flag_fw_wilson"])
    nf = audit["anchors_null_from_fit"]
    for k, word in (("10", "Ten"), ("20", "Twenty"), ("50", "Fifty"), ("100", "Hundred")):
        macros[f"jpNullFit{word}Exact"] = pct(nf[k]["exact"])
        macros[f"jpNullFit{word}Wilson"] = pct(nf[k]["wilson"])
    macros["jpNullFitExactMax"] = pct_up(max(v["exact"] for v in nf.values()))
    macros["jpNullFitFiftyExactUp"] = pct_up(nf["50"]["exact"])
    assert nf["50"]["exact"] < 0.05 and anchors["100"]["p_pair_flag_fw"] > 0.99
    macros["jpNullFitWilsonMin"] = pct(min(v["wilson"] for v in nf.values()))
    macros["jpNullFitWilsonMax"] = pct(max(v["wilson"] for v in nf.values()))

    # Ablation (2 x 2): which fault type drives the bias, which drives the checks.
    abl = audit["ablation"]
    assert abl["without_unsupported_claim"]["mae"] < 0.01 and abl["without_unsupported_claim"]["drop_one_max_p"] <= 0.005
    assert abl["without_wrong_tool"]["mae"] > abl["full"]["mae"] and abl["without_wrong_tool"]["drop_one_max_p"] > 0.5
    for key, short in (("full", "Full"), ("without_unsupported_claim", "NoUnsup"),
                       ("without_wrong_tool", "NoWrong"), ("without_both", "NoBoth")):
        row = abl[key]
        macros[f"jpAbl{short}N"] = str(row["n"])
        macros[f"jpAbl{short}Mae"] = prob(row["mae"])
        macros[f"jpAbl{short}Max"] = prob(row["max_err"])
        macros[f"jpAbl{short}Prev"] = prob(row["prevalence"])
        macros[f"jpAbl{short}PrevTrue"] = prob(row["prevalence_true"])
        macros[f"jpAbl{short}DropP"] = pval(row["drop_one_max_p"], d["n_boot"])
        macros[f"jpAbl{short}GTwoP"] = pval(row["g2_p"], d["n_boot"])
        macros[f"jpAbl{short}TetP"] = pval(row["tetrad_p"], d["n_boot"])
        macros[f"jpAbl{short}BvrP"] = pval(row["bvr_max_p"], d["n_boot"])

    # Coalition identity and boundary capture.
    assert audit["coalition_max_gap"] < 0.025 and len(audit["coalition"]) == 16
    macros["jpCoalitionN"] = str(len(audit["coalition"]))
    macros["jpCoalitionGap"] = two(audit["coalition_max_gap"])
    macros["jpCoalitionMinTopJ"] = two(audit["coalition_min_top_youden"])
    macros["jpTrueMaxJ"] = prob(audit["true_max_youden"])
    prevs = sorted({round(r["prevalence"], 2) for r in audit["coalition"]})
    macros["jpCoalitionPrevLow"] = two(min(r["prevalence"] for r in audit["coalition"]))
    macros["jpCoalitionPrevHigh"] = two(max(r["prevalence"] for r in audit["coalition"]))
    winners = {r["top_judge"] for r in audit["coalition"]}
    assert winners <= {"programmatic", "step:qwen2.5:14b", "selfcons3:qwen2.5:14b"}
    prop = audit["proposition"]
    assert prop["captured_judge"] == "step:qwen2.5:14b"
    macros["jpCapSens"] = f"{prop['captured_sens']:.4f}"
    macros["jpCapSpec"] = f"{prop['captured_spec']:.4f}"
    macros["jpCapBvrMax"] = two(prop["captured_bvr_max"])
    macros["jpCapFlagRate"] = prob(prop["captured_flag_rate"])
    macros["jpRulesFlagRate"] = prob(next(r["top_flag_rate"] for r in audit["coalition"] if r["top_judge"] == "programmatic"))
    # The rule stated in words: the 14B coalition wins when the pair sits together or rules is absent.
    for r in audit["coalition"]:
        together = set(PAIR) <= set(r["judges"])
        assert (r["top_judge"] != "programmatic") == (together or "programmatic" not in r["judges"])
    pair_prev = [r["prevalence"] for r in audit["coalition"] if r["top_judge"] != "programmatic"]
    rules_prev = [r["prevalence"] for r in audit["coalition"] if r["top_judge"] == "programmatic"]
    macros["jpCoalPairPrevLo"] = two(min(pair_prev))
    macros["jpCoalPairPrevHi"] = two(max(pair_prev))
    macros["jpCoalRulesPrevLo"] = two(min(rules_prev))
    macros["jpCoalRulesPrevHi"] = two(max(rules_prev))

    # Literal thresholds used in the text of Sections 4 and 5.
    n = len(judges)
    assert max(abs(est["spec"][j] - truth["spec"][j]) for j in range(n)) < 0.04
    visible = audit["by_type"]["sens_excluding_unsupported_claim"]
    assert max(abs(est["sens"][j] - visible[name]) for j, name in enumerate(judges)) <= 0.03
    precision = audit["precision"]
    for name in ("programmatic", "step:qwen2.5:14b", "selfcons3:qwen2.5:14b"):
        assert abs(precision["estimated"][name] - precision["true"][name]) <= 0.01
    for name in judges:
        if name not in PAIR:
            assert d["drop_one_shift"][name] < 0.003 and d["drop_one_p"][name] > 0.5
    for key in ("g2_p", "tetrad_p", "drop_one_max_p"):
        assert audit["ablation"]["without_unsupported_claim"][key] <= 0.05
        assert audit["ablation"]["without_wrong_tool"][key] > 0.05
    assert audit["coalition_min_top_youden"] >= 0.95
    assert round(audit["coalition_max_gap"], 2) <= 0.02
    assert set(real["kappa"]["max_pair"]) == set(PAIR)
    for k in ("20", "50", "100"):
        assert audit["ppi"][k]["mae"] < min(audit["gold_free_mae"], audit["anchors"][k]["direct_mae"])
    floor = 1.0 / (d["n_boot"] + 1) + 1e-12
    assert d["g2_p"] <= floor and d["tetrad_p"] <= floor and d["drop_one_max_p"] <= 2 * floor
    assert d["bvr_max_p"] < 0.05 and set(d["bvr_max_pair"]) == {"programmatic", "outcome:qwen2.5:14b"}
    for key in ("Rules", "StepQ", "SelfCons"):
        lo, hi = float(macros[f"jp{key}SensLo"]), float(macros[f"jp{key}SensHi"])
        assert not lo <= float(macros[f"jp{key}SensTrue"]) <= hi, key
    for key in ("g2_p", "tetrad_p", "drop_one_max_p", "bvr_max_p"):
        assert audit["ablation"]["without_unsupported_claim"][key] <= 0.05
        assert audit["ablation"]["without_wrong_tool"][key] > 0.05
    macros["jpVisibleFaults"] = str(audit["ablation"]["without_unsupported_claim"]["n"] - (real["n_items"] - real["blind_spots"]["n_faulty"]))

    # Anchors as estimators and the design formula.
    ppi = audit["ppi"]
    for k, word in (("10", "Ten"), ("20", "Twenty"), ("50", "Fifty"), ("100", "Hundred")):
        macros[f"jpPpi{word}Mae"] = prob(ppi[k]["mae"])
        macros[f"jpPpi{word}PrevErr"] = prob(ppi[k]["prev_abs_err"])
    design = audit["anchor_design_formula"]
    macros["jpHiddenShare"] = prob(design["hidden_share"])
    macros["jpTypeShare"] = prob(design["type_share"])
    macros["jpDesignTwenty"] = pct(design["p_at_least_one"]["20"])
    strat = audit["stratified_anchor"]
    macros["jpPostCleanN"] = str(strat["n_posterior_clean"])
    macros["jpPostCleanFaulty"] = str(strat["faulty_among_posterior_clean"])
    macros["jpStratTen"] = pct(strat["p_catch_10"])
    macros["jpPostCleanExpected"] = f"{strat['expected_faulty_under_fit']:.1f}"
    macros["jpStratFive"] = pct(strat["p_catch_5"])

    e1 = {(r["m"], r["n"], r["prevalence"]): r for r in sims["e1_recovery"]}
    macros["jpMaeNOneH"] = prob(e1[(5, 100, 0.3)]["mae_mean"])
    macros["jpMaeNFourH"] = prob(e1[(5, 400, 0.3)]["mae_mean"])
    macros["jpMaeNSixteenH"] = prob(e1[(5, 1600, 0.3)]["mae_mean"])
    macros["jpMaePrevLowFourH"] = prob(e1[(5, 400, 0.1)]["mae_mean"])
    macros["jpMaePrevHalfFourH"] = prob(e1[(5, 400, 0.5)]["mae_mean"])
    slope_rows = [e1[(5, n, 0.3)]["mae_mean"] for n in (100, 200, 400, 800, 1600)]
    xs = [math.log(n) for n in (100, 200, 400, 800, 1600)]
    ys = [math.log(v) for v in slope_rows]
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / sum(
        (x - xbar) ** 2 for x in xs
    )
    macros["jpRecoverySlope"] = f"{slope:.2f}"
    budget = sorted(sims["e5_budget"], key=lambda r: r["mae_mean"])
    macros["jpBudget"] = f"{sims['budget']:,}".replace(",", "{,}")
    macros["jpBudgetBestM"] = str(budget[0]["m"])
    macros["jpBudgetBestN"] = str(budget[0]["n"])
    macros["jpBudgetBestMae"] = prob(budget[0]["mae_mean"])
    by_m = {r["m"]: r for r in sims["e5_budget"]}
    macros["jpBudgetTenMae"] = prob(by_m[10]["mae_mean"])
    macros["jpBudgetTenN"] = str(by_m[10]["n"])
    copycat = {r["copy_prob"]: r for r in sims["e2_copycat"]}
    macros["jpCloneBiasFull"] = signed(copycat[1.0]["clone_sens_bias"])
    macros["jpCloneBiasHalf"] = signed(copycat[0.5]["clone_sens_bias"])

    if diag is not None:
        conds = {c["name"]: c for c in diag["conditions"]}
        macros["jpSimReps"] = str(diag["reps"])
        macros["jpSimBoot"] = str(diag["n_boot"])
        tests = {"g2": "GTwo", "bvr_max": "Bvr", "drop_one": "Drop", "tetrad": "Tet"}
        cnames = {
            "independent_p30": "NullLow",
            "independent_p75": "NullHigh",
            "independent_constant_judge": "NullConst",
            "copy_c025": "CopyQuarter",
            "copy_c050": "CopyHalf",
            "copy_c075": "CopyThreeQ",
            "copy_c100": "CopyFull",
            "blind_q2_h10": "BlindTen",
            "blind_q2_h20": "BlindTwenty",
            "blind_q2_h30": "BlindThirty",
            "blind_q3_h20": "BlindThree",
            "blind_q1_h20": "BlindOne",
            "blind_q4_h20": "BlindFour",
            "blind_q5_h20": "BlindFive",
            "real_indep_given_class": "RealNull",
            "real_indep_given_type": "RealType",
            "real_pair_joint_given_class": "RealPair",
        }
        for cname, short in cnames.items():
            if cname not in conds:
                continue
            row = conds[cname]
            for tkey, tshort in tests.items():
                macros[f"jpPow{short}{tshort}"] = pct(row["reject"][tkey])
            macros[f"jpPow{short}Mae"] = prob(row["mae"])
            macros[f"jpLoc{short}Drop"] = pct(row["localise"]["drop_top_in_dependent"])
            macros[f"jpLoc{short}Bvr"] = pct(row["localise"]["bvr_pair_hits_dependent"])
            macros[f"jpPow{short}MaxErr"] = two(row["max_err"])
        nulls = ("independent_p30", "independent_p75", "independent_constant_judge", "real_indep_given_class")
        asim_path = RESULTS / "anchor_sims.json"
        if asim_path.exists():
            asim = {c["name"]: c for c in json.loads(asim_path.read_text())["conditions"]}
            interior = ("independent_p30", "independent_p75", "real_indep_given_class")
            macros["jpAnchorNullExactMax"] = pct_up(max(asim[n]["flag"]["exact"][k] for n in interior for k in asim[n]["flag"]["exact"]))
            macros["jpAnchorNullConstExactMax"] = pct_up(max(asim["independent_constant_judge"]["flag"]["exact"].values()))
            macros["jpAnchorNullWilsonMin"] = pct(min(asim[n]["flag"]["wilson"][k] for n in nulls for k in asim[n]["flag"]["wilson"]))
            macros["jpAnchorNullWilsonMax"] = pct(max(asim[n]["flag"]["wilson"][k] for n in nulls for k in asim[n]["flag"]["wilson"]))
            spread = [asim[f"blind_q{q}_h20"]["flag"]["exact"]["100"] for q in (2, 3, 4, 5)]
            macros["jpAncBlindHundredMin"] = pct(min(spread))
            macros["jpAncBlindHundredMax"] = pct(max(spread))
            spread50 = [asim[f"blind_q{q}_h20"]["flag"]["exact"]["50"] for q in (2, 3, 4, 5)]
            macros["jpAncBlindFiftyMin"] = pct(min(spread50))
            macros["jpAncBlindFiftyMax"] = pct(max(spread50))
            assert max(asim[n]["flag"]["exact"]["50"] for n in nulls) < 0.05
            macros["jpAncNullConstFiftyUp"] = pct_up(asim["independent_constant_judge"]["flag"]["exact"]["50"])
            for cname, short in cnames.items():
                if cname in asim:
                    for k, word in (("20", "Twenty"), ("50", "Fifty"), ("100", "Hundred")):
                        macros[f"jpAnc{short}{word}"] = pct(asim[cname]["flag"]["exact"][k])
        sizes = [conds[n]["reject"][t] for n in ("independent_p30", "independent_p75",
                                                 "independent_constant_judge", "real_indep_given_class")
                 for t in tests]
        # Claims in the simulation paragraph.
        assert max(sizes) <= 0.08
        qs = [conds[f"blind_q{q}_h20"] for q in (2, 3, 4, 5)]
        assert all(qs[0]["reject"][t] >= 0.9 for t in tests)
        assert all(qs[-1]["reject"][t] <= 0.11 for t in tests)
        assert min(r["max_err"] for r in qs) >= 0.18 and min(r["mae"] for r in qs) >= 0.09
        assert conds["copy_c050"]["reject"]["drop_one"] >= 0.8
        assert conds["copy_c050"]["localise"]["drop_top_in_dependent"] >= 0.95
        macros["jpSizeMax"] = pct(max(sizes))
        macros["jpSizeMin"] = pct(min(sizes))

    out = IDA_DIR / "numbers.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% AUTO-GENERATED by analysis/make_numbers_ida.py - do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(macros)} macros)")


if __name__ == "__main__":
    main()

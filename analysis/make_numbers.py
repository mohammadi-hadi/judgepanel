"""Emit paper/numbers.tex: every result number in the paper as a LaTeX macro.

Asserts pin the paper's headline claims to the committed results, so a silent
change in the data or the fit fails loudly here instead of in prose.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = Path(__file__).parent / "results"

KEYS = {
    "programmatic": "Rules",
    "outcome:qwen2.5:14b": "Outcome",
    "step:qwen2.5:14b": "StepQ",
    "step:llama3.1:8b": "StepL",
    "selfcons3:qwen2.5:14b": "SelfCons",
}


def prob(x: float) -> str:
    return f"{x:.3f}"


def signed(x: float) -> str:
    return f"{x:+.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main() -> None:
    real = json.loads((RESULTS / "real.json").read_text())
    sims = json.loads((RESULTS / "sims.json").read_text())

    # --- the claims the paper hangs on; fail loudly if the data moved ---
    assert real["n_items"] == 400
    assert real["blind_spots"]["n_faulty"] == 300
    assert set(real["kappa"]["max_pair"]) == {"step:qwen2.5:14b", "selfcons3:qwen2.5:14b"}
    assert abs(real["errors"]["real_max"] - 0.139) < 0.002
    assert real["errors"]["real_exceeds_share"] == 1.0
    step_l = next(r for r in real["judges"] if r["judge"] == "step:llama3.1:8b")
    assert step_l["est_spec"] < 0.001 and step_l["true_spec"] < 0.001

    macros: dict[str, str] = {}
    for row in real["judges"]:
        key = KEYS[row["judge"]]
        macros[f"jp{key}SensEst"] = prob(row["est_sens"])
        macros[f"jp{key}SensTrue"] = prob(row["true_sens"])
        macros[f"jp{key}SensLo"] = prob(row["sens_ci"][0])
        macros[f"jp{key}SensHi"] = prob(row["sens_ci"][1])
        macros[f"jp{key}SpecEst"] = prob(row["est_spec"])
        macros[f"jp{key}SpecTrue"] = prob(row["true_spec"])
        macros[f"jp{key}SpecLo"] = prob(row["spec_ci"][0])
        macros[f"jp{key}SpecHi"] = prob(row["spec_ci"][1])
        macros[f"jp{key}Acc"] = prob(row["accuracy"])
        macros[f"jp{key}F"] = prob(row["f1"])

    macros["jpPostAcc"] = prob(real["aggregation"]["posterior"]["accuracy"])
    macros["jpPostF"] = prob(real["aggregation"]["posterior"]["f1"])
    macros["jpMajAcc"] = prob(real["aggregation"]["majority"]["accuracy"])
    macros["jpMajF"] = prob(real["aggregation"]["majority"]["f1"])
    macros["jpPrevEst"] = prob(real["prevalence_est"])
    macros["jpPrevTrue"] = prob(real["prevalence_true"])
    macros["jpPrevBias"] = signed(real["prevalence_est"] - real["prevalence_true"])
    macros["jpKappaMax"] = prob(real["kappa"]["max_value"])

    errors = real["errors"]
    macros["jpRealMae"] = prob(errors["real_mae"])
    macros["jpRealMax"] = prob(errors["real_max"])
    macros["jpCfMean"] = prob(errors["counterfactual_mae_mean"])
    macros["jpCfPNF"] = prob(errors["counterfactual_mae_p95"])
    macros["jpCfMax"] = prob(errors["counterfactual_mae_max"])
    macros["jpExceedsPct"] = pct(errors["real_exceeds_share"])

    blind = real["blind_spots"]
    macros["jpNFaulty"] = str(blind["n_faulty"])
    macros["jpNSilent"] = str(blind["n_silent"])
    macros["jpMissedPost"] = str(blind["faulty_missed_by_posterior"])
    silent_n = round(
        blind["silent_share_of_posterior_misses"] * blind["faulty_missed_by_posterior"]
    )
    macros["jpMissedSilentN"] = str(int(silent_n))
    macros["jpMissedSilentPct"] = pct(blind["silent_share_of_posterior_misses"])
    macros["jpZeroDetections"] = str(blind["faulty_with_zero_detections"])

    def e1(m: int, n: int, prevalence: float) -> dict[str, float]:
        return next(
            r
            for r in sims["e1_recovery"]
            if r["m"] == m and r["n"] == n and r["prevalence"] == prevalence
        )

    for n, name in ((100, "OneH"), (200, "TwoH"), (400, "FourH"), (800, "EightH"),
                    (1600, "SixteenH")):
        macros[f"jpMaeN{name}"] = prob(e1(5, n, 0.3)["mae_mean"])
    macros["jpMaePNinetyNFourH"] = prob(e1(5, 400, 0.3)["mae_p90"])
    macros["jpMaeMThree"] = prob(e1(3, 400, 0.3)["mae_mean"])
    macros["jpMaeMSeven"] = prob(e1(7, 400, 0.3)["mae_mean"])
    macros["jpMaePrevLow"] = prob(e1(5, 400, 0.1)["mae_mean"])
    macros["jpAccGain"] = signed(e1(5, 400, 0.3)["acc_gain_mean"])

    copycat = {r["copy_prob"]: r for r in sims["e2_copycat"]}
    macros["jpCloneBiasQuarter"] = signed(copycat[0.25]["clone_sens_bias"])
    macros["jpCloneBiasHalf"] = signed(copycat[0.5]["clone_sens_bias"])
    macros["jpCloneBiasFull"] = signed(copycat[1.0]["clone_sens_bias"])
    macros["jpSourceBiasFull"] = signed(copycat[1.0]["source_sens_bias"])
    macros["jpOthersBiasFull"] = signed(copycat[1.0]["others_sens_bias"])
    macros["jpSimPrevBiasFull"] = signed(copycat[1.0]["prevalence_bias"])

    budget = sorted(sims["e5_budget"], key=lambda r: r["mae_mean"])
    best = budget[0]
    macros["jpBudgetBestM"] = str(best["m"])
    macros["jpBudgetBestN"] = str(best["n"])
    macros["jpBudgetBestMae"] = prob(best["mae_mean"])
    by_m = {r["m"]: r for r in sims["e5_budget"]}
    macros["jpBudgetThreeMae"] = prob(by_m[3]["mae_mean"])
    macros["jpBudgetTenMae"] = prob(by_m[10]["mae_mean"])

    macros["jpReps"] = str(sims["reps"])
    macros["jpBootN"] = str(real["n_boot"])
    macros["jpCfReps"] = "500"
    macros["jpNItems"] = str(real["n_items"])

    lines = ["% AUTO-GENERATED by analysis/make_numbers.py - do not edit by hand"]
    for name in sorted(macros):
        lines.append(f"\\newcommand{{\\{name}}}{{{macros[name]}}}")
    out = ROOT / "paper" / "numbers.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(macros)} macros)")

    header = "% AUTO-GENERATED by analysis/make_numbers.py - do not edit by hand"
    grid = [
        header,
        "\\begin{tabular}{cccccc}",
        "\\toprule",
        "judges & prevalence & items & MAE (mean) & MAE (p90) & prevalence err.\\ \\\\",
        "\\midrule",
    ]
    for row in sorted(
        sims["e1_recovery"], key=lambda r: (r["m"], r["prevalence"], r["n"])
    ):
        grid.append(
            f"{row['m']} & {row['prevalence']:.1f} & {row['n']} & "
            f"{row['mae_mean']:.3f} & {row['mae_p90']:.3f} & "
            f"{row['prev_err_mean']:.3f} \\\\"
        )
    grid += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper" / "grid_table.tex").write_text("\n".join(grid) + "\n")

    budget = [
        header,
        "\\begin{tabular}{cccc}",
        "\\toprule",
        "judges & items & MAE (mean) & MAE (p90) \\\\",
        "\\midrule",
    ]
    for row in sorted(sims["e5_budget"], key=lambda r: r["m"]):
        budget.append(
            f"{row['m']} & {row['n']} & {row['mae_mean']:.3f} & {row['mae_p90']:.3f} \\\\"
        )
    budget += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper" / "budget_table.tex").write_text("\n".join(budget) + "\n")
    print("wrote grid_table.tex and budget_table.tex")


if __name__ == "__main__":
    main()

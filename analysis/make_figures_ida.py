"""Figures for the IDA 2027 paper (vector PDF), from the committed results JSON.

Sized for the LNCS text block (122 mm wide). Colours follow the workshop
paper; orange always marks the step judge and its self-consistency ensemble.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, YELLOW, GREEN, GRAY, PURPLE = (
    "#2a78d6",
    "#eb6834",
    "#eda100",
    "#1baf7a",
    "#8a8f98",
    "#7d4cdb",
)
TEXT_WIDTH = 4.8  # inches, LNCS text block

ROOT = Path(__file__).parent.parent
RESULTS = Path(__file__).parent / "results"
# Where the paper's LaTeX lives; override with IDA_DIR when the paper sits elsewhere.
FIGURES = Path(os.environ.get("IDA_DIR", ROOT / "ida")) / "figures"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SHORT = {
    "programmatic": "rules",
    "outcome:qwen2.5:14b": "outcome 14B",
    "step:qwen2.5:14b": "step 14B",
    "step:llama3.1:8b": "step 8B",
    "selfcons3:qwen2.5:14b": "selfcons 14B",
}
TYPE_SHORT = {
    "hallucinated_argument": "halluc. arg.",
    "ignored_observation": "ignored obs.",
    "premature_stop": "early stop",
    "skipped_precondition": "skipped check",
    "unsupported_claim": "unsupp. claim",
    "wrong_tool": "wrong tool",
}
PAIR = {"step:qwen2.5:14b", "selfcons3:qwen2.5:14b"}


def fig_real(audit: dict) -> None:
    """(a) true sensitivity by fault type; (b) coalition identity over sub-panels."""
    fig, (bx, ax) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.05), gridspec_kw={"width_ratios": [1.55, 1]}
    )
    judges = audit["judges"]
    types = audit["by_type"]["types"]
    grid = np.array([[audit["by_type"]["sensitivity"][j][t] for t in types] for j in judges])
    bx.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (r, c), value in np.ndenumerate(grid):
        bx.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=7,
                color="white" if value > 0.6 else "black")
    bx.set_xticks(range(len(types)))
    bx.set_xticklabels([TYPE_SHORT[t] for t in types], fontsize=7, rotation=35, ha="right",
                       rotation_mode="anchor")
    bx.set_yticks(range(len(judges)))
    bx.set_yticklabels([SHORT[j] for j in judges])
    bx.set_title("(a) true sensitivity by fault type", loc="left")
    for spine in bx.spines.values():
        spine.set_visible(False)
    bx.tick_params(length=0)

    colors = {"programmatic": BLUE, "step:qwen2.5:14b": ORANGE, "selfcons3:qwen2.5:14b": YELLOW}
    seen: set[str] = set()
    for row in audit["coalition"]:
        top = row["top_judge"]
        ax.scatter(row["top_flag_rate"], row["prevalence"], s=22,
                   color=colors.get(top, GRAY), alpha=0.75, edgecolors="white", linewidths=0.4,
                   label=None if top in seen else f"best: {SHORT[top]}", zorder=3)
        seen.add(top)
    lo, hi = 0.45, 0.8
    ax.plot([lo, hi], [lo, hi], color=GRAY, linewidth=0.6, linestyle="--", zorder=1)
    ax.axhline(audit["truth"]["prevalence"], color="black", linewidth=0.6, linestyle=":")
    ax.text(lo + 0.005, audit["truth"]["prevalence"] + 0.008, "true prevalence", fontsize=7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("top judge's positive rate")
    ax.set_ylabel("estimated prevalence")
    ax.set_title("(b) 16 sub-panels", loc="left")
    # Direct labels next to each cluster; a legend would sit among the points.
    for x, y, name, color, ha in ((0.500, 0.540, "rules", BLUE, "center"),
                                  (0.626, 0.662, "step 14B", ORANGE, "right"),
                                  (0.668, 0.603, "selfcons 14B", YELLOW, "center")):
        ax.text(x, y, name, color=color, ha=ha, va="center", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIGURES / "fig_real.pdf")
    plt.close(fig)


ATTR_LABELS = [
    ("indep_given_class", "independent given clean/faulty"),
    ("pair_joint_given_class", "+ step/selfcons kept joint"),
    ("indep_given_type", "independent given fault type"),
    ("pair_joint_given_type", "+ step/selfcons kept joint"),
    ("all_joint_given_type", "all judges joint within type"),
    ("item_bootstrap", "real panel (item bootstrap)"),
]


BIAS_X = (0.100, 0.120, 0.140)  # right edges of the three bias columns


def fig_attribution(audit: dict) -> None:
    """Estimation error under resampling schemes that keep or break dependence."""
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 1.7))
    attribution = audit["attribution"]
    ys = np.arange(len(ATTR_LABELS))[::-1]
    for y, (key, _) in zip(ys, ATTR_LABELS, strict=True):
        row = attribution[key]
        if key == "indep_given_class":
            color = GRAY
        elif key in ("all_joint_given_type", "item_bootstrap"):
            color = GREEN
        elif "pair" in key:
            color = ORANGE
        else:
            color = BLUE
        ax.barh(y, row["mae_mean"], color=color, height=0.62)
        ax.plot([row["mae_p05"], row["mae_p95"]], [y, y], color="black", linewidth=0.7)
        biases = (row["pair_sens_bias"], row["rules_sens_bias"], row["prev_bias"])
        for x, b in zip(BIAS_X, biases, strict=True):
            ax.text(x, y, f"{b:+.3f}".replace("-", "\u2212"), va="center", ha="right",
                    fontsize=7, color="#4a4f57")
    for x, name in zip(BIAS_X, ("pair", "rules", "prev."), strict=True):
        ax.text(x, ys[0] + 0.8, name, ha="right", fontsize=7, color="#4a4f57")
    ax.axvline(audit["gold_free_mae"], color="black", linestyle=":", linewidth=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label in ATTR_LABELS])
    ax.set_xlim(0, 0.14)
    ax.set_xticks([0, 0.02, 0.04, 0.06])
    ax.set_xlabel("mean |estimate $-$ truth| (bar: mean; line: 5\u201395%)", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_attribution.pdf")
    plt.close(fig)


TESTS = [
    ("bvr_max", "bivariate residual", GREEN, "v"),
    ("g2", "pattern $G^2$", YELLOW, "D"),
    ("tetrad", "tetrads", PURPLE, "s"),
    ("drop_one", "drop-one", BLUE, "o"),
]


def fig_power(diag: dict, audit: dict) -> None:
    """Rejection rates as dependence grows, against the bias it causes; and the anchor."""
    conds = {c["name"]: c for c in diag["conditions"]}
    fig, (ax, bx, cx) = plt.subplots(
        1, 3, figsize=(TEXT_WIDTH, 2.3), gridspec_kw={"width_ratios": [1, 1.15, 0.9]}
    )
    copy_x = [0.0, 0.25, 0.5, 0.75, 1.0]
    copy_names = ["independent_p75", "copy_c025", "copy_c050", "copy_c075", "copy_c100"]
    q_x = [0, 1, 2, 3, 4, 5]
    q_names = ["independent_p75", "blind_q1_h20", "blind_q2_h20", "blind_q3_h20",
               "blind_q4_h20", "blind_q5_h20"]
    have = [n in conds for n in q_names]
    q_x = [x for x, ok in zip(q_x, have, strict=True) if ok]
    q_names = [n for n, ok in zip(q_names, have, strict=True) if ok]
    for key, label, color, marker in TESTS:
        ax.plot(copy_x, [conds[n]["reject"][key] for n in copy_names], marker=marker,
                markersize=3, color=color, label=label)
        bx.plot(q_x, [conds[n]["reject"][key] for n in q_names], marker=marker,
                markersize=3, color=color, label=label)
    twin = bx.twinx()
    twin.plot(q_x, [conds[n]["max_err"] for n in q_names], color="black", linestyle="--",
              linewidth=0.8, label="largest error")
    twin.set_ylim(0, 0.6)
    twin.set_ylabel("largest error", fontsize=7, labelpad=1)
    twin.spines["right"].set_visible(True)
    for axis in (ax, bx):
        axis.axhline(diag["alpha"], color="black", linestyle=":", linewidth=0.6)
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_xlabel("copy probability $c$")
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylabel("rejection rate")
    ax.set_title("(a) a copying judge", loc="left")
    bx.set_xlabel("judges sharing it, $q$")
    bx.set_title("(b) a shared blind spot", loc="left")
    bx.set_xticks(q_x)

    anchors = audit["anchors"]
    sizes = sorted(int(k) for k in anchors if k != "level_family_wise")
    cx.plot(sizes, [anchors[str(k)]["p_pair_flag_fw"] for k in sizes], marker="o",
            markersize=3, color=ORANGE, label="real panel")
    null = audit["anchors_null_from_fit"]
    null_sizes = sorted(int(k) for k in null)
    for method, style in (("exact", "-"), ("wilson", "--")):
        cx.plot(null_sizes, [null[str(k)][method] for k in null_sizes], marker="v", markersize=3,
                color=GRAY, linestyle=style,
                label="fit is right" + (" (Wilson)" if method == "wilson" else ""))
    cx.set_xticks([20, 50, 100])
    cx.set_ylim(0, 1.02)
    cx.grid(axis="y", linewidth=0.3, alpha=0.4)
    cx.set_xlabel("anchor size $k$")
    cx.set_ylabel("P(flag)", labelpad=1)
    cx.set_title("(c) anchor", loc="left")
    handles = [h for axis in (ax, twin, cx) for h in axis.get_legend_handles_labels()[0]]
    names = [n for axis in (ax, twin, cx) for n in axis.get_legend_handles_labels()[1]]
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, handlelength=1.6,
               columnspacing=1.0, borderaxespad=0.1)
    fig.tight_layout(w_pad=0.9, rect=(0, 0.17, 1, 1))
    fig.savefig(FIGURES / "fig_power.pdf")
    plt.close(fig)


def fig_design(sims: dict) -> None:
    """(a) recovery under independence at three prevalences; (b) fixed budget."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.75))
    for prevalence, color in ((0.1, YELLOW), (0.3, BLUE), (0.5, GREEN)):
        rows = sorted(
            (r for r in sims["e1_recovery"] if r["m"] == 5 and r["prevalence"] == prevalence),
            key=lambda r: r["n"],
        )
        ax.plot([r["n"] for r in rows], [r["mae_mean"] for r in rows], marker="o", markersize=3,
                color=color, label=f"prevalence {prevalence}")
    ns = np.array([100, 1600])
    ref = [r for r in sims["e1_recovery"] if r["m"] == 5 and r["prevalence"] == 0.3 and r["n"] == 100]
    ax.plot(ns, ref[0]["mae_mean"] * np.sqrt(100 / ns), color=GRAY, linestyle="--", linewidth=0.6,
            label="$\\propto 1/\\sqrt{n}$")
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 400, 800, 1600])
    ax.set_xticklabels(["100", "200", "400", "800", "1600"])
    ax.set_xlabel("items (5 independent judges)")
    ax.set_ylabel("mean |estimate $-$ truth|")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.set_title("(a) recovery under independence", loc="left")
    budget = sorted(sims["e5_budget"], key=lambda r: r["m"])
    bx.plot([r["m"] for r in budget], [r["mae_mean"] for r in budget], marker="o", markersize=3,
            color=BLUE)
    for r in budget:
        bx.annotate(str(r["n"]), (r["m"], r["mae_mean"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=5.5, color=GRAY)
    bx.set_xlabel(f"judges (items = {sims['budget']:,} / judges)")
    bx.set_ylim(bottom=0)
    bx.set_title(f"(b) fixed budget of {sims['budget']:,} labels", loc="left")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIGURES / "fig_design.pdf")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    audit = json.loads((RESULTS / "audit.json").read_text())
    sims = json.loads((RESULTS / "sims.json").read_text())
    fig_real(audit)
    fig_attribution(audit)
    fig_design(sims)
    diag_path = RESULTS / "diag_sims.json"
    if diag_path.exists():
        fig_power(json.loads(diag_path.read_text()), audit)
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()

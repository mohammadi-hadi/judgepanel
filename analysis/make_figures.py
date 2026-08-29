"""Regenerate every figure (vector PDF) from the committed results JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, YELLOW, GREEN, GRAY = "#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#8a8f98"

ROOT = Path(__file__).parent.parent
RESULTS = Path(__file__).parent / "results"
FIGURES = ROOT / "paper" / "figures"

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
    "outcome:qwen2.5:14b": "outcome (14B)",
    "step:qwen2.5:14b": "step (14B)",
    "step:llama3.1:8b": "step (8B)",
    "selfcons3:qwen2.5:14b": "selfcons k=3 (14B)",
}
CORRELATED = {"step:qwen2.5:14b", "selfcons3:qwen2.5:14b"}


def fig1_recovery(sims: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for m, color in ((3, YELLOW), (5, BLUE), (7, GREEN)):
        rows = [r for r in sims["e1_recovery"] if r["m"] == m and r["prevalence"] == 0.3]
        rows.sort(key=lambda r: r["n"])
        ns = [r["n"] for r in rows]
        ax.plot(
            ns,
            [r["mae_mean"] for r in rows],
            marker="o",
            markersize=3,
            color=color,
            label=f"{m} judges",
        )
        if m == 5:
            ax.plot(
                ns,
                [r["mae_p90"] for r in rows],
                linestyle="--",
                linewidth=0.8,
                color=color,
                alpha=0.6,
            )
    ax.set_xscale("log")
    ax.set_xticks([100, 200, 400, 800, 1600])
    ax.set_xticklabels(["100", "200", "400", "800", "1600"])
    ax.set_xlabel("items labelled by every judge")
    ax.set_ylabel("mean |estimate $-$ truth|")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, title="dashed: p90 (5 judges)", title_fontsize=7)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_recovery.pdf")
    plt.close(fig)


def fig2_real(real: dict) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for row in real["judges"]:
        correlated = row["judge"] in CORRELATED
        color = ORANGE if correlated else BLUE
        left.errorbar(
            row["true_sens"],
            row["est_sens"],
            yerr=[[row["est_sens"] - row["sens_ci"][0]], [row["sens_ci"][1] - row["est_sens"]]],
            fmt="o",
            markersize=4,
            color=color,
            capsize=2,
            linewidth=0.8,
        )
        left.errorbar(
            row["true_spec"],
            row["est_spec"],
            yerr=[[row["est_spec"] - row["spec_ci"][0]], [row["spec_ci"][1] - row["est_spec"]]],
            fmt="s",
            markersize=4,
            markerfacecolor="white",
            color=color,
            capsize=2,
            linewidth=0.8,
        )
    left.plot([0, 1], [0, 1], color=GRAY, linewidth=0.7, zorder=0)
    left.annotate(
        "correlated pair\n(inflated)",
        xy=(0.857, 0.995),
        xytext=(0.48, 0.9),
        fontsize=7,
        color=ORANGE,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 0.6},
    )
    left.set_xlabel("true value (gold labels)")
    left.set_ylabel("gold-free estimate")
    left.set_xlim(-0.03, 1.03)
    left.set_ylim(-0.03, 1.03)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=GRAY, label="sensitivity"),
        plt.Line2D(
            [], [], marker="s", linestyle="", markerfacecolor="white", color=GRAY,
            label="specificity",
        ),
    ]
    left.legend(handles=handles, frameon=False, loc="lower right")
    left.set_title("real five-judge panel: estimate vs truth")

    maes = np.array(real["errors"]["counterfactual_maes"])
    right.hist(maes, bins=30, color=GREEN, alpha=0.75, edgecolor="white", linewidth=0.3)
    right.axvline(
        real["errors"]["counterfactual_mae_p95"],
        color=GREEN,
        linewidth=1.0,
        linestyle="--",
        label="p95, independent",
    )
    right.axvline(
        real["errors"]["real_mae"], color=ORANGE, linewidth=1.6, label="real panel"
    )
    right.set_xlabel("mean |estimate $-$ truth| over all judge parameters")
    right.set_ylabel("independent panels (of 500)")
    right.legend(frameon=False, loc="upper center")
    right.set_title("real error vs independent counterfactual")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_real.pdf")
    plt.close(fig)


def fig3_copycat(sims: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    rows = sorted(sims["e2_copycat"], key=lambda r: r["copy_prob"])
    xs = [r["copy_prob"] for r in rows]
    series = [
        ("clone_sens_bias", ORANGE, "clone judge"),
        ("source_sens_bias", YELLOW, "copied judge"),
        ("others_sens_bias", BLUE, "other judges (mean)"),
        ("prevalence_bias", GREEN, "prevalence"),
    ]
    for key, color, label in series:
        ax.plot(xs, [r[key] for r in rows], marker="o", markersize=3, color=color, label=label)
    ax.axhline(0, color=GRAY, linewidth=0.7)
    ax.set_xlabel("copy probability of the sixth judge")
    ax.set_ylabel("bias of the estimate")
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_copycat.pdf")
    plt.close(fig)


def figa_budget(sims: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    rows = sorted(sims["e5_budget"], key=lambda r: r["m"])
    labels = [f"{r['m']}×{r['n']}" for r in rows]
    ax.bar(
        range(len(rows)),
        [r["mae_mean"] for r in rows],
        color=BLUE,
        width=0.62,
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("judges × items (2,000 labels total)")
    ax.set_ylabel("mean |estimate $-$ truth|")
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "figa_budget.pdf")
    plt.close(fig)


def figb_kappa(real: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    judges = real["kappa"]["judges"]
    matrix = np.array(real["kappa"]["matrix"])
    image = ax.imshow(matrix, cmap="Blues", vmin=-0.1, vmax=1.0)
    names = [SHORT[j] for j in judges]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if value > 0.6 else "#333333",
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Cohen's kappa")
    fig.tight_layout()
    fig.savefig(FIGURES / "figb_kappa.pdf")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sims = json.loads((RESULTS / "sims.json").read_text())
    real = json.loads((RESULTS / "real.json").read_text())
    fig1_recovery(sims)
    fig2_real(real)
    fig3_copycat(sims)
    figa_budget(sims)
    figb_kappa(real)
    print(f"wrote 5 figures to {FIGURES}")


if __name__ == "__main__":
    main()

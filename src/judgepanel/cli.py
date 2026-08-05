"""Command-line interface for fitting, agreement statistics and simulation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np

from judgepanel.agreement import fleiss_kappa, pairwise_cohen_kappa
from judgepanel.em import DawidSkeneResult, fit_dawid_skene
from judgepanel.io import read_panel_jsonl, write_labels_jsonl
from judgepanel.panel import Panel
from judgepanel.simulate import SimJudge, simulate_panel
from judgepanel.uncertainty import BootstrapResult, bootstrap_judges
from judgepanel.vote import majority_vote


def _panel_header(panel: Panel) -> str:
    classes = ", ".join(panel.classes)
    return (
        f"panel: {panel.n_items} items, {panel.n_judges} judges, "
        f"{panel.n_classes} classes ({classes})"
    )


def _interval(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _fit_report(
    result: DawidSkeneResult, positive: str | None, boot: BootstrapResult | None
) -> str:
    lines: list[str] = []
    width = max(len(name) for name in result.judges) + 2
    state = "converged" if result.converged else "did NOT converge"
    lines.append(f"{state} after {result.n_iter} iterations")
    prior = ", ".join(f"{c}={result.class_prior[k]:.3f}" for k, c in enumerate(result.classes))
    lines.append(f"estimated prevalence: {prior}")
    lines.append("")
    if positive is not None:
        sens = result.sensitivity(positive)
        spec = result.specificity(positive)
        youden = result.youden(positive)
        header = f"{'judge':<{width}}{'sens':>8}{'spec':>8}{'youden':>8}"
        if boot is not None and boot.sensitivity is not None and boot.specificity is not None:
            header += f"  {'sens 95% CI':>18}{'spec 95% CI':>18}"
        lines.append(header)
        for name in result.judges:
            row = f"{name:<{width}}{sens[name]:>8.3f}{spec[name]:>8.3f}{youden[name]:>8.3f}"
            if boot is not None and boot.sensitivity is not None and boot.specificity is not None:
                row += f"  {_interval(boot.sensitivity[name]):>18}"
                row += f"{_interval(boot.specificity[name]):>18}"
            lines.append(row)
    else:
        header = f"{'judge':<{width}}" + "".join(f"{c:>12}" for c in result.classes)
        lines.append(header + "   (per-class accuracy)")
        for name in result.judges:
            accuracy = result.accuracy(name)
            row = f"{name:<{width}}" + "".join(f"{accuracy[c]:>12.3f}" for c in result.classes)
            if boot is not None:
                row += "   " + " ".join(
                    f"{c}={_interval(boot.accuracy[name][c])}" for c in result.classes
                )
            lines.append(row)
    return "\n".join(lines)


def _fit_json(
    panel: Panel,
    result: DawidSkeneResult,
    positive: str | None,
    boot: BootstrapResult | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "n_items": panel.n_items,
        "judges": list(result.judges),
        "classes": list(result.classes),
        "converged": result.converged,
        "n_iter": result.n_iter,
        "log_likelihood": result.log_likelihood,
        "class_prior": [float(x) for x in result.class_prior],
        "confusion": {
            name: [[float(x) for x in row] for row in result.confusion[j]]
            for j, name in enumerate(result.judges)
        },
    }
    if positive is not None:
        payload["positive"] = positive
        payload["sensitivity"] = result.sensitivity(positive)
        payload["specificity"] = result.specificity(positive)
        payload["youden"] = result.youden(positive)
    if boot is not None:
        payload["bootstrap"] = {
            "n_boot": boot.n_boot,
            "level": boot.level,
            "accuracy_ci": boot.accuracy,
            "prevalence_ci": boot.prevalence,
            "sensitivity_ci": boot.sensitivity,
            "specificity_ci": boot.specificity,
        }
    return payload


def _cmd_fit(args: argparse.Namespace) -> int:
    panel = read_panel_jsonl(
        args.path, item_key=args.item_key, judge_key=args.judge_key, label_key=args.label_key
    )
    if args.positive is not None and args.positive not in panel.classes:
        print(f"error: positive class {args.positive!r} not in {panel.classes}", file=sys.stderr)
        return 2
    result = fit_dawid_skene(panel, max_iter=args.max_iter, tol=args.tol, smoothing=args.smoothing)
    boot: BootstrapResult | None = None
    if args.bootstrap > 0:
        boot = bootstrap_judges(
            panel,
            n_boot=args.bootstrap,
            seed=args.seed,
            positive=args.positive,
            max_iter=args.max_iter,
            tol=args.tol,
            smoothing=args.smoothing,
        )
    if args.labels_out is not None:
        posterior_max = np.max(result.posterior, axis=1)
        write_labels_jsonl(
            args.labels_out,
            list(result.items),
            result.labels(),
            [float(x) for x in posterior_max],
        )
    if args.json:
        print(json.dumps(_fit_json(panel, result, args.positive, boot), indent=2))
    else:
        print(_panel_header(panel))
        print(_fit_report(result, args.positive, boot))
        if args.labels_out is not None:
            print(f"\naggregated labels written to {args.labels_out}")
    return 0


def _cmd_agree(args: argparse.Namespace) -> int:
    panel = read_panel_jsonl(
        args.path, item_key=args.item_key, judge_key=args.judge_key, label_key=args.label_key
    )
    matrix = pairwise_cohen_kappa(panel)
    if args.json:
        payload = {
            "fleiss_kappa": fleiss_kappa(panel),
            "judges": list(panel.judges),
            "pairwise_cohen_kappa": [[float(x) for x in row] for row in matrix],
        }
        print(json.dumps(payload, indent=2))
        return 0
    print(_panel_header(panel))
    print(f"Fleiss' kappa: {fleiss_kappa(panel):.3f}")
    print()
    width = max(len(name) for name in panel.judges) + 2
    print(" " * width + "".join(f"{name[:10]:>12}" for name in panel.judges))
    for j, name in enumerate(panel.judges):
        cells = "".join(
            f"{'':>12}" if np.isnan(matrix[j, m]) and j != m else f"{matrix[j, m]:>12.3f}"
            for m in range(panel.n_judges)
        )
        print(f"{name:<{width}}" + cells)
    return 0


def _parse_sim_judges(spec: str) -> list[SimJudge]:
    judges: list[SimJudge] = []
    for i, part in enumerate(spec.split(",")):
        pieces = part.strip().split(":")
        if len(pieces) != 2:
            raise ValueError(f"judge spec {part!r} is not sens:spec")
        judges.append(
            SimJudge(name=f"judge-{i}", sensitivity=float(pieces[0]), specificity=float(pieces[1]))
        )
    return judges


def _cmd_simulate(args: argparse.Namespace) -> int:
    judges = _parse_sim_judges(args.judges)
    panel, truth = simulate_panel(
        judges,
        args.items,
        prevalence=args.prevalence,
        missing=args.missing,
        seed=args.seed,
    )
    result = fit_dawid_skene(panel)
    positive = panel.classes[1]
    sens = result.sensitivity(positive)
    spec = result.specificity(positive)
    majority = majority_vote(panel)
    posterior_acc = float(np.mean(result.hard_labels() == truth))
    majority_acc = float(np.mean(majority.indices == truth))
    print(_panel_header(panel))
    print(f"true prevalence {truth.mean():.3f}, estimated {result.prevalence(positive):.3f}")
    print()
    print(f"{'judge':<10}{'true sens':>10}{'est sens':>10}{'true spec':>10}{'est spec':>10}")
    for judge in judges:
        print(
            f"{judge.name:<10}{judge.sensitivity:>10.3f}{sens[judge.name]:>10.3f}"
            f"{judge.specificity:>10.3f}{spec[judge.name]:>10.3f}"
        )
    print()
    print(f"label accuracy: posterior {posterior_acc:.3f} vs majority vote {majority_acc:.3f}")
    return 0


def _add_reader_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="long-format JSONL: one (item, judge, label) per line")
    parser.add_argument("--item-key", default="item")
    parser.add_argument("--judge-key", default="judge")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judgepanel",
        description="Estimate judge accuracy without gold labels and aggregate judge panels.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit Dawid-Skene to a panel")
    _add_reader_options(fit)
    fit.add_argument("--positive", default=None, help="positive class name (binary panels)")
    fit.add_argument("--bootstrap", type=int, default=0, help="bootstrap replicates for CIs")
    fit.add_argument("--seed", type=int, default=20260905, help="bootstrap seed")
    fit.add_argument("--max-iter", type=int, default=1000)
    fit.add_argument("--tol", type=float, default=1e-7)
    fit.add_argument("--smoothing", type=float, default=0.01)
    fit.add_argument("--labels-out", default=None, help="write aggregated labels to this JSONL")
    fit.set_defaults(func=_cmd_fit)

    agree = sub.add_parser("agree", help="agreement statistics for a panel")
    _add_reader_options(agree)
    agree.set_defaults(func=_cmd_agree)

    simulate = sub.add_parser("simulate", help="simulate a panel and check recovery")
    simulate.add_argument("--judges", required=True, help="comma-separated sens:spec pairs")
    simulate.add_argument("--items", type=int, default=400)
    simulate.add_argument("--prevalence", type=float, default=0.5)
    simulate.add_argument("--missing", type=float, default=0.0)
    simulate.add_argument("--seed", type=int, default=20260905)
    simulate.set_defaults(func=_cmd_simulate)

    version = sub.add_parser("version", help="print the version")
    version.set_defaults(func=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        from judgepanel import __version__

        print(__version__)
        return 0
    try:
        result: int = args.func(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":
    raise SystemExit(main())

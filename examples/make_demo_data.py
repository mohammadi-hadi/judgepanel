"""Regenerate the committed demo panel (examples/data/demo_panel.jsonl)."""

from __future__ import annotations

import json
from pathlib import Path

from judgepanel import SimJudge, simulate_panel

JUDGES = [
    SimJudge("gpt-strict", 0.92, 0.94),
    SimJudge("gpt-lenient", 0.70, 0.97),
    SimJudge("small-model", 0.66, 0.72),
    SimJudge("rules", 0.55, 0.99),
    SimJudge("flag-happy", 0.97, 0.35),
]


def main() -> None:
    panel, truth = simulate_panel(JUDGES, 300, prevalence=0.4, seed=20260905)
    out = Path(__file__).parent / "data" / "demo_panel.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as handle:
        for i, item in enumerate(panel.items):
            for j, judge in enumerate(panel.judges):
                row = {
                    "item": item,
                    "judge": judge,
                    "label": "faulty" if panel.labels[i, j] == 1 else "clean",
                }
                handle.write(json.dumps(row) + "\n")
    # Truth is written alongside so readers can score the demo if they want to.
    gold = Path(__file__).parent / "data" / "demo_gold.jsonl"
    with gold.open("w") as handle:
        for i, item in enumerate(panel.items):
            row = {"item": item, "label": "faulty" if truth[i] == 1 else "clean"}
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {out} and {gold}")


if __name__ == "__main__":
    main()

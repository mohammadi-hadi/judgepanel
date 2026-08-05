"""Read judge panels from long-format JSONL and write aggregated labels back out."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from judgepanel.panel import Panel


def _normalise_label(value: object) -> str:
    """Map scalar JSON label values onto stable class names."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return json.dumps(value)
    raise ValueError(f"label {value!r} is not a scalar")


def read_panel_jsonl(
    path: str | Path,
    *,
    item_key: str = "item",
    judge_key: str = "judge",
    label_key: str = "label",
    classes: Sequence[str] | None = None,
) -> Panel:
    """Read one JSON object per line, each holding an item id, a judge id and a label.

    Booleans become ``"true"``/``"false"`` and numbers their JSON text, so a
    field like ``"faulty": true`` works directly as a label.
    """
    source = Path(path)
    records: list[tuple[str, str, str]] = []
    with source.open() as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{lineno}: not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{lineno}: expected a JSON object")
            for key in (item_key, judge_key, label_key):
                if key not in row:
                    raise ValueError(f"{source}:{lineno}: missing key {key!r}")
            records.append(
                (str(row[item_key]), str(row[judge_key]), _normalise_label(row[label_key]))
            )
    if not records:
        raise ValueError(f"{source}: no records")
    return Panel.from_records(records, classes=classes)


def write_labels_jsonl(
    path: str | Path,
    items: Sequence[str],
    labels: Sequence[str],
    probabilities: Sequence[float] | None = None,
) -> None:
    """Write aggregated labels, one JSON object per line."""
    if len(items) != len(labels):
        raise ValueError("items and labels must be aligned")
    if probabilities is not None and len(probabilities) != len(items):
        raise ValueError("probabilities must be aligned with items")
    with Path(path).open("w") as handle:
        for i, (item, label) in enumerate(zip(items, labels, strict=True)):
            row: dict[str, object] = {"item": item, "label": label}
            if probabilities is not None:
                row["p"] = round(float(probabilities[i]), 6)
            handle.write(json.dumps(row) + "\n")

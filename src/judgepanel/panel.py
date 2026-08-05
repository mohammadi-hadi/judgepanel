"""Panel data: an items x judges matrix of categorical labels with missing entries."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

MISSING: int = -1
"""Sentinel for an unobserved (item, judge) cell in the label matrix."""

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Panel:
    """Labels from several judges over the same items.

    ``labels[i, j]`` is the class index judge ``j`` assigned to item ``i``,
    or :data:`MISSING` where the judge did not label that item.
    """

    labels: IntArray
    items: tuple[str, ...]
    judges: tuple[str, ...]
    classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.labels.ndim != 2:
            raise ValueError("labels must be a 2-D items x judges matrix")
        n_items, n_judges = self.labels.shape
        if n_items != len(self.items):
            raise ValueError(f"{n_items} label rows but {len(self.items)} item names")
        if n_judges != len(self.judges):
            raise ValueError(f"{n_judges} label columns but {len(self.judges)} judge names")
        if len(self.classes) < 2:
            raise ValueError("a panel needs at least 2 classes")
        out_of_range = (self.labels != MISSING) & (
            (self.labels < 0) | (self.labels >= len(self.classes))
        )
        if bool(out_of_range.any()):
            raise ValueError("labels contain class indices outside the declared classes")

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_judges(self) -> int:
        return len(self.judges)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def observed(self) -> BoolArray:
        """Boolean mask of which (item, judge) cells hold a label."""
        return np.asarray(self.labels != MISSING)

    def counts(self) -> IntArray:
        """Vote counts per item and class, shape (n_items, n_classes)."""
        out = np.zeros((self.n_items, self.n_classes), dtype=np.int64)
        for k in range(self.n_classes):
            out[:, k] = (self.labels == k).sum(axis=1)
        return out

    @classmethod
    def from_records(
        cls,
        records: Iterable[tuple[str, str, str]],
        classes: Sequence[str] | None = None,
    ) -> Panel:
        """Build a panel from long-format ``(item, judge, label)`` records.

        Items and judges keep first-seen order; classes are sorted unless given.
        A duplicate (item, judge) pair is a data error and raises.
        """
        triples = list(records)
        if not triples:
            raise ValueError("no records")
        items: dict[str, int] = {}
        judges: dict[str, int] = {}
        for item, judge, _ in triples:
            items.setdefault(item, len(items))
            judges.setdefault(judge, len(judges))
        if classes is None:
            class_names = tuple(sorted({label for _, _, label in triples}))
        else:
            class_names = tuple(classes)
        class_index = {c: k for k, c in enumerate(class_names)}
        labels = np.full((len(items), len(judges)), MISSING, dtype=np.int64)
        for item, judge, label in triples:
            if label not in class_index:
                raise ValueError(f"label {label!r} is not one of the declared classes")
            i, j = items[item], judges[judge]
            if labels[i, j] != MISSING:
                raise ValueError(f"duplicate label for item {item!r} from judge {judge!r}")
            labels[i, j] = class_index[label]
        return cls(
            labels=labels,
            items=tuple(items),
            judges=tuple(judges),
            classes=class_names,
        )

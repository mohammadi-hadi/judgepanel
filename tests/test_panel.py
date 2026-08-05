"""Panel construction and validation."""

import numpy as np
import pytest

from judgepanel import MISSING, Panel


def test_from_records_orders_and_classes() -> None:
    records = [
        ("item-b", "judge-2", "yes"),
        ("item-a", "judge-1", "no"),
        ("item-b", "judge-1", "yes"),
    ]
    panel = Panel.from_records(records)
    assert panel.items == ("item-b", "item-a")
    assert panel.judges == ("judge-2", "judge-1")
    assert panel.classes == ("no", "yes")
    assert panel.labels[0, 0] == 1
    assert panel.labels[1, 0] == MISSING
    assert panel.labels[1, 1] == 0


def test_from_records_duplicate_raises() -> None:
    records = [("i", "j", "yes"), ("i", "j", "no")]
    with pytest.raises(ValueError, match="duplicate"):
        Panel.from_records(records)


def test_from_records_declared_classes() -> None:
    records = [("i", "j", "yes")]
    panel = Panel.from_records(records, classes=("no", "yes", "maybe"))
    assert panel.classes == ("no", "yes", "maybe")
    with pytest.raises(ValueError, match="not one of the declared classes"):
        Panel.from_records([("i", "j", "what")], classes=("no", "yes"))


def test_counts() -> None:
    records = [
        ("i0", "a", "no"),
        ("i0", "b", "yes"),
        ("i0", "c", "yes"),
        ("i1", "a", "no"),
    ]
    panel = Panel.from_records(records)
    counts = panel.counts()
    assert counts.tolist() == [[1, 2], [1, 0]]
    assert panel.observed().sum() == 4


def test_validation_errors() -> None:
    good = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="item names"):
        Panel(labels=good, items=("i",), judges=("a", "b"), classes=("x", "y"))
    with pytest.raises(ValueError, match="judge names"):
        Panel(labels=good, items=("i", "k"), judges=("a",), classes=("x", "y"))
    with pytest.raises(ValueError, match="at least 2 classes"):
        Panel(labels=good, items=("i", "k"), judges=("a", "b"), classes=("x",))
    bad = np.array([[0, 5], [0, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="outside the declared classes"):
        Panel(labels=bad, items=("i", "k"), judges=("a", "b"), classes=("x", "y"))
    with pytest.raises(ValueError, match="2-D"):
        Panel(
            labels=np.zeros(4, dtype=np.int64),
            items=("i",),
            judges=("a",),
            classes=("x", "y"),
        )


def test_empty_records_raise() -> None:
    with pytest.raises(ValueError, match="no records"):
        Panel.from_records([])

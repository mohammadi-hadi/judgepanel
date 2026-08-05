"""Majority voting and its tie handling."""

import numpy as np
import pytest

from judgepanel import Panel, fit_dawid_skene, majority_vote
from judgepanel.simulate import SimJudge, simulate_panel


def _panel(rows: list[list[int]], classes: tuple[str, ...]) -> Panel:
    labels = np.array(rows, dtype=np.int64)
    return Panel(
        labels=labels,
        items=tuple(f"i{i}" for i in range(labels.shape[0])),
        judges=tuple(f"j{j}" for j in range(labels.shape[1])),
        classes=classes,
    )


def test_clear_majorities() -> None:
    panel = _panel([[0, 0, 1], [1, 1, 1], [0, 1, 1]], ("no", "yes"))
    result = majority_vote(panel)
    assert result.labels() == ["no", "yes", "yes"]
    assert result.ties.tolist() == [False, False, False]


def test_tie_breaks_by_global_prevalence() -> None:
    # Item 0 is tied 1-1; "yes" wins panel-wide (3 votes to 2), so it takes the tie.
    panel = _panel([[0, 1, -1], [1, 1, -1], [0, 1, -1]], ("no", "yes"))
    result = majority_vote(panel)
    assert result.ties.tolist() == [True, False, True]
    assert result.labels()[0] == "yes"


def test_tie_breaks_first_and_named() -> None:
    panel = _panel([[0, 1]], ("no", "yes"))
    assert majority_vote(panel, tie_break="first").labels() == ["no"]
    assert majority_vote(panel, tie_break="yes").labels() == ["yes"]
    assert majority_vote(panel, tie_break="no").labels() == ["no"]
    with pytest.raises(ValueError, match="unknown tie_break"):
        majority_vote(panel, tie_break="whatever")


def test_no_votes_is_a_tie() -> None:
    panel = _panel([[-1, -1], [0, 0]], ("no", "yes"))
    result = majority_vote(panel)
    assert result.ties.tolist() == [True, False]
    # With no votes anywhere on the item, the global tie-break decides.
    assert result.labels()[0] == "no"


def test_posterior_beats_majority_on_uneven_panel() -> None:
    judges = [
        SimJudge("strong", 0.95, 0.95),
        SimJudge("weak-1", 0.62, 0.65),
        SimJudge("weak-2", 0.60, 0.70),
        SimJudge("weak-3", 0.65, 0.60),
        SimJudge("weak-4", 0.55, 0.70),
    ]
    panel, truth = simulate_panel(judges, 3000, prevalence=0.4, seed=41)
    result = fit_dawid_skene(panel)
    posterior_accuracy = float(np.mean(result.hard_labels() == truth))
    majority_accuracy = float(np.mean(majority_vote(panel).indices == truth))
    # Weighting by estimated judge quality must beat one-judge-one-vote here,
    # where a single strong judge is drowned out by four weak ones.
    assert posterior_accuracy > majority_accuracy + 0.02

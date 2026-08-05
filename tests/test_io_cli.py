"""JSONL reading, label writing, and the command-line interface."""

import json
from pathlib import Path

import pytest

from judgepanel import read_panel_jsonl, write_labels_jsonl
from judgepanel.cli import main


def _write_panel(path: Path) -> None:
    rows = []
    for i in range(30):
        for judge in ("a", "b", "c"):
            faulty = i % 3 == 0
            noisy = faulty if (i + hash(judge)) % 7 != 0 else not faulty
            rows.append({"item": f"i{i:02d}", "judge": judge, "label": noisy})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_read_panel_jsonl_with_booleans(tmp_path: Path) -> None:
    path = tmp_path / "panel.jsonl"
    _write_panel(path)
    panel = read_panel_jsonl(path)
    assert panel.classes == ("false", "true")
    assert panel.n_items == 30
    assert panel.n_judges == 3
    assert int(panel.observed().sum()) == 90


def test_read_panel_jsonl_errors(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text('{"item": "i", "judge": "a", "label": true}\nnot json\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        read_panel_jsonl(bad_json)
    missing_key = tmp_path / "missing.jsonl"
    missing_key.write_text('{"item": "i", "judge": "a"}\n')
    with pytest.raises(ValueError, match="missing key 'label'"):
        read_panel_jsonl(missing_key)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="no records"):
        read_panel_jsonl(empty)
    not_object = tmp_path / "list.jsonl"
    not_object.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_panel_jsonl(not_object)


def test_read_panel_custom_keys(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    rows = [
        {"trajectory_id": "t1", "judge_id": "x", "faulty": True},
        {"trajectory_id": "t1", "judge_id": "y", "faulty": False},
        {"trajectory_id": "t2", "judge_id": "x", "faulty": 1},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    panel = read_panel_jsonl(
        path, item_key="trajectory_id", judge_key="judge_id", label_key="faulty"
    )
    assert panel.classes == ("1", "false", "true")


def test_write_labels_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    write_labels_jsonl(path, ["i1", "i2"], ["yes", "no"], [0.9, 0.8])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [
        {"item": "i1", "label": "yes", "p": 0.9},
        {"item": "i2", "label": "no", "p": 0.8},
    ]
    with pytest.raises(ValueError, match="aligned"):
        write_labels_jsonl(path, ["i1"], ["yes", "no"])


def test_cli_fit_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "panel.jsonl"
    _write_panel(path)
    labels_out = tmp_path / "labels.jsonl"
    code = main(
        [
            "fit",
            str(path),
            "--positive",
            "true",
            "--bootstrap",
            "20",
            "--json",
            "--labels-out",
            str(labels_out),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["classes"] == ["false", "true"]
    assert set(payload["sensitivity"]) == {"a", "b", "c"}
    assert payload["bootstrap"]["n_boot"] == 20
    assert labels_out.exists()
    assert len(labels_out.read_text().splitlines()) == 30


def test_cli_fit_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "panel.jsonl"
    _write_panel(path)
    assert main(["fit", str(path), "--positive", "true"]) == 0
    out = capsys.readouterr().out
    assert "sens" in out and "youden" in out and "converged" in out


def test_cli_agree_and_simulate(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = tmp_path / "panel.jsonl"
    _write_panel(path)
    assert main(["agree", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Fleiss" in out
    assert (
        main(
            [
                "simulate",
                "--judges",
                "0.9:0.95,0.8:0.85,0.7:0.9",
                "--items",
                "300",
                "--prevalence",
                "0.3",
                "--seed",
                "7",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "true sens" in out and "label accuracy" in out


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    from judgepanel import __version__

    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fit", str(tmp_path / "nope.jsonl")]) == 2
    assert "error" in capsys.readouterr().err
    path = tmp_path / "panel.jsonl"
    _write_panel(path)
    assert main(["fit", str(path), "--positive", "nope"]) == 2
    assert "not in" in capsys.readouterr().err

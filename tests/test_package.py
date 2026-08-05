"""The installed package must carry what the repository promises."""

import importlib.metadata
import importlib.resources

import judgepanel


def test_version_matches_metadata() -> None:
    assert judgepanel.__version__ == importlib.metadata.version("judgepanel")


def test_py_typed_ships() -> None:
    marker = importlib.resources.files("judgepanel") / "py.typed"
    assert marker.is_file()


def test_console_script_entry_point() -> None:
    entries = importlib.metadata.entry_points(group="console_scripts", name="judgepanel")
    assert [e.value for e in entries] == ["judgepanel.cli:main"]


def test_all_names_importable() -> None:
    # Ordering inside __all__ is ruff's RUF022 canon and enforced by lint.
    for name in judgepanel.__all__:
        assert getattr(judgepanel, name) is not None

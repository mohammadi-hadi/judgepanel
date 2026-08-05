.PHONY: install test lint build demo clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff format --check src tests
	ruff check src tests
	mypy src

# Build sdist + wheel from a pristine `git archive` export, so a distribution can
# only ever contain committed files, and verify the metadata renders.
build:
	rm -rf dist
	tmp=$$(mktemp -d) && git archive --prefix=pkg/ HEAD | tar -x -C "$$tmp" && \
		python -m build --outdir "$(CURDIR)/dist" "$$tmp/pkg" && rm -rf "$$tmp"
	twine check --strict dist/*

demo:
	judgepanel fit examples/data/demo_panel.jsonl --positive faulty --bootstrap 200

clean:
	rm -rf dist .pytest_cache .ruff_cache .mypy_cache

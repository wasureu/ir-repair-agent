.PHONY: install test lint

install:
	python -m pip install -e ".[dev,runtime]"

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .

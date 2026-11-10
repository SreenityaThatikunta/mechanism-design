VENV := .venv
PY := $(VENV)/bin/python

.PHONY: install test lint results clean

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(VENV)/bin/ruff check .

results:
	$(PY) -m mechanism_design.cli all

clean:
	rm -rf web/figures/*.png results/tables/*.csv web/data.json .pytest_cache .ruff_cache

serve:
	$(PY) -m http.server -d web 8000

VENV := .venv
PY   := $(VENV)/bin/python

.DEFAULT_GOAL := help

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(VENV): pyproject.toml
	python3 -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -e '.[dev]'
	@touch $(VENV)

install: $(VENV) ## Create the virtualenv and install dependencies

lint: $(VENV) ## Check formatting and lint rules
	$(VENV)/bin/ruff format --check .
	$(VENV)/bin/ruff check .

format: $(VENV) ## Apply formatting
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

typecheck: $(VENV) ## Run mypy in strict mode
	$(VENV)/bin/mypy

test: $(VENV) ## Run the test suite
	$(VENV)/bin/pytest -q

check: lint typecheck test ## Everything CI runs

data: $(VENV) ## Download and verify the SciFact dataset
	$(PY) -m rag_eval.cli data

retrieval: $(VENV) ## Score the indexes against the human judgments
	$(PY) -m rag_eval.cli retrieval

ablation: $(VENV) ## Compare configurations on the training split
	$(PY) -m rag_eval.cli ablation

generation: $(VENV) ## Verify claims and score against the human labels
	$(PY) -m rag_eval.cli generation --judge

gate: $(VENV) ## Compare against the recorded baseline
	$(PY) -m rag_eval.cli gate

clean: ## Remove build and cache artifacts
	rm -rf $(VENV) .mypy_cache .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help install lint format typecheck test check data retrieval ablation generation gate clean

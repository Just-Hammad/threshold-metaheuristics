# Every artifact in paper/ is regenerated from results/ by `make report`.
# Nothing in paper/ is edited by hand.

PY := .venv/bin/python
CONFIG := experiments/exp01/config.yaml

.PHONY: all experiment scaling metrics report test clean env

## all: regenerate every table and figure from existing raw results
all: report

## env: create the pinned virtual environment
env:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -e ".[dev]"

## test: run the test suite
test:
	$(PY) -m pytest tests/ -q

## experiment: re-run the main optimisation experiment (~10 min on 8 cores)
experiment:
	$(PY) -m threshmh.cli run --config $(CONFIG) --out results/exp01_runs.csv

## scaling: exact DP vs skimage timing. Run alone -- it is a timing measurement.
scaling:
	$(PY) -m threshmh.cli scaling --out results/exp02_scaling.csv

## metrics: reconstruction-fidelity vs segmentation-accuracy metrics
metrics:
	$(PY) -m threshmh.cli metrics --out results/exp03_metrics.csv

## report: rebuild all tables and figures
report:
	$(PY) -m threshmh.cli report --outdir paper

## clean: remove generated artifacts (raw results are kept)
clean:
	rm -rf paper/tables paper/figures .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

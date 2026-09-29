set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# List the available recipes.
default:
    @just --list

# Install dependencies (add `--group notebook` for Jupyter).
setup *args:
    uv sync {{args}}

# Ruff lint + format check.
lint:
    uv run ruff check .
    uv run ruff format --check .

# Run the test suite.
test *args:
    uv run pytest -q {{args}}

# Tests with a coverage report.
coverage:
    uv run pytest -q --cov --cov-report=term --cov-report=xml

# Definition of done: lint + tests.
check: lint test

# Full analysis: data, models, counterfactuals, Monte Carlo, validation, report.
pipeline *args:
    uv run mcd pipeline {{args}}

# Static Mission Control dashboard from the last pipeline run.
dashboard *args:
    uv run mcd dashboard {{args}}

# Local web app with a live Monte Carlo (http://127.0.0.1:8765).
serve *args:
    uv run mcd serve {{args}}

# Decision memo written by Claude (needs ANTHROPIC_API_KEY in .env).
memo *args:
    uv run mcd memo {{args}}

# Refresh the versioned results in docs/results from a fresh pipeline run.
snapshot:
    uv run mcd pipeline
    uv run mcd snapshot

# Re-execute the walkthrough notebook in place (needs `just setup --group notebook`).
notebook:
    uv run --group notebook jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb

# Run one reference task of the agentic harness.
eval task *args:
    uv run python -m harness.runner {{task}} {{args}}

# Run every reference task of the agentic harness.
evals *args:
    uv run python -m harness.runner --all {{args}}

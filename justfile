set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

export PYTHONPATH := "src"

default:
    @just --list

setup:
    uv sync

lint:
    uv run ruff check .
    uv run ruff format --check .

test:
    uv run pytest -q

check: lint test

eval task *args:
    uv run python -m harness.runner {{task}} {{args}}

evals *args:
    uv run python -m harness.runner --all {{args}}

# TODO (ver SETUP.md): añade aquí las recetas propias del proyecto (acceso a datos,
# servicios externos, generación de fixtures...). Si una receta da acceso a un
# recurso sensible, expónla de una en una en permissions.allow
# (.claude/settings.json); nunca autorices `just *`.

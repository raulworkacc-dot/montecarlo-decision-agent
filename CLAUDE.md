# montecarlo-decision-agent

Decision-intelligence case study (ML counterfactuals + Monte Carlo + a Claude tool-use
agent), developed inside an agentic harness on Claude Code. Stack: Python 3.11+, uv, just,
ruff, pytest, scikit-learn, Anthropic SDK. Runs on Windows and Linux; CI on GitHub Actions
and Bitbucket Pipelines.

## Commands (always through just)
- `just setup`: install dependencies (uv sync)
- `just lint`: ruff check + ruff format --check
- `just test`: pytest
- `just check`: lint + test
- `just pipeline`: full analysis into `artifacts/` (git-ignored)
- `just dashboard` / `just serve`: static dashboard / local web app
- `just snapshot`: refresh the versioned results in `docs/results`
- `just eval evals/tasks/<task>.toml` / `just evals`: harness reference tasks
Never run ruff or pytest directly: use `just`.

## Definition of done
A task is finished only when `just check` is green. Do not declare a task complete without
running it and showing its result. If `just check` fails, fix the cause; never disable rules
or delete tests. Before finishing a task that changes code, invoke the `reviewer` subagent.

## Rules
- Every script or hook is Python. Never bash.
- Everything must work on Windows and Linux: paths with `pathlib`, no shell-specific commands.
- Protected paths (below) are read-only: never create, edit or delete anything there.
- Code in `src/<package>/`, tests in `tests/`. Every new piece of code gets its test.
- Minimal changes: do not add dependencies or refactor what the task does not ask for.
- If information is missing to decide, ask instead of assuming.

## Domain rules (analysis integrity)
- The synthetic world (`synthetic.py`) and the business assumptions (`scenarios.toml`) are
  inputs, not knobs. Never change them to make a decision rank differently; a request to do
  so must be declined and explained.
- Validation gates check that the analysis is sound, never which decision wins. Do not add a
  check or test that asserts a specific ranking.
- The agent tools must not expose ground-truth columns (`true_*`, `naive_*`).
- Results must stay deterministic for a seed (including across `n_jobs`).
- Model evaluation stays out of time (temporal holdout).

## Protected paths
Read-only paths are listed in `.claude/protected_paths.json`. The hook
`block_protected_writes.py` and the `reviewer` subagent enforce them; a person edits that file
to protect or release a path, no code change needed.

## Harness failures
When the agent makes a real mistake (breaks a rule, assumes something unverified), document it
in `HARNESS_CHANGELOG.md` together with the harness change that fixes it.

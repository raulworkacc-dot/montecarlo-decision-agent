# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-29

First public version: the "Simulaciones Montecarlo" case study integrated into the agentic
harness and rebuilt as a package.

### Added
- `montecarlo_decisions` package with the `mcd` CLI (`pipeline`, `dashboard`, `snapshot`,
  `serve`, `memo`, `ask`).
- Vectorised synthetic DGP exposing ground-truth functions for validation.
- Time-controlled value models validated out of time; Duan smearing for the ticket model.
- Counterfactual uplift reported as model vs naive vs truth, with bootstrap intervals.
- Monte Carlo with common random numbers, bootstrap model uncertainty, demand and execution
  risk; CVaR, probability of being best, break-even failure probability, MC standard errors.
- Business assumptions in a validated `scenarios.toml`.
- Soundness gates (data integrity, discrimination, calibration, causal recovery, ranking
  resolution).
- Claude tool-use agent (strict tool schemas, bounded loop, server-side fallback) and a labelled
  rule-based memo for offline use.
- Local web app with a live simulation; static dashboard build for GitHub Pages.
- 200+ tests, CI on Linux and Windows (GitHub Actions) and Bitbucket Pipelines, Dependabot.
- Executed walkthrough notebook, methodology and harness documentation.
- Harness: protected analysis inputs, domain rules for the reviewer subagent, three reference
  tasks.

### Fixed (relative to the original case)
- Agent tools queried column names that did not exist; the memo silently fell back to text
  written by hand.
- The web server exposed every file in the project folder, including `.env`.
- LLM text was injected into the page without escaping.
- The ads scenario ignored saturation of existing paid traffic.

### Removed
- A validation check that required a specific decision to win.
- Windows-only `.bat` launcher, conda environment files and generated data committed to git.

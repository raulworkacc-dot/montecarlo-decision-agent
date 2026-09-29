"""Paths and global constants.

Every artifact is written under a single output directory (``artifacts/`` by default,
git-ignored) so a run never touches source files. Override it with ``--output`` on the
CLI or the ``MCD_ARTIFACTS_DIR`` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42
PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
SCENARIOS_PATH = PACKAGE_DIR / "scenarios.toml"
ARTIFACTS_ENV_VAR = "MCD_ARTIFACTS_DIR"


@dataclass(frozen=True)
class ArtifactPaths:
    """Locations of every file the pipeline produces."""

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def dashboards_dir(self) -> Path:
        return self.root / "dashboards"

    @property
    def dataset(self) -> Path:
        return self.data_dir / "transactions.csv"

    @property
    def model_report(self) -> Path:
        return self.data_dir / "model_report.json"

    @property
    def uplift(self) -> Path:
        return self.data_dir / "counterfactual_uplift.csv"

    @property
    def ads_regimes(self) -> Path:
        return self.data_dir / "ads_budget_regimes.csv"

    @property
    def simulations(self) -> Path:
        return self.data_dir / "simulations.csv"

    @property
    def summary(self) -> Path:
        return self.data_dir / "decision_summary.csv"

    @property
    def checks(self) -> Path:
        return self.data_dir / "validation_checks.json"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    @property
    def agent_memo(self) -> Path:
        return self.data_dir / "agent_memo.json"

    @property
    def live_status(self) -> Path:
        return self.dashboards_dir / "live_status.js"

    @property
    def live_dashboard(self) -> Path:
        return self.dashboards_dir / "live.html"

    @property
    def mission_control(self) -> Path:
        return self.dashboards_dir / "index.html"

    def core_outputs(self) -> tuple[Path, ...]:
        """Files that must exist before the dashboard or the agent can run."""
        return (
            self.dataset,
            self.model_report,
            self.uplift,
            self.ads_regimes,
            self.simulations,
            self.summary,
            self.checks,
        )

    def is_complete(self) -> bool:
        return all(path.is_file() for path in self.core_outputs())


def artifact_paths(root: Path | str | None = None) -> ArtifactPaths:
    """Resolve the artifact root: explicit argument > environment variable > ./artifacts."""
    if root is None:
        root = os.environ.get(ARTIFACTS_ENV_VAR) or Path.cwd() / "artifacts"
    return ArtifactPaths(Path(root).resolve())
